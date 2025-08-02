# file: nmt_translator.py

import os
import pickle
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

# --- Data Loading ---
def load_and_prep_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads and preprocesses the parallel corpus from a CSV file.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file at {filepath} was not found.")
        return None
    df = pd.read_csv(filepath)
    def clean_text(text: str) -> str:
        """Clean and normalize text by removing HTML tags and extra whitespace."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    df['english_cleaned'] = df['english'].apply(clean_text)
    df['nagamese_cleaned'] = df['nagamese'].apply(clean_text)
    # Simple split by space for tokenization
    df['english_tokens'] = df['english_cleaned'].apply(lambda x: x.split())
    df['nagamese_tokens'] = df['nagamese_cleaned'].apply(lambda x: x.split())
    return df

# --- Vocabulary and Dataset ---
class Vocab:
    """Vocabulary class for mapping between tokens and indices."""
    
    def __init__(self, tokens: List[List[str]], min_freq: int = 2) -> None:
        self.pad_token, self.sos_token, self.eos_token, self.unk_token = '<pad>', '<sos>', '<eos>', '<unk>'
        self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx = 0, 1, 2, 3

        specials = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        counter = Counter(tok for seq in tokens for tok in seq)
        vocab = sorted([tok for tok, freq in counter.items() if freq >= min_freq])

        self.idx_to_token = specials + vocab
        self.token_to_idx = {tok: idx for idx, tok in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

class TranslationDataset(Dataset):
    """PyTorch Dataset for translation data."""
    
    def __init__(self, df: pd.DataFrame, src_vocab: Vocab, tgt_vocab: Vocab) -> None:
        self.src_sents = df['nagamese_tokens'].tolist()
        self.tgt_sents = df['english_tokens'].tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src_tokens = [self.src_vocab.token_to_idx.get(tok, self.src_vocab.unk_idx) for tok in self.src_sents[idx]]
        tgt_tokens = [self.tgt_vocab.token_to_idx.get(tok, self.tgt_vocab.unk_idx) for tok in self.tgt_sents[idx]]
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

# --- Model Components ---
class Encoder(nn.Module):
    """Encoder module for the sequence-to-sequence model."""
    
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim * 2]
        # hidden = [n_layers * 2, batch_size, hid_dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # hidden = [batch_size, hid_dim]
        return outputs, hidden

class Attention(nn.Module):
    """Attention mechanism for the decoder."""
    
    def __init__(self, hid_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim) # hid_dim * 2 (encoder) + hid_dim (decoder)
        self.v = nn.Parameter(torch.rand(hid_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch_size, hid_dim]
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch_size, src_len, hid_dim]
        # encoder_outputs = [batch_size, src_len, hid_dim * 2]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        # energy = [batch_size, src_len, hid_dim]
        energy = energy.permute(0, 2, 1)
        # energy = [batch_size, hid_dim, src_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch_size, 1, hid_dim]
        attention = torch.bmm(v, energy).squeeze(1)
        # attention = [batch_size, src_len]
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    """Decoder module with attention for the sequence-to-sequence model."""
    
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hid_dim: int,
        dropout: float,
        attention: Attention
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim * 2 + emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with attention."""
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

def collate_fn(
    batch: List[Tuple[List[str], List[str]]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences, adds SOS/EOS, and moves tensors to the correct device."""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.cat([torch.tensor([src_vocab.sos_idx]), src_sample, torch.tensor([src_vocab.eos_idx])], dim=0))
        tgt_batch.append(torch.cat([torch.tensor([tgt_vocab.sos_idx]), tgt_sample, torch.tensor([tgt_vocab.eos_idx])], dim=0))

    src_padded = pad_sequence(src_batch, padding_value=src_vocab.pad_idx)
    tgt_padded = pad_sequence(tgt_batch, padding_value=tgt_vocab.pad_idx)
    return src_padded.to(device), tgt_padded.to(device)

def train_model(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> float:
    """Main training loop for one epoch."""
    model.train()
    epoch_loss = 0
    for src, trg in loader:
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        # Flatten the output and target tensors
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # Clip gradients
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

class Translator:
    """
    A neural machine translation model for translating Nagamese to English.
    
    This class handles the loading and inference of a sequence-to-sequence model
    with attention mechanism for translating from Nagamese to English.
    
    Example:
        >>> translator = Translator(
        ...     model_path='nmt_model.pt',
        ...     vocabs_path='vocabs.pkl',
        ...     device='cuda' if torch.cuda.is_available() else 'cpu'
        ... )
        >>> translation = translator.translate("moi school jai")
        >>> print(translation)
        'I go to school'
    """
    
    def __init__(self, model_path: str, vocabs_path: str, device: Optional[str] = None) -> None:
        """Initialize the translator with a pre-trained model.
        
        Args:
            model_path: Path to the trained model weights (.pt file).
            vocabs_path: Path to the saved vocabulary files (.pkl file).
            device: Device to run the model on ('cuda' or 'cpu').
                   If None, automatically selects GPU if available.
        
        Raises:
            FileNotFoundError: If model or vocabulary files are not found.
            RuntimeError: If there's an error loading the model or vocabularies.
            ValueError: If the model architecture doesn't match the expected format.
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Validate and load model files
        self._validate_paths(model_path, vocabs_path)
        self._load_vocabs(vocabs_path)
        self._initialize_model(model_path)
    
    def _validate_paths(self, model_path: str, vocabs_path: str) -> None:
        """Validate that the required files exist.
        
        Args:
            model_path: Path to the model file.
            vocabs_path: Path to the vocabulary file.
            
        Raises:
            FileNotFoundError: If either file is not found.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.isfile(vocabs_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {vocabs_path}")
    
    def _load_vocabs(self, vocabs_path: str) -> None:
        """Load source and target language vocabularies.
        
        Args:
            vocabs_path: Path to the pickled vocabulary file.
            
        Raises:
            RuntimeError: If there's an error loading the vocabularies.
        """
        try:
            with open(vocabs_path, 'rb') as f:
                self.src_vocab, self.tgt_vocab = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary files: {e}") from e
    
    def _initialize_model(self, model_path: str) -> None:
        """Initialize the model architecture and load weights.
        
        Args:
            model_path: Path to the trained model weights.
            
        Raises:
            RuntimeError: If there's an error initializing the model.
        """
        try:
            # Model hyperparameters (should match training config)
            enc_emb_dim = 256
            dec_emb_dim = 256
            hid_dim = 512
            enc_dropout = 0.5
            dec_dropout = 0.5
            
            # Initialize model components
            enc = Encoder(
                input_dim=len(self.src_vocab),
                emb_dim=enc_emb_dim,
                hid_dim=hid_dim,
                dropout=enc_dropout
            )
            
            attn = Attention(hid_dim)
            
            dec = Decoder(
                output_dim=len(self.tgt_vocab),
                emb_dim=dec_emb_dim,
                hid_dim=hid_dim,
                dropout=dec_dropout,
                attention=attn
            )
            
            self.model = Seq2Seq(enc, dec, self.device).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}") from e
    
    def translate(self, sentence: str, max_len: int = 50) -> str:
        """Translate a Nagamese sentence to English.
        
        Args:
            sentence: The input sentence in Nagamese.
            max_len: Maximum length of the output sequence. 
                   Defaults to 50.
    
        Returns:
            The translated English sentence. Returns an empty string if input is empty after stripping.
            
        Raises:
            ValueError: If the input is not a string.
            RuntimeError: If translation fails for any reason.
            
        Example:
            >>> translator = Translator('model.pt', 'vocabs.pkl')
            >>> translation = translator.translate("moi school te jai thake")
            >>> print(translation)
            'I go to school'
            
            >>> translator.translate("   ")
            ''
        """
        if not isinstance(sentence, str):
            raise ValueError(f"Input must be a string, got {type(sentence).__name__}")
            
        # Handle empty input gracefully
        sentence = sentence.strip()
        if not sentence:
            return ""
            
        try:
            # Tokenize and numericalize
            tokens = sentence.lower().split()
            src_indexes = [
                self.src_vocab.token_to_idx.get(token, self.src_vocab.unk_idx)
                for token in tokens
            ]
            
            # Convert to tensor and add batch dimension
            src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(self.device)
            
            # Run through model
            with torch.no_grad():
                encoder_outputs, hidden = self.model.encoder(src_tensor)
                
            trg_indexes = [self.tgt_vocab.sos_idx]
            
            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).unsqueeze(0).to(self.device)  # [1, 1]
                with torch.no_grad():
                    output, hidden = self.model.decoder(
                        trg_tensor, hidden, encoder_outputs
                    )
                    
                # Get the most likely next token
                output = output.squeeze(0)  # Remove sequence length dimension if present
                if output.dim() > 1:
                    output = output[-1]  # Take the last output if multiple
                pred_token = output.argmax().item()
                trg_indexes.append(pred_token)
                
                if pred_token == self.tgt_vocab.eos_idx:
                    break
            
            # Convert indices to tokens
            trg_tokens = [self.tgt_vocab.idx_to_token[i] for i in trg_indexes]
            
            # Remove <sos> and <eos> tokens and join
            return ' '.join(trg_tokens[1:-1])
            
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}") from e
    
    def __repr__(self) -> str:
        """Return a string representation of the translator.
        
        Returns:
            A string representing the translator instance with device and vocabulary sizes.
        """
        src_size = len(self.src_vocab) if hasattr(self, 'src_vocab') else 0
        tgt_size = len(self.tgt_vocab) if hasattr(self, 'tgt_vocab') else 0
        return (
            f"Translator(device='{self.device}', "
            f"src_vocab_size={src_size}, tgt_vocab_size={tgt_size})"
        )

def main() -> None:
    """Main function to train and test the NMT model."""
    # --- Configuration ---
    n_epochs = 10
    model_path = 'nmt-nagamese-english.pt'
    vocabs_path = 'nmt-vocabs.pkl'
    data_path = 'merged.csv'
    
    # --- Data Loading ---
    print("Loading and preprocessing data...")
    df = load_and_prep_data(data_path)
    
    if df is None:
        print(f"Error: Could not load data from {data_path}")
        return
        
    # --- Build Vocabularies ---
    print("Building vocabularies...")
    src_vocab = Vocab(df['nagamese_tokens'].tolist())
    tgt_vocab = Vocab(df['english_tokens'].tolist())
    
    # Save vocabularies
    try:
        with open(vocabs_path, 'wb') as f:
            pickle.dump((src_vocab, tgt_vocab), f)
        print(f"Vocabularies saved to {vocabs_path}")
    except IOError as e:
        print(f"Error saving vocabularies: {e}")
        return
            
    # --- Create Dataset and DataLoader ---
    dataset = TranslationDataset(df, src_vocab, tgt_vocab)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, src_vocab, tgt_vocab, device)
    )
    
    # --- Model Setup ---
    # Hyperparameters
    enc_emb_dim = 256
    dec_emb_dim = 256
    hid_dim = 512
    enc_dropout = 0.5
    dec_dropout = 0.5
    
    # Initialize model
    enc = Encoder(len(src_vocab), enc_emb_dim, hid_dim, enc_dropout)
    attn = Attention(hid_dim)
    dec = Decoder(len(tgt_vocab), dec_emb_dim, hid_dim, dec_dropout, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = optim.Adam(model.parameters())
    
    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(n_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}")
    
    # Save the model
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except IOError as e:
        print(f"Error saving model: {e}")
        return
    
    # --- Test Translation ---
    try:
        translator = Translator(model_path, vocabs_path)
        test_sentence = "moi school jai"
        translation = translator.translate(test_sentence)
        print(f"\nTest Translation:")
        print(f"Input (Nagamese): {test_sentence}")
        print(f"Output (English): {translation}")
    except Exception as e:
        print(f"Error during translation: {e}")

if __name__ == '__main__':
    main()
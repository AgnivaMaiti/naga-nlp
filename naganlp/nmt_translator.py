# file: nmt_translator.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import random
import os
import pickle
import re
import pandas as pd 

# --- Data Loading ---
def load_and_prep_data(filepath: str):
    """
    Loads and preprocesses the parallel corpus from a CSV file.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file at {filepath} was not found.")
        return None
    df = pd.read_csv(filepath)
    def clean_text(text):
        if not isinstance(text, str): return ""
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
    def __init__(self, tokens, min_freq=2):
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
    def __init__(self, df, src_vocab, tgt_vocab):
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
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
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
    def __init__(self, hid_dim):
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
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
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
    def __init__(self, encoder, decoder, device):
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

def collate_fn(batch, src_vocab, tgt_vocab, device):
    """Pads sequences, adds SOS/EOS, and moves tensors to the correct device."""
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.cat([torch.tensor([src_vocab.sos_idx]), src_sample, torch.tensor([src_vocab.eos_idx])], dim=0))
        tgt_batch.append(torch.cat([torch.tensor([tgt_vocab.sos_idx]), tgt_sample, torch.tensor([tgt_vocab.eos_idx])], dim=0))

    src_padded = pad_sequence(src_batch, padding_value=src_vocab.pad_idx)
    tgt_padded = pad_sequence(tgt_batch, padding_value=tgt_vocab.pad_idx)
    return src_padded.to(device), tgt_padded.to(device)

def train_model(model, loader, optimizer, criterion):
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
        >>> translator = Translator(model_path='nmt_model.pt', 
        ...                       vocabs_path='vocabs.pkl',
        ...                       device='cuda' if torch.cuda.is_available() else 'cpu')
        >>> translation = translator.translate("moi school jai")
        >>> print(translation)
        'I go to school'
    """
    
    def __init__(self, model_path: str, vocabs_path: str, device: str = None):
        """
        Initialize the translator with a pre-trained model.
        
        Args:
            model_path (str): Path to the trained model weights (.pt file).
            vocabs_path (str): Path to the saved vocabulary files (.pkl file).
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                 If None, automatically selects GPU if available.
        
        Raises:
            FileNotFoundError: If model or vocabulary files are not found.
            RuntimeError: If there's an error loading the model or vocabularies.
            ValueError: If the model architecture doesn't match the expected format.
        """
        import torch
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Validate and load model files
        self._validate_paths(model_path, vocabs_path)
        self._load_vocabs(vocabs_path)
        self._initialize_model(model_path)
    
    def _validate_paths(self, model_path: str, vocabs_path: str):
        """Validate that the required files exist."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NMT model not found: {model_path}")
        if not os.path.exists(vocabs_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocabs_path}")
    
    def _load_vocabs(self, vocabs_path: str):
        """Load source and target language vocabularies."""
        try:
            with open(vocabs_path, 'rb') as f:
                self.src_vocab, self.tgt_vocab = pickle.load(f)
                
            # Validate vocabularies
            if not hasattr(self.src_vocab, 'token_to_idx') or not hasattr(self.tgt_vocab, 'idx_to_token'):
                raise ValueError("Invalid vocabulary format. Expected Vocab objects with token_to_idx and idx_to_token attributes.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary file {vocabs_path}: {str(e)}")
    
    def _initialize_model(self, model_path: str):
        """Initialize the model architecture and load weights."""
        try:
            import torch
            
            # Model hyperparameters (must match training configuration)
            ENC_EMB_DIM = 256
            DEC_EMB_DIM = 256
            HID_DIM = 512
            DROPOUT = 0.5
            
            # Initialize model components
            enc = Encoder(len(self.src_vocab), ENC_EMB_DIM, HID_DIM, DROPOUT)
            attn = Attention(HID_DIM)
            dec = Decoder(len(self.tgt_vocab), DEC_EMB_DIM, HID_DIM, DROPOUT, attn)
            
            # Create and load model
            self.model = Seq2Seq(enc, dec, self.device).to(self.device)
            
            # Load state dict with error handling for device mismatch
            state_dict = torch.load(model_path, map_location=torch.device(self.device))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                raise RuntimeError("CUDA out of memory. Try reducing batch size or using CPU.")
            raise RuntimeError(f"Error loading model: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def translate(self, sentence: str, max_len: int = 50) -> str:
        """
        Translate a Nagamese sentence to English.
        
        Args:
            sentence (str): The input sentence in Nagamese.
            max_len (int, optional): Maximum length of the output sequence. 
                                   Defaults to 50.
        
        Returns:
            str: The translated English sentence.
            
        Raises:
            ValueError: If the input is not a valid string or is empty.
            RuntimeError: If translation fails for any reason.
            
        Example:
            >>> translator = Translator('model.pt', 'vocabs.pkl')
            >>> translation = translator.translate("moi school te jai thake")
            >>> print(translation)
            'I go to school'
        """
        if not isinstance(sentence, str):
            raise ValueError(f"Input must be a string, got {type(sentence).__name__}")
            
        if not sentence.strip():
            return ""
            
        try:
            import torch
            
            # Tokenize and convert to tensor
            tokens = sentence.lower().split()
            if not tokens:
                return ""
                
            # Convert tokens to indices
            try:
                src_indices = [self.src_vocab.token_to_idx.get(tok, self.src_vocab.unk_idx) 
                             for tok in tokens]
                # Add batch dimension and ensure correct shape [seq_len, batch_size]
                src_tensor = torch.tensor(src_indices, device=self.device).unsqueeze(1)  # [seq_len, 1]
                src_len = len(src_indices)
            except Exception as e:
                raise ValueError(f"Error processing input tokens: {str(e)}")
            
            # Encode the source sentence
            self.model.eval()
            with torch.no_grad():
                # Ensure the model is on the correct device
                self.model.to(self.device)
                # Call the encoder and handle its return value
                encoder_output = self.model.encoder(src_tensor)
                
                # Handle the case when the encoder returns a tuple or a single value
                if isinstance(encoder_output, tuple):
                    if len(encoder_output) >= 2:
                        encoder_outputs, hidden = encoder_output[0], encoder_output[1]
                    else:
                        raise RuntimeError(f"Unexpected number of outputs from encoder: {len(encoder_output)}")
                else:
                    # If the encoder returns a single value, use it as both outputs and hidden
                    encoder_outputs = hidden = encoder_output
                    
                # Initialize target with <sos> token
                trg_indices = [self.tgt_vocab.sos_idx]
                
                # Generate translation token by token
                for _ in range(max_len):
                    # Get the last token in the sequence
                    trg_tensor = torch.tensor([trg_indices[-1]], device=self.device).unsqueeze(0)  # [1, 1]
                    
                    # Call the decoder and handle its return value
                    decoder_output = self.model.decoder(trg_tensor, hidden, encoder_outputs)
                    
                    # Handle the case when the decoder returns a tuple or a single value
                    if isinstance(decoder_output, tuple):
                        if len(decoder_output) >= 2:
                            output, hidden = decoder_output[0], decoder_output[1]
                        else:
                            output = decoder_output[0]
                    else:
                        output = decoder_output
                        
                    # Get the most likely next token
                    if output.dim() == 3:  # [batch_size, seq_len, vocab_size] -> take last output
                        output = output[:, -1, :]  # Take the last output in the sequence
                    
                    # Ensure output is 2D [batch_size, vocab_size]
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                        
                    # Get the token with highest probability
                    pred_token = output.argmax(dim=-1).squeeze().item()
                    trg_indices.append(pred_token)
                    
                    # Stop if we predict the end-of-sentence token
                    if pred_token == self.tgt_vocab.eos_idx:
                        break
                
                # Convert indices back to tokens
                try:
                    trg_tokens = [self.tgt_vocab.idx_to_token[i] 
                                for i in trg_indices[1:]]  # Remove <sos> and keep <eos> if present
                    # Remove <eos> if present
                    if trg_tokens and trg_tokens[-1] == self.tgt_vocab.eos_token:
                        trg_tokens = trg_tokens[:-1]
                    return ' '.join(trg_tokens)
                    
                except Exception as e:
                    raise RuntimeError(f"Error generating output tokens: {str(e)}")
                    
        except Exception as e:
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    def __repr__(self) -> str:
        """Return a string representation of the translator."""
        return (f"Translator("
                f"device='{self.device}', "
                f"src_vocab_size={len(self.src_vocab)}, "
                f"tgt_vocab_size={len(self.tgt_vocab)})")

if __name__ == '__main__':
    # --- Configuration ---
    N_EPOCHS = 10
    MODEL_PATH = 'nmt-nagamese-english.pt'
    VOCABS_PATH = 'nmt-vocabs.pkl'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- 1. Load and Prepare Data ---
    df = load_and_prep_data('merged.csv')
    if df is not None:
        src_vocab = Vocab(df['nagamese_tokens'].tolist())
        tgt_vocab = Vocab(df['english_tokens'].tolist())

        with open(VOCABS_PATH, 'wb') as f:
            pickle.dump((src_vocab, tgt_vocab), f)
        print(f"Source vocab size: {len(src_vocab)}")
        print(f"Target vocab size: {len(tgt_vocab)}")

        dataset = TranslationDataset(df, src_vocab, tgt_vocab)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Correctly create the collate function with arguments
        collate_with_args = lambda batch: collate_fn(batch, src_vocab, tgt_vocab, DEVICE)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_with_args)

        # --- 2. Initialize Model ---
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        DROPOUT = 0.5

        enc = Encoder(len(src_vocab), ENC_EMB_DIM, HID_DIM, DROPOUT)
        attn = Attention(HID_DIM)
        dec = Decoder(len(tgt_vocab), DEC_EMB_DIM, HID_DIM, DROPOUT, attn)
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)

        # --- 3. Train the Model ---
        print("\n--- Starting NMT Model Training ---")
        for epoch in range(N_EPOCHS):
            train_loss = train_model(model, train_loader, optimizer, criterion)
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        # --- 4. Load and Test the Translator ---
        print("\n--- Loading Trained Model for Inference ---")
        translator = Translator(MODEL_PATH, VOCABS_PATH, DEVICE)
        test_sentence = "moi ghor te jai ase"
        translation = translator.translate(test_sentence)
        print(f"Nagamese Input: '{test_sentence}'")
        print(f"Predicted English Translation: '{translation}'")
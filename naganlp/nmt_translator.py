"""
Neural Machine Translation (NMT) module for Nagamese-English translation.

This module provides functionality for training and using a sequence-to-sequence
model with attention for translating between Nagamese and English. It includes
training, evaluation, and inference capabilities with support for beam search.
"""

import json
import logging
import os
import pickle
import random
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import HfApi, hf_hub_download, Repository
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# --- User-Facing Translator Class (for inference after training) ---
class Translator:
    """A neural machine translation model for Nagamese to English translation.
    
    This class provides an interface for loading and using a pre-trained NMT model
    for translating between Nagamese and English. It uses the 'agnivamaiti/naganlp-nmt-en'
    model by default from the Hugging Face Hub.
    
    Args:
        model_id: The Hugging Face model ID (default: 'agnivamaiti/naganlp-nmt-en')
        device: The device to load the model on ('cuda' or 'cpu'). If None, will auto-detect.
        
    Example:
        >>> # Using the default model
        >>> translator = Translator()
        >>> translation = translator.translate("moi bhal aase")
        >>> print(translation)
        'I am fine'
        
        >>> # Using a custom model from Hugging Face
        >>> custom_translator = Translator("username/custom-nmt-model")
    """
    
    def __init__(self, model_id: str = 'agnivamaiti/naganlp-nmt-en', device: Optional[str] = None):
        """Initialize the translator with a pre-trained model from Hugging Face Hub.
        
        Args:
            model_id: Hugging Face model ID (default: 'agnivamaiti/naganlp-nmt-en')
            device: Device to load the model on ('cuda' or 'cpu')
            
        Raises:
            RuntimeError: If model loading fails
            FileNotFoundError: If required model files are not found
        """
        try:
            self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_id = model_id
            logger.info(f"Loading translator model '{model_id}' to device: {self.device}")
            
            # Define expected file names
            model_files = {
                'model': 'pytorch_model.bin',
                'src_tokenizer': 'source.spm',
                'tgt_tokenizer': 'target.spm',
                'config': 'config.json'
            }
            
            # Download model and tokenizers
            downloaded_files = {}
            for file_type, filename in model_files.items():
                try:
                    file_path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        cache_dir=os.path.expanduser("~/.cache/naganlp")
                    )
                    downloaded_files[file_type] = file_path
                    logger.debug(f"Downloaded {file_type}: {file_path}")
                except Exception as e:
                    if file_type == 'model':  # Model file is required
                        raise FileNotFoundError(
                            f"Required model file '{filename}' not found in {model_id}. "
                            f"Error: {str(e)}"
                        ) from e
                    logger.warning(f"Could not load {file_type} file: {str(e)}")
            
            # Initialize tokenizers
            self.src_vocab = SPVocab(downloaded_files['src_tokenizer'])
            self.tgt_vocab = SPVocab(downloaded_files['tgt_tokenizer'])
            
            # Load model configuration
            with open(downloaded_files['config'], 'r') as f:
                config = json.load(f)
            
            # Initialize model architecture
            attn = Attention(config['hid_dim'])
            enc = Encoder(
                input_dim=len(self.src_vocab),
                emb_dim=config['emb_dim'],
                hid_dim=config['hid_dim'],
                n_layers=config['n_layers'],
                dropout=config['dropout'],
                pad_idx=self.src_vocab.pad_idx
            )
            dec = Decoder(
                output_dim=len(self.tgt_vocab),
                emb_dim=config['emb_dim'],
                hid_dim=config['hid_dim'],
                n_layers=config['n_layers'],
                dropout=config['dropout'],
                attention=attn,
                pad_idx=self.tgt_vocab.pad_idx
            )
            
            # Initialize and load model weights
            self.model = Seq2Seq(enc, dec, self.device).to(self.device)
            self.model.load_state_dict(torch.load(downloaded_files['model'], map_location=self.device))
            self.model.eval()
            
            logger.info(f"Successfully loaded translator model from {model_id}")
            
        except Exception as e:
            error_msg = f"Failed to initialize translator with model '{model_id}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def translate(
        self, 
        sentence: str, 
        beam_size: int = 5, 
        max_len: int = 50,
        length_penalty: float = 0.7,
        return_tokens: bool = False
    ) -> Union[str, Tuple[str, List[int]]]:
        """Translate a sentence from Nagamese to English.
        
        Args:
            sentence: The input sentence in Nagamese
            beam_size: Number of beams to use during decoding (default: 5)
            max_len: Maximum length of the generated translation (default: 50)
            length_penalty: Length penalty parameter for beam search (default: 0.7)
            return_tokens: If True, returns a tuple of (translation, token_ids)
            
        Returns:
            str or tuple: The translated sentence in English, and optionally the token IDs
            
        Raises:
            ValueError: If the input sentence is empty or invalid
            RuntimeError: If translation fails
            
        Example:
            >>> translator = Translator()
            >>> # Basic translation
            >>> translation = translator.translate("moi bhal aase")
            >>> print(translation)
            'I am fine'
            
            >>> # Get translation with token IDs
            >>> translation, token_ids = translator.translate("tumi kene asa?", return_tokens=True)
            >>> print(f"Translation: {translation}")
            >>> print(f"Token IDs: {token_ids}")
        """
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("Input must be a non-empty string")
            
        try:
            self.model.eval()
            
            # Preprocess and tokenize input
            sentence = sentence.strip()
            logger.debug(f"Translating: {sentence}")
            
            # Tokenize and prepare input
            tokens = self.src_vocab.sp.encode(sentence.lower())
            if not tokens:
                logger.warning(f"No tokens generated for input: {sentence}")
                return ("", []) if return_tokens else ""
                
            src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(self.device)
            
            # Generate translation with beam search
            with torch.no_grad():
                translation_ids, score = beam_search_decode(
                    self.model, 
                    src_tensor, 
                    self.tgt_vocab, 
                    beam_size=beam_size, 
                    max_len=max_len,
                    length_penalty=length_penalty
                )
            
            # Process output
            if not translation_ids:
                logger.warning(f"No translation generated for input: {sentence}")
                return ("", []) if return_tokens else ""
                
            # Store original token IDs if needed
            output_token_ids = translation_ids.copy()
            
            # Remove special tokens
            if translation_ids and translation_ids[0] == self.tgt_vocab.sos_idx: 
                translation_ids = translation_ids[1:]
            if translation_ids and translation_ids[-1] == self.tgt_vocab.eos_idx: 
                translation_ids = translation_ids[:-1]
                
            # Decode tokens to string
            translation = self.tgt_vocab.sp.decode(translation_ids)
            
            logger.debug(f"Translated '{sentence}' -> '{translation}' (score: {score:.2f})")
            
            if return_tokens:
                return translation, output_token_ids
            return translation
            
        except Exception as e:
            error_msg = f"Translation failed for input '{sentence}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


# --- Data and Tokenizer Utilities ---
def train_sentencepiece_model(
    text_data_path: str, 
    model_prefix: str, 
    vocab_size: int, 
    lang: str
) -> Tuple[str, str]:
    """Train a SentencePiece tokenizer on the given text data.
    
    Args:
        text_data_path: Path to the text file containing training data
        model_prefix: Prefix for the output model files
        vocab_size: Size of the vocabulary
        lang: Language code (used in filenames)
        
    Returns:
        Tuple containing paths to the trained model and vocab files
        
    Raises:
        FileNotFoundError: If the input text file is not found
        RuntimeError: If training fails
    """
    if not os.path.exists(text_data_path):
        raise FileNotFoundError(f"Text data file not found: {text_data_path}")
        
    model_path = f"{model_prefix}_{lang}.model"
    vocab_path = f"{model_prefix}_{lang}.vocab"
    
    if os.path.exists(model_path):
        logger.info(f"Using existing tokenizer model: {model_path}")
        return model_path, vocab_path
    
    logger.info(f"Training SentencePiece model for {lang}...")
    
    try:
        spm.SentencePieceTrainer.train(
            f'--input={text_data_path} '
            f'--model_prefix={model_prefix}_{lang} '
            f'--vocab_size={vocab_size} '
            '--character_coverage=1.0 '
            '--model_type=bpe '
            '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
            '--pad_piece=<pad> --unk_piece=<unk> --bos_piece=<sos> --eos_piece=<eos>'
        )
        
        logger.info(f"Successfully trained tokenizer model: {model_path}")
        return model_path, vocab_path
        
    except Exception as e:
        logger.error(f"Failed to train SentencePiece model: {str(e)}")
        raise RuntimeError(f"SentencePiece training failed: {str(e)}") from e

def prep_dataframe(
    df: pd.DataFrame, 
    src_tokenizer: spm.SentencePieceProcessor, 
    tgt_tokenizer: spm.SentencePieceProcessor, 
    max_len: int = 256
) -> pd.DataFrame:
    """Preprocess and tokenize the parallel dataset.
    
    Args:
        df: Input DataFrame with 'nagamese' and 'english' columns
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        max_len: Maximum sequence length (longer sequences will be filtered out)
        
    Returns:
        Processed DataFrame with tokenized sequences
        
    Raises:
        ValueError: If required columns are missing from the input DataFrame
    """
    required_columns = {'nagamese', 'english'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns in DataFrame: {missing}")
    
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        # Remove HTML tags, normalize whitespace, and convert to lowercase
        text = re.sub(r'<[^>]+>', '', text).strip().lower()
        return re.sub(r'\s+', ' ', text)
    
    # Make a copy to avoid modifying the input DataFrame
    df = df.copy()
    
    # Clean text
    logger.info("Cleaning text data...")
    df['english_cleaned'] = df['english'].apply(clean_text)
    df['nagamese_cleaned'] = df['nagamese'].apply(clean_text)
    
    # Remove empty or invalid rows
    initial_count = len(df)
    df.dropna(subset=['english_cleaned', 'nagamese_cleaned'], inplace=True)
    df = df[(df['english_cleaned'].str.len() > 0) & (df['nagamese_cleaned'].str.len() > 0)]
    
    # Tokenize
    logger.info("Tokenizing text...")
    df['nagamese_tokens'] = df['nagamese_cleaned'].apply(lambda x: src_tokenizer.encode(x))
    df['english_tokens'] = df['english_cleaned'].apply(lambda x: tgt_tokenizer.encode(x))
    
    # Filter by sequence length
    df = df[df['nagamese_tokens'].str.len() <= max_len]
    df = df[df['english_tokens'].str.len() <= max_len]
    
    final_count = len(df)
    logger.info(
        f"Preprocessing complete. Kept {final_count} of {initial_count} "
        f"samples ({(final_count/max(initial_count, 1))*100:.1f}%)"
    )
    
    return df

def load_and_prep_data(
    filepath: str, 
    src_tokenizer: spm.SentencePieceProcessor, 
    tgt_tokenizer: spm.SentencePieceProcessor, 
    max_len: int = 256
) -> Optional[pd.DataFrame]:
    """Load and preprocess data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        max_len: Maximum sequence length
        
    Returns:
        Preprocessed DataFrame or None if file doesn't exist
        
    Raises:
        pd.errors.EmptyDataError: If the CSV file is empty
        pd.errors.ParserError: If the CSV file is malformed
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None
        
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(
            filepath, 
            encoding='utf-8', 
            on_bad_lines='warn', 
            header=None, 
            names=['nagamese', 'english']
        )
        
        if df.empty:
            logger.warning(f"Empty DataFrame loaded from {filepath}")
            return None
            
        return prep_dataframe(df, src_tokenizer, tgt_tokenizer, max_len)
        
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file: {filepath}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {filepath}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {str(e)}")
        raise


# --- Vocabulary and Dataset Classes ---
class SPVocab:
    """Wrapper around SentencePiece model for vocabulary operations."""
    
    def __init__(self, sp_model_path: str):
        """Initialize with a SentencePiece model.
        
        Args:
            sp_model_path: Path to the SentencePiece model file
            
        Raises:
            RuntimeError: If the model fails to load
        """
        try:
            self.sp = spm.SentencePieceProcessor()
            if not os.path.exists(sp_model_path):
                raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")
                
            self.sp.load(sp_model_path)
            
            # Store special token indices
            self.pad_idx = self.sp.pad_id()
            self.unk_idx = self.sp.unk_id()
            self.sos_idx = self.sp.bos_id()
            self.eos_idx = self.sp.eos_id()
            
        except Exception as e:
            logger.error(f"Failed to load SentencePiece model from {sp_model_path}: {str(e)}")
            raise RuntimeError(f"Failed to load SentencePiece model: {str(e)}") from e
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return self.sp.get_piece_size()
    
    def __getitem__(self, idx: int) -> str:
        """Get the token string for a given index."""
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for vocabulary of size {len(self)}")
        return self.sp.id_to_piece(idx)
    
    def __contains__(self, token: str) -> bool:
        """Check if a token exists in the vocabulary."""
        return self.sp.piece_to_id(token) != self.unk_idx

class TranslationDataset(Dataset):
    """PyTorch Dataset for parallel sentence pairs."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame containing tokenized sentences.
        
        Args:
            df: DataFrame with 'nagamese_tokens' and 'english_tokens' columns
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = {'nagamese_tokens', 'english_tokens'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in DataFrame: {missing}")
            
        self.src_sents = df['nagamese_tokens'].tolist()
        self.tgt_sents = df['english_tokens'].tolist()
        
        logger.info(f"Initialized TranslationDataset with {len(self)} examples")
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.src_sents)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single example by index.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Tuple of (source_tokens, target_tokens) as PyTorch tensors
            
        Raises:
            IndexError: If the index is out of bounds
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        return (
            torch.tensor(self.src_sents[idx], dtype=torch.long),
            torch.tensor(self.tgt_sents[idx], dtype=torch.long)
        )


# --- Model Component Classes ---
class Encoder(nn.Module):
    """Encoder module for the sequence-to-sequence model.
    
    Uses a bidirectional GRU to encode the source sequence into a fixed-size
    representation that captures the meaning of the input sequence.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        emb_dim: int, 
        hid_dim: int, 
        n_layers: int, 
        dropout: float, 
        pad_idx: int
    ) -> None:
        """Initialize the encoder.
        
        Args:
            input_dim: Size of the input vocabulary
            emb_dim: Size of the token embeddings
            hid_dim: Size of the hidden states
            n_layers: Number of RNN layers
            dropout: Dropout probability
            pad_idx: Index of the padding token
        """
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            emb_dim, 
            hid_dim, 
            num_layers=n_layers, 
            bidirectional=True, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=False
        )
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(
        self, 
        src: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.
        
        Args:
            src: Source sequence tensor of shape (src_len, batch_size)
            
        Returns:
            outputs: Encoder outputs of shape (src_len, batch_size, hid_dim * 2)
            hidden: Final hidden state of shape (n_layers, batch_size, hid_dim)
        """
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        
        outputs, hidden = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim * 2]
        # hidden = [n_layers * 2, batch_size, hid_dim]
        
        # Concatenate the final forward and backward hidden states
        # and apply a linear layer to get the final hidden state
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # hidden = [batch_size, hid_dim]
        hidden = hidden.unsqueeze(0)  # Add layer dimension
        # hidden = [1, batch_size, hid_dim]
        
        return outputs, hidden

class Attention(nn.Module):
    """Attention mechanism for the decoder.
    
    Implements additive attention (Bahdanau et al., 2014) which computes a weighted
    sum of the encoder outputs based on the current decoder hidden state.
    """
    
    def __init__(self, hid_dim: int) -> None:
        """Initialize the attention mechanism.
        
        Args:
            hid_dim: Size of the hidden states
        """
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(
        self, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the attention mechanism.
        
        Args:
            hidden: Current decoder hidden state of shape (batch_size, hid_dim)
            encoder_outputs: Encoder outputs of shape (src_len, batch_size, hid_dim * 2)
            
        Returns:
            Attention weights of shape (batch_size, src_len)
        """
        # hidden = [batch_size, hid_dim]
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        
        src_len = encoder_outputs.shape[0]
        
        # Repeat hidden state for each source token
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch_size, src_len, hid_dim]
        
        # Permute encoder outputs for attention calculation
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, hid_dim * 2]
        
        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch_size, src_len, hid_dim]
        
        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        # attention = [batch_size, src_len]
        
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    """Decoder module for the sequence-to-sequence model with attention."""
    
    def __init__(
        self, 
        output_dim: int, 
        emb_dim: int, 
        hid_dim: int, 
        n_layers: int, 
        dropout: float, 
        attention: nn.Module, 
        pad_idx: int
    ) -> None:
        """Initialize the decoder.
        
        Args:
            output_dim: Size of the output vocabulary
            emb_dim: Size of the token embeddings
            hid_dim: Size of the hidden states
            n_layers: Number of RNN layers
            dropout: Dropout probability
            attention: Attention module
            pad_idx: Index of the padding token
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        
        # Input: [emb_dim + (hid_dim * 2)]
        self.rnn = nn.GRU(
            (hid_dim * 2) + emb_dim, 
            hid_dim, 
            num_layers=n_layers, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=False
        )
        
        # Input: [hid_dim + (hid_dim * 2) + emb_dim]
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(
        self, 
        input: torch.Tensor, 
        hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the decoder.
        
        Args:
            input: Input token indices of shape (batch_size,)
            hidden: Previous hidden state of shape (n_layers, batch_size, hid_dim)
            encoder_outputs: Encoder outputs of shape (src_len, batch_size, hid_dim * 2)
            
        Returns:
            prediction: Output logits of shape (batch_size, output_dim)
            hidden: Updated hidden state of shape (n_layers, batch_size, hid_dim)
            attention: Attention weights of shape (batch_size, src_len)
        """
        # input = [batch_size]
        # hidden = [n_layers, batch_size, hid_dim]
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        
        # Add sequence length dimension
        input = input.unsqueeze(0)  # [1, batch_size]
        
        # Get embeddings and apply dropout
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]
        
        # Calculate attention weights
        # hidden[-1] = [batch_size, hid_dim]
        # a = [batch_size, src_len]
        a = self.attention(hidden[-1], encoder_outputs)
        
        # Calculate weighted sum of encoder outputs
        # a = [batch_size, 1, src_len]
        # encoder_outputs = [batch_size, src_len, hid_dim * 2] (after permute)
        # weighted = [batch_size, 1, hid_dim * 2]
        weighted = torch.bmm(a.unsqueeze(1), encoder_outputs.permute(1, 0, 2))
        
        # Prepare RNN input
        # weighted = [1, batch_size, hid_dim * 2] (after permute)
        # embedded = [1, batch_size, emb_dim]
        # rnn_input = [1, batch_size, (hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, weighted.permute(1, 0, 2)), dim=2)
        
        # Pass through RNN
        # output = [1, batch_size, hid_dim]
        # hidden = [n_layers, batch_size, hid_dim]
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Prepare for prediction
        # output = [batch_size, hid_dim] (after squeeze)
        # weighted = [batch_size, hid_dim * 2]
        # embedded = [batch_size, emb_dim] (after squeeze)
        # prediction = [batch_size, output_dim]
        prediction = self.fc_out(
            torch.cat((output.squeeze(0), weighted.squeeze(1), embedded.squeeze(0)), dim=1)
        )
        
        return prediction, hidden, a

class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with attention for machine translation."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device) -> None:
        """Initialize the sequence-to-sequence model.
        
        Args:
            encoder: Encoder module
            decoder: Decoder module with attention
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Ensure encoder and decoder hidden dimensions match
        if self.encoder.hid_dim != self.decoder.hid_dim:
            raise ValueError(
                f"Hidden dimensions do not match! "
                f"Encoder: {self.encoder.hid_dim}, Decoder: {self.decoder.hid_dim}"
            )
    
    def forward(
        self, 
        src: torch.Tensor, 
        trg: torch.Tensor, 
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """Forward pass of the sequence-to-sequence model.
        
        Args:
            src: Source sequence tensor of shape (src_len, batch_size)
            trg: Target sequence tensor of shape (trg_len, batch_size)
            teacher_forcing_ratio: Probability of using teacher forcing
                (using true previous token instead of predicted one)
                
        Returns:
            Output logits of shape (trg_len, batch_size, output_dim)
        """
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        # hidden = [1, batch_size, hid_dim]
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the <sos> tokens
        input = trg[0, :]  # [batch_size]
        
        # Decode the target sequence
        for t in range(1, trg_len):
            # Insert input token embedding, previous hidden state and all encoder outputs
            # Receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            # Place predictions in the output tensor
            outputs[t] = output
            
            # Decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)  # [batch_size]
            
            # If teacher forcing, use actual next token as next input
            # If not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs


# --- Training and Evaluation Utilities ---
def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], 
    pad_idx: int, 
    device: torch.device, 
    sos_idx: int = 2, 
    eos_idx: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader to process batches.
    
    Args:
        batch: List of (source, target) token index pairs
        pad_idx: Index of the padding token
        device: Device to move tensors to
        sos_idx: Start-of-sequence token index (default: 2)
        eos_idx: End-of-sequence token index (default: 3)
        
    Returns:
        Tuple of (padded_src, padded_tgt) tensors with special tokens added
    """
    if not batch:
        raise ValueError("Batch cannot be empty")
        
    src_batch, tgt_batch = [], []
    
    for src_sample, tgt_sample in batch:
        # Add SOS and EOS tokens
        src_with_special = torch.cat([
            torch.tensor([sos_idx], dtype=torch.long),
            src_sample,
            torch.tensor([eos_idx], dtype=torch.long)
        ])
        
        tgt_with_special = torch.cat([
            torch.tensor([sos_idx], dtype=torch.long),
            tgt_sample,
            torch.tensor([eos_idx], dtype=torch.long)
        ])
        
        src_batch.append(src_with_special)
        tgt_batch.append(tgt_with_special)
    
    # Pad sequences to the same length
    padded_src = pad_sequence(src_batch, padding_value=pad_idx, batch_first=False)
    padded_tgt = pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=False)
    
    # Move to specified device
    padded_src = padded_src.to(device)
    padded_tgt = padded_tgt.to(device)
    
    return padded_src, padded_tgt

def train_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    clip: float
) -> float:
    """Train the model for one epoch.
    
    Args:
        model: The sequence-to-sequence model to train
        loader: DataLoader for the training data
        optimizer: Optimizer to use for training
        criterion: Loss function
        clip: Gradient clipping value
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    total_tokens = 0
    
    progress_bar = tqdm(
        loader, 
        desc="Training", 
        leave=False,
        bar_format='{l_bar}{bar:20}{r_bar}'
    )
    
    for src, trg in progress_bar:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        # output = [trg_len, batch_size, output_dim]
        output = model(src, trg)
        
        # Calculate loss (ignoring padding tokens)
        # output = [(trg_len - 1) * batch_size, output_dim]
        # trg = [(trg_len - 1) * batch_size]
        loss = criterion(
            output[1:].view(-1, output.shape[-1]), 
            trg[1:].view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        # Update statistics
        batch_loss = loss.item()
        epoch_loss += batch_loss * (trg.size(1))  # Weight by batch size
        total_tokens += (trg != 0).sum().item()  # Count non-padding tokens
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{batch_loss:.3f}',
            'ppl': f'{math.exp(batch_loss):.2f}'
        })
    
    # Calculate average loss per token
    avg_loss = epoch_loss / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    src_vocab: SPVocab,
    tgt_vocab: SPVocab,
    beam_size: int,
    length_penalty: float,
    max_examples: int = 3
) -> Tuple[float, float]:
    """Evaluate the model on the validation set.
    
    Args:
        model: The sequence-to-sequence model to evaluate
        loader: DataLoader for the validation data
        criterion: Loss function
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        beam_size: Number of beams for beam search
        length_penalty: Length penalty for beam search
        max_examples: Maximum number of examples to print
        
    Returns:
        Tuple of (average_loss, bleu_score)
    """
    model.eval()
    epoch_loss = 0.0
    total_tokens = 0
    
    # For BLEU score calculation
    candidate_corpus = []
    references_corpus = []
    
    # For logging examples
    examples_logged = 0
    
    with torch.no_grad():
        progress_bar = tqdm(
            loader, 
            desc="Validation", 
            leave=False,
            bar_format='{l_bar}{bar:20}{r_bar}'
        )
        
        for src, trg in progress_bar:
            # Forward pass (teacher forcing ratio = 0)
            output = model(src, trg, 0)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output_flat = output[1:].view(-1, output_dim)
            trg_flat = trg[1:].view(-1)
            loss = criterion(output_flat, trg_flat)
            
            # Update statistics
            batch_loss = loss.item()
            epoch_loss += batch_loss * (trg.size(1))  # Weight by batch size
            total_tokens += (trg != 0).sum().item()  # Count non-padding tokens
            
            # Generate translations for BLEU score and examples
            for i in range(src.shape[1]):
                # Process source and reference
                src_sent = src[:, i].unsqueeze(1)  # [src_len, 1]
                trg_sent = trg[:, i]  # [trg_len]
                
                # Decode source and reference
                src_tokens = [
                    token for token in src_sent.squeeze().tolist() 
                    if token not in {src_vocab.pad_idx, src_vocab.sos_idx, src_vocab.eos_idx}
                ]
                source_text = src_vocab.sp.decode(src_tokens)
                
                ref_tokens = [
                    token for token in trg_sent.tolist() 
                    if token not in {tgt_vocab.pad_idx, tgt_vocab.sos_idx, tgt_vocab.eos_idx}
                ]
                reference_text = tgt_vocab.sp.decode(ref_tokens)
                
                # Generate translation with beam search
                translation_ids, _ = beam_search_decode(
                    model, 
                    src_sent, 
                    tgt_vocab, 
                    beam_size, 
                    max_len=50, 
                    length_penalty=length_penalty
                )
                
                # Process and decode translation
                pred_tokens = [
                    token for token in translation_ids 
                    if token not in {tgt_vocab.pad_idx, tgt_vocab.sos_idx, tgt_vocab.eos_idx}
                ]
                candidate_text = tgt_vocab.sp.decode(pred_tokens)
                
                # Add to BLEU score calculation
                references_corpus.append([reference_text.split()])
                candidate_corpus.append(candidate_text.split())
                
                # Log examples
                if examples_logged < max_examples:
                    logger.info("\n=== Translation Example ===")
                    logger.info(f"Source:     {source_text}")
                    logger.info(f"Reference:  {reference_text}")
                    logger.info(f"Predicted:  {candidate_text}")
                    examples_logged += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.3f}',
                'ppl': f'{math.exp(batch_loss):.2f}'
            })
    
    # Calculate BLEU score
    smoothing = SmoothingFunction().method4
    bleu_score = corpus_bleu(references_corpus, candidate_corpus, smoothing_function=smoothing)
    
    # Calculate average loss per token
    avg_loss = epoch_loss / total_tokens if total_tokens > 0 else 0.0
    
    logger.info(f"Validation - Loss: {avg_loss:.3f} | PPL: {math.exp(avg_loss):.2f} | BLEU: {bleu_score:.4f}")
    
    return avg_loss, bleu_score

def beam_search_decode(
    model: nn.Module,
    src_tensor: torch.Tensor,
    tgt_vocab: SPVocab,
    beam_size: int,
    max_len: int,
    length_penalty: float = 0.7
) -> Tuple[List[int], Optional[float]]:
    """Generate translation using beam search.
    
    Args:
        model: The sequence-to-sequence model
        src_tensor: Source sequence tensor of shape (src_len, 1)
        tgt_vocab: Target vocabulary
        beam_size: Number of beams to use
        max_len: Maximum length of the generated sequence
        length_penalty: Length penalty parameter (0.0 for no penalty, >1.0 for shorter sequences)
        
    Returns:
        Tuple of (best_sequence, score) where:
            - best_sequence: List of token indices representing the best translation
            - score: Score of the best sequence (or None if no valid sequence found)
    """
    model.eval()
    
    # Encode the source sequence
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        # hidden = [1, 1, hid_dim] -> [n_layers, 1, hid_dim]
        hidden = hidden.unsqueeze(0).repeat(model.decoder.n_layers, 1, 1)
    
    # Initialize beam with start token
    # Each element is (score, sequence, hidden_state)
    beam = [(0.0, [tgt_vocab.sos_idx], hidden)]
    completed_sequences = []
    
    # Beam search
    for step in range(max_len):
        new_beam = []
        
        for score, seq, hidden_state in beam:
            # If sequence is already complete, add to completed sequences
            if seq[-1] == tgt_vocab.eos_idx:
                # Apply length normalization
                final_score = score / ((len(seq) ** length_penalty) if length_penalty > 0 else 1.0)
                completed_sequences.append((final_score, seq))
                continue
                
            # Prepare input for decoder
            trg_tensor = torch.LongTensor([seq[-1]]).to(model.device)  # [1]
            
            # Decode one step
            with torch.no_grad():
                output, new_hidden, _ = model.decoder(
                    trg_tensor, 
                    hidden_state, 
                    encoder_outputs
                )
            
            # Get top-k predictions
            log_probs = torch.log_softmax(output, dim=1)  # [1, output_dim]
            top_log_probs, top_indices = log_probs.topk(beam_size)  # [1, beam_size]
            
            # Expand beam with new candidates
            for i in range(beam_size):
                next_token = top_indices[0, i].item()
                log_prob = top_log_probs[0, i].item()
                
                # Update score (sum of log probs)
                new_score = score + log_prob
                
                # Add to new beam
                new_beam.append((
                    new_score,
                    seq + [next_token],
                    new_hidden
                ))
        
        # If no new candidates, stop
        if not new_beam:
            break
        
        # Keep top-k candidates
        beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # Check if all sequences in the beam end with EOS
        if all(seq[-1] == tgt_vocab.eos_idx for _, seq, _ in beam):
            # Apply length normalization and add to completed sequences
            completed_sequences.extend([
                (s / ((len(q) ** length_penalty) if length_penalty > 0 else 1.0), q) 
                for s, q, _ in beam
            ])
            break
    
    # If no sequences were completed, use the best from the beam
    if not completed_sequences:
        completed_sequences.extend([
            (s / ((len(q) ** length_penalty) if length_penalty > 0 else 1.0), q)
            for s, q, _ in beam
        ])
    
    # Sort completed sequences by score and return the best one
    if completed_sequences:
        completed_sequences.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sequence = completed_sequences[0]
        return best_sequence, best_score
    
    # Fallback: return empty sequence if no valid sequence was found
    return [tgt_vocab.sos_idx], None


# --- Main Training Orchestration Function ---
def train_and_upload_translator(
    csv_path: str,
    hub_model_id: str,
    gloss_path: Optional[str] = None,
    intermediate_phrases_path: Optional[str] = None
):
    # --- Hyperparameters ---
    MAX_FINETUNE_EPOCHS = 0
    PRETRAIN_EPOCHS = 20
    INTERMEDIATE_EPOCHS = 40
    BATCH_SIZE = 32
    PATIENCE = 5
    ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS = 256, 256, 512, 2
    DROPOUT = 0.4
    LEARNING_RATE, WEIGHT_DECAY, CLIP = 1e-4, 1e-5, 1.0
    SP_VOCAB_SIZE, BEAM_SIZE, LENGTH_PENALTY = 8000, 3, 0.7
    VAL_SPLIT_SIZE = 0.1
    CHECKPOINT_PATH = 'nmt_checkpoint.pt'

    # --- CORRECTED: Safe saving function to prevent corruption ---
    def safe_save(state: dict, path: str):
        """Atomically saves a checkpoint to avoid corruption from interruptions."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=os.path.dirname(path), suffix='.tmp') as f:
                temp_path = f.name
                torch.save(state, f)
            os.rename(temp_path, path)
        except Exception as e:
            print(f"!!! FAILED to save checkpoint: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Phase 1a: Tokenizer Training ---
    print("--- Phase 1a: Preparing Data and Training Tokenizers ---")
    full_df_for_tok = pd.read_csv(csv_path)
    if gloss_path and os.path.exists(gloss_path):
        gloss_df_temp = pd.read_csv(gloss_path, header=None, names=['nagamese', 'english'])
        full_df_for_tok = pd.concat([full_df_for_tok, gloss_df_temp], ignore_index=True)
    
    nagamese_text_path = "temp_nagamese.txt"
    english_text_path = "temp_english.txt"
    with open(nagamese_text_path, "w", encoding="utf-8") as f: f.write("\n".join(full_df_for_tok['nagamese'].dropna().astype(str).tolist()))
    with open(english_text_path, "w", encoding="utf-8") as f: f.write("\n".join(full_df_for_tok['english'].dropna().astype(str).tolist()))
    
    src_sp_model_path, src_sp_vocab_path = train_sentencepiece_model(nagamese_text_path, "naga_sp", SP_VOCAB_SIZE, "nagamese")
    tgt_sp_model_path, tgt_sp_vocab_path = train_sentencepiece_model(english_text_path, "eng_sp", SP_VOCAB_SIZE, "english")
    
    src_vocab = SPVocab(src_sp_model_path)
    tgt_vocab = SPVocab(tgt_sp_model_path)
    
    os.remove(nagamese_text_path); os.remove(english_text_path)
    print(f"Source vocab size: {len(src_vocab)}; Target vocab size: {len(tgt_vocab)}")

    # --- Phase 1b: Model and Optimizer Initialization ---
    print("--- Phase 1b: Initializing Model ---")
    attn = Attention(HID_DIM)
    enc = Encoder(len(src_vocab), ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, src_vocab.pad_idx)
    dec = Decoder(len(tgt_vocab), DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, attn, tgt_vocab.pad_idx)
    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    # --- CORRECTED: Checkpoint Loading with Error Handling ---
    start_epoch = 0
    best_bleu = -1.0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_bleu = checkpoint.get('bleu', -1.0)
            print(f"Resumed from Epoch {start_epoch}. Best BLEU so far: {best_bleu*100:.2f}")
        except (RuntimeError, EOFError) as e:
            print(f"!!! WARNING: Could not load checkpoint file. It may be corrupt: {e}")
            print("--- Deleting corrupt checkpoint and starting training from scratch. ---")
            os.remove(CHECKPOINT_PATH)
            start_epoch = 0
            best_bleu = -1.0

    # --- Phase 2: Pre-training (Glossary) ---
    if start_epoch < PRETRAIN_EPOCHS:
        if gloss_path and os.path.exists(gloss_path):
            print("\n--- Phase 2: Pre-training on Augmented Glossary Data ---")
            gloss_df = pd.read_csv(gloss_path, header=None, names=['nagamese', 'english'])
            gloss_df_flipped = gloss_df.rename(columns={'nagamese': 'english', 'english': 'nagamese'})
            augmented_gloss_df = pd.concat([gloss_df, gloss_df_flipped], ignore_index=True)
            print(f"Original glossary size: {len(gloss_df)}. Augmented size: {len(augmented_gloss_df)}")
            
            pretrain_dataset = TranslationDataset(prep_dataframe(augmented_gloss_df, src_vocab.sp, tgt_vocab.sp))
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, device))

            for epoch in range(start_epoch, PRETRAIN_EPOCHS):
                train_loss = train_epoch(model, pretrain_loader, optimizer, criterion, CLIP)
                print(f"Pre-train Epoch: {epoch+1:02}/{PRETRAIN_EPOCHS} | Train Loss: {train_loss:.3f}")
                safe_save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'bleu': best_bleu}, CHECKPOINT_PATH)
        else:
             print("\n--- Skipping Phase 2: Pre-training (no glossary file provided or found) ---")
    
    # --- Phase 3: Intermediate Training (Phrases) ---
    INTERMEDIATE_END_EPOCH = PRETRAIN_EPOCHS + INTERMEDIATE_EPOCHS
    if start_epoch < INTERMEDIATE_END_EPOCH:
        if intermediate_phrases_path and os.path.exists(intermediate_phrases_path):
            print("\n--- Phase 3: Intermediate Training on Phrases Data ---")
            phrases_df = pd.read_csv(intermediate_phrases_path, header=None, names=['nagamese', 'english'])
            print(f"Phrases data size: {len(phrases_df)}")
            
            intermediate_dataset = TranslationDataset(prep_dataframe(phrases_df, src_vocab.sp, tgt_vocab.sp))
            intermediate_loader = DataLoader(intermediate_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, device))
            
            loop_start_epoch = max(start_epoch, PRETRAIN_EPOCHS)
            for epoch in range(loop_start_epoch, INTERMEDIATE_END_EPOCH):
                train_loss = train_epoch(model, intermediate_loader, optimizer, criterion, CLIP)
                print(f"Intermediate-train Epoch: {epoch+1:02}/{INTERMEDIATE_END_EPOCH} | Train Loss: {train_loss:.3f}")
                safe_save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'bleu': best_bleu}, CHECKPOINT_PATH)
        else:
            if intermediate_phrases_path:
                 print(f"\n--- WARNING: Skipping Phase 3. Intermediate phrases file not found at: '{intermediate_phrases_path}' ---")
            else:
                 print("\n--- Skipping Phase 3: Intermediate Training (no phrases file provided) ---")

    # --- Phase 4: Fine-tuning with Validation (Sentences) ---
    print("\n--- Phase 4: Fine-tuning on Sentence Data ---")
    main_df_for_finetune = pd.read_csv(csv_path)
    full_dataset = TranslationDataset(prep_dataframe(main_df_for_finetune, src_vocab.sp, tgt_vocab.sp))
    
    indices = list(range(len(full_dataset)))
    split = int(np.floor(VAL_SPLIT_SIZE * len(full_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Full dataset size: {len(full_dataset)}. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, device))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, device))

    patience_counter = 0
    fine_tune_start_epoch = max(start_epoch, INTERMEDIATE_END_EPOCH)
    FINETUNE_END_EPOCH = fine_tune_start_epoch + MAX_FINETUNE_EPOCHS

    for epoch in range(fine_tune_start_epoch, FINETUNE_END_EPOCH):
        print(f"\n--- Fine-tuning Epoch {epoch+1} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP)
        val_loss, val_bleu = evaluate_model(model, val_loader, criterion, src_vocab, tgt_vocab, BEAM_SIZE, LENGTH_PENALTY)
        
        print(f"Epoch Summary: Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Val. BLEU: {val_bleu*100:.2f}")

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            patience_counter = 0
            print(f"\n  New best BLEU score! Saving checkpoint to '{CHECKPOINT_PATH}'")
            safe_save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'bleu': best_bleu}, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(f"\n  BLEU did not improve. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"--- Early stopping triggered after {PATIENCE} epochs with no improvement. ---")
            break

    # --- Phase 5: Uploading to Hugging Face Hub ---
    print(f"\n--- Phase 5: Uploading Best Model and Tokenizers to {hub_model_id} ---")
    try:
        api = HfApi()
        api.upload_file(path_or_fileobj=CHECKPOINT_PATH, path_in_repo="nmt_checkpoint.pt", repo_id=hub_model_id, repo_type="model")
        api.upload_file(path_or_fileobj=src_sp_model_path, path_in_repo=os.path.basename(src_sp_model_path), repo_id=hub_model_id)
        api.upload_file(path_or_fileobj=tgt_sp_model_path, path_in_repo=os.path.basename(tgt_sp_model_path), repo_id=hub_model_id)
        api.upload_file(path_or_fileobj=src_sp_vocab_path, path_in_repo=os.path.basename(src_sp_vocab_path), repo_id=hub_model_id)
        api.upload_file(path_or_fileobj=tgt_sp_vocab_path, path_in_repo=os.path.basename(tgt_sp_vocab_path), repo_id=hub_model_id)
        print("--- Best model, tokenizers, and vocabs successfully uploaded. ---")
    except Exception as e:
        print(f"!!! FAILED TO UPLOAD TO HUB: {e}")
"""Subword tokenization module using SentencePiece.

This module provides functionality for training and using subword tokenizers
based on the SentencePiece algorithm. It supports both training new tokenizers
and using pre-trained models for tokenization and detokenization.

Example:
    >>> from naganlp.subword_tokenizer import SubwordTokenizer, train_tokenizer
    >>> # Train a new tokenizer
    >>> train_tokenizer('data.txt', 'my_tokenizer', vocab_size=8000)
    >>> # Load and use the tokenizer
    >>> tokenizer = SubwordTokenizer('my_tokenizer.model')
    >>> tokens = tokenizer.tokenize("This is a test")
    >>> print(tokens)
    ['▁This', '▁is', '▁a', '▁test']
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

import pandas as pd
import sentencepiece as spm

# Default configuration for tokenizer training
DEFAULT_CONFIG = {
    'model_type': 'bpe',  # or 'unigram'
    'vocab_size': 8000,
    'character_coverage': 1.0,
    'pad_id': 0,
    'unk_id': 1,
    'bos_id': 2,
    'eos_id': 3,
    'pad_piece': '[PAD]',
    'unk_piece': '[UNK]',
    'bos_piece': '[BOS]',
    'eos_piece': '[EOS]',
    'user_defined_symbols': ['[MASK]', '[SEP]', '[CLS]']
}

class SubwordTokenizer:
    """A wrapper for SentencePiece tokenizer with additional utilities.
    
    This class provides a simple interface to the SentencePiece tokenizer
    with additional convenience methods for common NLP tasks.
    
    Args:
        model_path: Path to the SentencePiece model file (.model)
        
    Raises:
        FileNotFoundError: If the model file does not exist
        RuntimeError: If the model cannot be loaded
    """
    
    def __init__(self, model_path: Union[str, os.PathLike]):
        """Initialize the tokenizer with a pre-trained SentencePiece model."""
        model_path = str(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
            
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.Load(model_path):
            raise RuntimeError(f"Failed to load SentencePiece model from {model_path}")
    
    def tokenize(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[str]:
        """Tokenize text into subword tokens.
        
        Args:
            text: Input text to tokenize
            add_bos: Whether to add beginning-of-sentence token
            add_eos: Whether to add end-of-sentence token
            
        Returns:
            List of subword tokens
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text).__name__}")
        return self.sp.EncodeAsPieces(text, add_bos=add_bos, add_eos=add_eos)
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to a string.
        
        Args:
            tokens: List of subword tokens
            
        Returns:
            Reconstructed text string
        """
        if not isinstance(tokens, (list, tuple)) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("tokens must be a list of strings")
        return self.sp.DecodePieces(tokens)
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert text to token IDs.
        
        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sentence token ID
            add_eos: Whether to add end-of-sentence token ID
            
        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text).__name__}")
        return self.sp.EncodeAsIds(text, add_bos=add_bos, add_eos=add_eos)
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not isinstance(ids, (list, tuple)) or not all(isinstance(i, int) for i in ids):
            raise ValueError("ids must be a list of integers")
        return self.sp.DecodeIds(ids)
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return self.sp.GetPieceSize()
    
    def id_to_piece(self, id: int) -> str:
        """Convert a token ID to its corresponding subword piece."""
        if not isinstance(id, int):
            raise ValueError(f"id must be an integer, got {type(id).__name__}")
        return self.sp.IdToPiece(id)
    
    def piece_to_id(self, piece: str) -> int:
        """Convert a subword piece to its corresponding token ID."""
        if not isinstance(piece, str):
            raise ValueError(f"piece must be a string, got {type(piece).__name__}")
        return self.sp.PieceToId(piece)
    
    def save(self, model_path: Union[str, os.PathLike]) -> None:
        """Save the tokenizer model to disk.
        
        Args:
            model_path: Path where to save the model
        """
        self.sp.save(model_path)

def train_tokenizer(
    input_path: Union[str, os.PathLike, List[str], pd.DataFrame],
    output_prefix: str,
    vocab_size: int = 8000,
    model_type: str = 'bpe',
    character_coverage: float = 1.0,
    input_sentence_size: Optional[int] = None,
    shuffle_input_sentence: bool = True,
    user_defined_symbols: Optional[List[str]] = None,
    **kwargs
) -> None:
    """Train a new SentencePiece tokenizer.
    
    Args:
        input_path: Input file path, list of texts, or DataFrame with text data
        output_prefix: Prefix for output model files
        vocab_size: Size of the vocabulary
        model_type: 'bpe' or 'unigram'
        character_coverage: Amount of characters covered by the model (0.0-1.0)
        input_sentence_size: Maximum number of training sentences to use
        shuffle_input_sentence: Whether to shuffle the input sentences
        user_defined_symbols: List of user-defined symbols to include in the vocabulary
        **kwargs: Additional arguments passed to SentencePieceTrainer.Train()
        
    Returns:
        None. Saves the model to disk with the given prefix.
        
    Raises:
        ValueError: If input data is invalid or parameters are incorrect
        RuntimeError: If training fails
    """
    # Validate input
    if not isinstance(input_path, (str, os.PathLike, list, pd.DataFrame)):
        raise ValueError("input_path must be a file path, list of strings, or DataFrame")
        
    if model_type not in ('bpe', 'unigram'):
        raise ValueError("model_type must be either 'bpe' or 'unigram'")
    
    # Prepare temporary input file if input is not a file
    temp_file = None
    input_file = str(input_path)
    
    try:
        if isinstance(input_path, (list, pd.DataFrame)):
            # Create a temporary file for the input texts
            temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
            input_file = temp_file.name
            
            if isinstance(input_path, list):
                texts = input_path
            else:  # DataFrame
                # Try to find text columns automatically
                text_cols = [col for col in input_path.columns 
                           if isinstance(input_path[col].iloc[0], str)]
                if not text_cols:
                    raise ValueError("No text columns found in DataFrame")
                texts = []
                for col in text_cols:
                    texts.extend(input_path[col].dropna().astype(str).tolist())
            
            # Write texts to temporary file
            for text in texts:
                if not isinstance(text, str):
                    continue
                temp_file.write(f"{text}\n")
            temp_file.close()
        
        # Build the training command
        cmd = [
            f'--input={input_file}',
            f'--model_prefix={output_prefix}',
            f'--vocab_size={vocab_size}',
            f'--model_type={model_type}',
            f'--character_coverage={character_coverage}',
            '--pad_id=0',
            '--unk_id=1',
            '--bos_id=2',
            '--eos_id=3',
            '--pad_piece=[PAD]',
            '--unk_piece=[UNK]',
            '--bos_piece=[BOS]',
            '--eos_piece=[EOS]',
        ]
        
        # Add user-defined symbols if provided
        if user_defined_symbols:
            symbols = ','.join(user_defined_symbols)
            cmd.append(f'--user_defined_symbols={symbols}')
        
        # Add additional parameters
        if input_sentence_size is not None:
            cmd.append(f'--input_sentence_size={input_sentence_size}')
        if not shuffle_input_sentence:
            cmd.append('--shuffle_input_sentence=false')
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                cmd.append(f'--{key}={value}')
        
        # Train the model
        spm.SentencePieceTrainer.Train(' '.join(cmd))
        
        print(f"\nTraining complete. Model saved to:"
              f"\n  - {output_prefix}.model (model file)"
              f"\n  - {output_prefix}.vocab (vocabulary file)")
    
    finally:
        # Clean up temporary file if it was created
        if temp_file is not None and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

def load_parallel_data(
    source_path: Union[str, os.PathLike],
    target_path: Optional[Union[str, os.PathLike]] = None,
    source_col: str = 'nagamese',
    target_col: str = 'english',
    **kwargs
) -> pd.DataFrame:
    """Load and clean parallel text data for tokenizer training.
    
    Args:
        source_path: Path to source text file or DataFrame
        target_path: Path to target text file (if source_path is not a DataFrame)
        source_col: Column name for source text (if input is DataFrame)
        target_col: Column name for target text (if input is DataFrame)
        **kwargs: Additional arguments passed to pandas.read_csv()
        
    Returns:
        DataFrame with cleaned parallel texts
        
    Raises:
        FileNotFoundError: If input files are not found
        ValueError: If input files have different number of lines
    """
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        # Remove HTML tags and extra whitespace
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Handle different input types
    if target_path is None:
        # Assume source_path is a DataFrame
        if not isinstance(source_path, pd.DataFrame):
            raise ValueError("target_path must be provided when source_path is not a DataFrame")
        df = source_path.copy()
    else:
        # Read from file paths
        for path in [source_path, target_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        
        # Read files
        with open(source_path, 'r', encoding='utf-8') as f_src, \
             open(target_path, 'r', encoding='utf-8') as f_tgt:
            src_lines = [clean_text(line) for line in f_src if line.strip()]
            tgt_lines = [clean_text(line) for line in f_tgt if line.strip()]
        
        # Verify same number of lines
        if len(src_lines) != len(tgt_lines):
            raise ValueError("Input files must have the same number of lines")
        
        df = pd.DataFrame({
            source_col: src_lines,
            target_col: tgt_lines
        })
    
    # Clean text columns
    df[f'{source_col}_clean'] = df[source_col].apply(clean_text)
    df[f'{target_col}_clean'] = df[target_col].apply(clean_text)
    
    return df


def main():
    """Command-line interface for training and using the tokenizer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train or use a SentencePiece subword tokenizer.'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new tokenizer')
    train_parser.add_argument(
        'input',
        help='Input file path or directory containing text files'
    )
    train_parser.add_argument(
        'output_prefix',
        help='Output model prefix (will create .model and .vocab files)'
    )
    train_parser.add_argument(
        '--vocab-size',
        type=int,
        default=8000,
        help='Vocabulary size (default: 8000)'
    )
    train_parser.add_argument(
        '--model-type',
        choices=['bpe', 'unigram'],
        default='bpe',
        help='Tokenization model type (default: bpe)'
    )
    train_parser.add_argument(
        '--character-coverage',
        type=float,
        default=1.0,
        help='Character coverage (default: 1.0)'
    )
    
    # Tokenize command
    tokenize_parser = subparsers.add_parser('tokenize', help='Tokenize text')
    tokenize_parser.add_argument(
        'model',
        help='Path to the SentencePiece model file (.model)'
    )
    tokenize_parser.add_argument(
        'text',
        nargs='?',
        help='Text to tokenize (or read from stdin if not provided)'
    )
    tokenize_parser.add_argument(
        '--detokenize',
        action='store_true',
        help='Detokenize instead of tokenize'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            print(f"Training tokenizer with vocab_size={args.vocab_size}...")
            train_tokenizer(
                input_path=args.input,
                output_prefix=args.output_prefix,
                vocab_size=args.vocab_size,
                model_type=args.model_type,
                character_coverage=args.character_coverage
            )
            
        elif args.command == 'tokenize':
            tokenizer = SubwordTokenizer(args.model)
            
            if args.text:
                text = args.text
            else:
                # Read from stdin
                print("Enter text to tokenize (Ctrl+D to finish):")
                text = sys.stdin.read().strip()
            
            if not text:
                print("No input text provided")
                return
                
            if args.detokenize:
                # Split input into tokens if it's a string representation of a list
                if text.startswith('[') and text.endswith(']'):
                    import ast
                    try:
                        tokens = ast.literal_eval(text)
                        if not isinstance(tokens, list):
                            raise ValueError("Input must be a list of tokens")
                    except (ValueError, SyntaxError) as e:
                        tokens = text.split()
                else:
                    tokens = text.split()
                
                result = tokenizer.detokenize(tokens)
                print("Detokenized:")
                print(result)
            else:
                tokens = tokenizer.tokenize(text)
                print("Tokens:")
                print(tokens)
                print("\nToken IDs:")
                print(tokenizer.encode(text))
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

# file: subword_tokenizer.py

import os
import re
from typing import List, Optional

import pandas as pd
import sentencepiece as spm

def load_data_for_spm(filepath: str) -> Optional[pd.DataFrame]:
    """Loads and cleans data specifically for SentencePiece training."""
    if not os.path.exists(filepath):
        print(f"Error: The file at {filepath} was not found.")
        return None
    df = pd.read_csv(filepath)
    def clean_text(text: str) -> str:
        """Clean text by removing HTML tags and extra whitespace."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['english_cleaned'] = df['english'].apply(clean_text)
    df['nagamese_cleaned'] = df['nagamese'].apply(clean_text)
    return df

def train_sentencepiece_model(
    df: pd.DataFrame,
    model_prefix: str = 'naga_eng_bpe',
    vocab_size: int = 8000
) -> None:
    """
    Trains a joint SentencePiece BPE model on the Nagamese and English text.

    Args:
        df (pd.DataFrame): DataFrame containing 'nagamese_cleaned' and 'english_cleaned' columns.
        model_prefix (str): Prefix for the saved model files (.model, .vocab).
        vocab_size (int): The target size of the vocabulary.
    """
    # 1. Prepare a joint corpus file
    joint_corpus_path = 'joint_corpus.txt'
    with open(joint_corpus_path, 'w', encoding='utf-8') as f:
        for text in df['nagamese_cleaned'].tolist():
            f.write(f"{text}\n")
        for text in df['english_cleaned'].tolist():
            f.write(f"{text}\n")

    print(f"Joint corpus file created at '{joint_corpus_path}'")

    # 2. Train the SentencePiece model
    spm.SentencePieceTrainer.Train(
        f'--input={joint_corpus_path} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type=bpe '
        f'--character_coverage=1.0'
    )
    print(f"SentencePiece model trained. Files '{model_prefix}.model' and '{model_prefix}.vocab' are saved.")

class SubwordTokenizer:
    """A wrapper for a trained SentencePiece model."""
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model file not found at: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into subword pieces."""
        return self.sp.encode_as_pieces(text)

    def detokenize(self, pieces: List[str]) -> str:
        """Converts a list of pieces back into a string."""
        return self.sp.decode_pieces(pieces)

if __name__ == '__main__':
    # --- 1. Load Data ---
    dataframe = load_data_for_spm('merged.csv')

    if dataframe is not None:
        # --- 2. Train the Model ---
        model_prefix = 'nagamese_english_spm'
        train_sentencepiece_model(dataframe, model_prefix=model_prefix, vocab_size=8000)

        # --- 3. Load and Test the Tokenizer ---
        print("\n--- Loading and Testing the Subword Tokenizer ---")
        try:
            tokenizer = SubwordTokenizer(f'{model_prefix}.model')

            test_sentence = "abraham laga chokra david laga chokra"
            tokens = tokenizer.tokenize(test_sentence)
            reconstructed = tokenizer.detokenize(tokens)

            print(f"\nOriginal: {test_sentence}")
            print(f"Tokens: {tokens}")
            print(f"Reconstructed: {reconstructed}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
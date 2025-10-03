"""Word alignment module for Nagamese-English parallel text.

This module provides functionality for aligning words between Nagamese and English
sentences using the awesome-align model. It includes utilities for data loading,
cleaning, and running the alignment process.

Example:
    >>> from naganlp.word_aligner import align_parallel_texts
    >>> df = align_parallel_texts('nagamese.txt', 'english.txt')
    >>> print(df[['nagamese', 'english', 'alignment']].head())
"""

import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Default configuration
DEFAULT_CONFIG = {
    'model_name': 'bert-base-multilingual-cased',
    'batch_size': 32,
    'device': 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu',
    'extraction': 'softmax',
    'output_file': 'word_alignments.txt'
}

def load_parallel_data(
    source_path: str,
    target_path: Optional[str] = None,
    source_col: str = 'nagamese',
    target_col: str = 'english',
    output_format: str = 'dataframe'
) -> Union[pd.DataFrame, List[Tuple[str, str]]]:
    """Load and clean parallel text data for alignment.
    
    Args:
        source_path: Path to the source (Nagamese) text file or DataFrame
        target_path: Path to the target (English) text file (if source_path is not a DataFrame)
        source_col: Column name for source text (if input is DataFrame)
        target_col: Column name for target text (if input is DataFrame)
        output_format: Return format - 'dataframe' or 'list' of tuples
        
    Returns:
        Union[pd.DataFrame, List[Tuple[str, str]]]: Cleaned parallel sentences
        
    Raises:
        FileNotFoundError: If either input file is not found
        ValueError: If input files have different number of lines or invalid format
    """
    def clean_text(text: str) -> str:
        """Normalize whitespace and clean text."""
        if not isinstance(text, str):
            return ""
        return re.sub(r'\s+', ' ', text).strip()
    
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
    
    # Return in requested format
    if output_format.lower() == 'dataframe':
        return df
    else:
        return list(zip(
            df[f'{source_col}_clean'],
            df[f'{target_col}_clean']
        ))

def align_parallel_texts(
    source_texts: Union[str, List[str], pd.DataFrame],
    target_texts: Optional[Union[str, List[str]]] = None,
    source_col: str = 'nagamese',
    target_col: str = 'english',
    config: Optional[Dict[str, Any]] = None,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """Align words between parallel Nagamese and English texts.
    
    This function uses awesome-align to find word-level alignments between
    parallel sentences in Nagamese and English.
    
    Args:
        source_texts: Path to Nagamese text file, list of sentences, or DataFrame
        target_texts: Path to English text file or list of sentences (if source is not DataFrame)
        source_col: Column name for source text (if input is DataFrame)
        target_col: Column name for target text (if input is DataFrame)
        config: Optional configuration dictionary
        output_file: Optional path to save alignments (default: 'word_alignments.txt')
        
    Returns:
        pd.DataFrame: DataFrame containing aligned sentences and their word alignments
        
    Raises:
        ImportError: If awesome-align is not installed
        ValueError: If inputs are invalid or alignment fails
        
    Example:
        >>> # From files
        >>> df = align_parallel_texts('nagamese.txt', 'english.txt')
        >>> # From lists
        >>> df = align_parallel_texts(['moi school te jai'], ['I go to school'])
    """
    # Merge with default config
    config = {**DEFAULT_CONFIG, **(config or {})}
    output_file = output_file or config['output_file']
    
    # Handle different input types
    if isinstance(source_texts, pd.DataFrame):
        df = source_texts.copy()
        required_cols = {source_col, target_col}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
    else:
        if target_texts is None:
            raise ValueError("target_texts must be provided when source_texts is not a DataFrame")
        
        # Convert file paths to lists if needed
        if isinstance(source_texts, str) and os.path.isfile(source_texts):
            with open(source_texts, 'r', encoding='utf-8') as f:
                source_texts = [line.strip() for line in f if line.strip()]
                
        if isinstance(target_texts, str) and os.path.isfile(target_texts):
            with open(target_texts, 'r', encoding='utf-8') as f:
                target_texts = [line.strip() for line in f if line.strip()]
        
        # Create DataFrame
        df = pd.DataFrame({
            source_col: source_texts,
            target_col: target_texts
        })
    
    # Clean text columns
    df[f'{source_col}_clean'] = df[source_col].apply(_clean_text)
    df[f'{target_col}_clean'] = df[target_col].apply(_clean_text)
    
    # Prepare temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare input file for awesome-align
        input_file = os.path.join(tmpdir, 'input.txt')
        with open(input_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f"{row[f'{target_col}_clean']}\t{row[f'{source_col}_clean']}\n")
        
        # Build command
        align_script = 'awesome-align/run_align.py'
        if not os.path.exists(align_script):
            raise ImportError(
                "awesome-align not found. Install it with: "
                "git clone https://github.com/neulab/awesome-align.git\n"
                "cd awesome-align\n"
                "pip install -e ."
            )
            
        cmd = [
            'python', align_script,
            '--model_name_or_path', config['model_name'],
            '--data_file', input_file,
            '--output_file', output_file,
            '--extraction', config['extraction'],
            '--batch_size', str(config['batch_size']),
            '--device', config['device']
        ]
        
        # Run alignment
        print(f"\n--- Starting Word Alignment (this may take several minutes) ---")
        try:
            process = subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding='utf-8'
            )
            print(process.stdout)
            print(f"--- Alignment Complete. Results saved to '{output_file}' ---")
        except subprocess.CalledProcessError as e:
            print("--- An error occurred during alignment. ---")
            print(f"Return Code: {e.returncode}")
            print("----- STDOUT -----")
            print(e.stdout)
            print("----- STDERR -----")
            print(e.stderr)
            raise RuntimeError("Alignment process failed") from e
    
    # Read and process alignments
    alignments = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            alignments = [line.strip() for line in f]
    except Exception as e:
        raise IOError(f"Failed to read alignment file: {e}") from e
    
    # Add alignments to DataFrame
    df['alignment'] = alignments
    return df

def _clean_text(text: str) -> str:
    """Clean and normalize text for alignment."""
    if not isinstance(text, str):
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _parse_alignment(alignment_str: str) -> List[Tuple[int, int]]:
    """Parse alignment string into list of index pairs."""
    if not alignment_str:
        return []
    try:
        return [tuple(map(int, pair.split('-'))) for pair in alignment_str.split()]
    except (ValueError, AttributeError):
        return []

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Align Nagamese-English parallel texts using awesome-align.'
    )
    parser.add_argument(
        'source_file',
        help='Path to file containing Nagamese sentences (one per line)'
    )
    parser.add_argument(
        'target_file',
        help='Path to file containing English sentences (one per line)'
    )
    parser.add_argument(
        '-o', '--output',
        default='word_alignments.txt',
        help='Output file path for alignments (default: word_alignments.txt)'
    )
    parser.add_argument(
        '--model',
        default='bert-base-multilingual-cased',
        help='Pretrained model name (default: bert-base-multilingual-cased)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for alignment (default: 32)'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Aligning {args.source_file} and {args.target_file}...")
        df = align_parallel_texts(
            args.source_file,
            args.target_file,
            config={
                'model_name': args.model,
                'batch_size': args.batch_size
            },
            output_file=args.output
        )
        print(f"Successfully aligned {len(df)} sentence pairs")
        print(f"Alignments saved to: {args.output}")
        
        # Display sample of alignments
        print("\n--- Sample Alignments ---")
        for i, row in df.head(3).iterrows():
            print(f"\nSource: {row['nagamese']}")
            print(f"Target: {row['english']}")
            print(f"Alignments: {row['alignment']}")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

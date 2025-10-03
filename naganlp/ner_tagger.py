"""
Named Entity Recognition (NER) module for Nagamese text.

This module provides a CRF-based NER tagger that can identify and classify named entities in Nagamese text.
It supports loading pre-trained models from Hugging Face Hub and can be used for inference.

Available Models:
    - Default model: 'agnivamaiti/naganlp-ner-crf-tagger' (Hugging Face)
    - Additional models can be found at: https://huggingface.co/agnivamaiti/

Example:
    >>> from naganlp.ner_tagger import NerTagger
    >>> # Initialize with default model from Hugging Face Hub
    >>> tagger = NerTagger()
    >>> # Tag a sentence
    >>> entities = tagger.tag("Edgar Allan Poe ekta bishi bhal writer thakishe.")
    >>> for ent in entities:
    ...     print(f"{ent['word']} -> {ent['entity']}")
    
    >>> # Load a specific model from Hugging Face Hub
    >>> tagger = NerTagger(model_id="agnivamaiti/naganlp-ner-crf-tagger")
"""

import os
import re
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Sequence

import nltk
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from nltk.tokenize import word_tokenize
from sklearn_crfsuite import CRF

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Default configuration
DEFAULT_CONFIG = {
    # Model parameters
    'algorithm': 'lbfgs',
    'c1': 0.1,  # L1 regularization coefficient
    'c2': 0.1,  # L2 regularization coefficient
    'max_iterations': 100,
    'all_possible_transitions': True,
    
    # Default model from Hugging Face Hub
    'default_model_id': 'agnivamaiti/naganlp-ner-crf-tagger',
    'model_filename': 'nagamese_ner_case_insensitive.pkl',
    
    # IOB tags for NER
    'iob_tags': ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
}

# Type aliases
Token = str
Label = str
TokenWithPos = Tuple[str, str]
TokenWithLabel = Tuple[str, str, str]
Sentence = List[TokenWithLabel]
Features = List[Dict[str, Any]]


class NerTagger:
    """A CRF-based Named Entity Recognition tagger for Nagamese text.
    
    This class provides functionality for loading pre-trained CRF-based NER models
    from Hugging Face Hub and using them to identify named entities in Nagamese text.
    
    Args:
        model_id: Hugging Face model ID (e.g., 'username/model-name') or path to local model
        model_path: (Deprecated) Path to a local model file
        **kwargs: Additional configuration overrides
    
    Example:
        >>> # Load default model from Hugging Face Hub
        >>> tagger = NerTagger()
        >>> 
        >>> # Tag a sentence
        >>> entities = tagger.tag("Edgar Allan Poe ekta bishi bhal writer thakishe.")
        >>> for ent in entities:
        ...     print(f"{ent['word']} -> {ent['entity']}")
        
        >>> # Load a specific model from Hugging Face Hub
        >>> tagger = NerTagger(model_id="agnivamaiti/naganlp-ner-crf-tagger")
    """
    
    def __init__(
        self, 
        model_id: Optional[str] = None,
        model_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """Initialize the NER tagger with a pre-trained model."""
        # Merge config with defaults and provided kwargs
        self.config = {**DEFAULT_CONFIG, **kwargs}
        
        # Initialize components
        self.model = None
        self._is_initialized = False
        
        # Load NLTK resources
        self._ensure_nltk_resources()
        
        # Load model if ID or path is provided
        if model_id:
            self.load_from_hub(model_id)
        elif model_path:
            self.load(model_path)
    
    def _ensure_nltk_resources(self) -> None:
        """Ensure required NLTK resources are downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def load_from_hub(self, model_id: str) -> None:
        """Load a pre-trained NER model from Hugging Face Hub.
        
        Args:
            model_id: Hugging Face model ID (e.g., 'username/model-name')
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Download the model from Hugging Face Hub
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=self.config['model_filename']
            )
            
            # Load the model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self._is_initialized = True
            print(f"Loaded NER model from Hugging Face Hub: {model_id}")
            
        except Exception as e:
            self._is_initialized = False
            error_msg = f"Failed to load model from Hugging Face Hub: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    def load(self, model_path: Union[str, Path]) -> None:
        """Load a pre-trained NER model from disk.
        
        Args:
            model_path: Path to the model file
            
        Raises:
            FileNotFoundError: If model file is not found
            RuntimeError: If model loading fails
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self._is_initialized = True
            print(f"Loaded NER model from {model_path}")
            
        except Exception as e:
            self._is_initialized = False
            error_msg = f"Failed to load NER model from {model_path}: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save the NER model to disk.
        
        Args:
            output_path: Path to save the model file
            
        Raises:
            RuntimeError: If model is not initialized or saving fails
        """
        if not self._is_initialized or not self.model:
            raise RuntimeError("Model is not initialized. Nothing to save.")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            print(f"NER model saved to {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to save NER model to {output_path}: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    def _word2features(self, sent: List[str], i: int) -> Dict[str, Any]:
        """Extract features for a word in a sentence.
        
        Args:
            sent: List of words
            i: Index of the current word
            
        Returns:
            Dictionary of features for the word
        """
        word = sent[i]
        features = {
            'bias': 1.0,
            'word': word,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isdigit()': word.isdigit(),
            'word.istitle()': word.istitle(),
            'word.isupper()': word.isupper(),
        }
        
        # Add context features
        if i > 0:
            prev_word = sent[i-1]
            features.update({
                '-1:word': prev_word,
                '-1:word.lower()': prev_word.lower(),
                '-1:word.istitle()': prev_word.istitle(),
                '-1:word.isupper()': prev_word.isupper(),
            })
        else:
            features['BOS'] = True
            
        if i < len(sent)-1:
            next_word = sent[i+1]
            features.update({
                '+1:word': next_word,
                '+1:word.lower()': next_word.lower(),
                '+1:word.istitle()': next_word.istitle(),
                '+1:word.isupper()': next_word.isupper(),
            })
        else:
            features['EOS'] = True
            
        return features
    
    def _sent2features(self, sent: List[str]) -> List[Dict[str, Any]]:
        """Convert a sentence to features.
        
        Args:
            sent: List of words
            
        Returns:
            List of feature dictionaries, one per word
        """
        return [self._word2features(sent, i) for i in range(len(sent))]
    
    def _sent2labels(self, sent: List[TokenWithLabel]) -> List[Label]:
        """Extract labels from a labeled sentence.
        
        Args:
            sent: List of (word, pos, label) tuples
            
        Returns:
            List of labels
        """
        return [label for _, _, label in sent]
    
    def _sent2tokens(self, sent: List[TokenWithLabel]) -> List[Token]:
        """Extract tokens from a labeled sentence.
        
        Args:
            sent: List of (word, pos, label) tuples
            
        Returns:
            List of tokens
        """
        return [token for token, _, _ in sent]
    
    def train(
        self,
        train_data: List[List[Tuple[str, str]]],
        **kwargs
    ) -> 'NerTagger':
        """Train the NER model on the provided data.
        
        Note: This method is kept for backward compatibility but will raise a NotImplementedError
        as training is not supported in this version. Use the Hugging Face model for inference.
        
        Args:
            train_data: List of training sentences, where each sentence is a list of (word, label) tuples
            **kwargs: Additional arguments (ignored)
            
        Raises:
            NotImplementedError: Always raised as training is not supported
        """
        raise NotImplementedError(
            "Training is not supported in this version. "
            "Please use the pre-trained model from Hugging Face Hub."
        )
    
    def _evaluate(
        self,
        y_true: List[List[Label]],
        y_pred: List[List[Label]],
        split: str = 'test'
    ) -> Dict[str, Any]:
        """Evaluate the model on the given data.
        
        Args:
            y_true: List of true label sequences
            y_pred: List of predicted label sequences
            split: Name of the data split (e.g., 'train', 'val', 'test')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Flatten the lists of sequences
        y_true_flat = [label for seq in y_true for label in seq]
        y_pred_flat = [label for seq in y_pred for label in seq]
        
        # Get unique labels (excluding 'O' for per-class metrics)
        labels = sorted(set(y_true_flat) - {'O'})
        
        # Calculate metrics
        report = classification_report(
            y_true_flat, 
            y_pred_flat, 
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Calculate overall metrics (micro-averaged)
        overall = {
            'precision': report['micro avg']['precision'],
            'recall': report['micro avg']['recall'],
            'f1': report['micro avg']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        # Prepare per-class metrics
        per_class = {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1': report[label]['f1-score'],
                'support': report[label]['support']
            }
            for label in labels
        }
        
        # Log metrics
        logger.info(f"{split.capitalize()} metrics:")
        logger.info(f"  - F1: {overall['f1']:.4f}")
        logger.info(f"  - Precision: {overall['precision']:.4f}")
        logger.info(f"  - Recall: {overall['recall']:.4f}")
        logger.info(f"  - Accuracy: {overall['accuracy']:.4f}")
        
        return {
            'overall': overall,
            'per_class': per_class,
            'report': report
        }
    
    def tag(
        self, 
        text: str, 
        return_tokens: bool = False,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[str]]]:
        """Tag named entities in the input text.
        
        Args:
            text: Input text to tag
            return_tokens: Whether to return the tokenized text along with entities
            batch_size: Number of sentences to process in parallel (not used in this implementation)
            show_progress: Whether to show a progress bar (not used in this implementation)
            
        Returns:
            List of dictionaries containing entity information, and optionally the tokenized text.
            Each entity dictionary contains:
            - 'word': The entity text
            - 'entity': The entity label (e.g., 'B-PER', 'I-LOC')
            - 'start': Start position in the original text
            - 'end': End position in the original text
            - 'score': Confidence score (1.0 for CRF models)
            
        Raises:
            RuntimeError: If model is not initialized or tagging fails
        """
        if not self._is_initialized or not self.model:
            raise RuntimeError("Model is not initialized. Please load a model first.")
        
        try:
            # Tokenize the input text
            tokens = word_tokenize(text)
            if not tokens:
                return [] if not return_tokens else ([], [])
            
            # Convert tokens to lowercase for prediction (as the model was trained on lowercase)
            lower_tokens = [t.lower() for t in tokens]
            
            # Create dummy POS tags (not used in this version)
            sent = [(token, 'UNK') for token in lower_tokens]
            
            # Extract features and predict
            features = self._sent2features([t[0] for t in sent])
            labels = self.model.predict_single(features)
            
            # Convert to entity format
            entities = []
            current_entity = None
            
            for i, (token, label) in enumerate(zip(tokens, labels)):
                if label == 'O':
                    if current_entity is not None:
                        entities.append(current_entity)
                        current_entity = None
                    continue
                
                # Handle B- and I- prefixes
                if label.startswith('B-') or current_entity is None or label[2:] != current_entity['entity'][2:]:
                    if current_entity is not None:
                        entities.append(current_entity)
                    
                    current_entity = {
                        'entity': label,
                        'word': token,
                        'start': text.find(token),
                        'end': text.find(token) + len(token),
                        'score': 1.0  # CRF doesn't provide probabilities by default
                    }
                else:
                    # Continue the current entity
                    current_entity['word'] += ' ' + token
                    # Update end position to the end of the current token
                    current_entity['end'] = text.find(token, current_entity['end']) + len(token)
            
            # Add the last entity if exists
            if current_entity is not None:
                entities.append(current_entity)
            
            return (entities, tokens) if return_tokens else entities
            
        except Exception as e:
            error_msg = f"Error during NER tagging: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def tag_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """Tag named entities in a batch of texts.
        
        Args:
            texts: List of input texts to tag
            batch_size: Number of texts to process in parallel (not used in this implementation)
            show_progress: Whether to show a progress bar
            
        Returns:
            List of lists of entity dictionaries, one per input text
            
        Raises:
            RuntimeError: If model is not initialized
        """
        if not self._is_initialized or not self.model:
            raise RuntimeError("Model is not initialized. Please load or train a model first.")
        
        results = []
        iterator = tqdm(texts, desc="Tagging", disable=not show_progress)
        
        for text in iterator:
            try:
                entities = self.tag(text, return_tokens=False)
                results.append(entities)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                results.append([])
        
        return results


def read_conll_ner_data(
    file_path: Union[str, Path],
    sep: str = '\t',
    encoding: str = 'utf-8',
    word_col: int = 0,
    pos_col: int = 1,
    label_col: int = 2,
    skip_comments: bool = True,
    lowercase: bool = True
) -> List[Sentence]:
    """Read NER data in CoNLL format.
    
    Args:
        file_path: Path to the CoNLL file
        sep: Column separator
        encoding: File encoding
        word_col: Column index for words
        pos_col: Column index for POS tags
        label_col: Column index for NER labels
        skip_comments: Whether to skip comment lines (starting with #)
        lowercase: Whether to lowercase words
        
    Returns:
        List of sentences, where each sentence is a list of (word, pos, label) tuples
        
    Raises:
        FileNotFoundError: If the input file does not exist
        ValueError: If the file is empty or has invalid format
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
                
            if skip_comments and line.startswith('#'):
                continue
            
            # Split the line into columns
            parts = line.split(sep)
            
            # Skip malformed lines
            if len(parts) <= max(word_col, pos_col, label_col):
                logger.warning(f"Skipping malformed line {line_num}: {line}")
                continue
            
            word = parts[word_col]
            pos = parts[pos_col] if pos_col < len(parts) else 'UNK'
            label = parts[label_col] if label_col < len(parts) else 'O'
            
            if lowercase:
                word = word.lower()
            
            current_sentence.append((word, pos, label))
    
    # Add the last sentence if not empty
    if current_sentence:
        sentences.append(current_sentence)
    
    if not sentences:
        raise ValueError(f"No valid sentences found in {file_path}")
    
    logger.info(f"Read {len(sentences)} sentences from {file_path}")
    return sentences


def train_ner_model(*args, **kwargs):
    """Train a new NER model (not supported in this version).
    
    This function is kept for backward compatibility but will raise a NotImplementedError
    as training is not supported in this version. Use the Hugging Face model for inference.
    
    Raises:
        NotImplementedError: Always raised as training is not supported
    """
    raise NotImplementedError(
        "Training is not supported in this version. "
        "Please use the pre-trained model from Hugging Face Hub."
    )


def main():
    """Command-line interface for training and using the NER tagger."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Named Entity Recognition for Nagamese')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new NER model')
    train_parser.add_argument('train_file', help='Path to training data file (CoNLL format)')
    train_parser.add_argument('--output-dir', required=True, help='Directory to save the model')
    train_parser.add_argument('--val-file', help='Path to validation data file (CoNLL format)')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of training data to use for validation if val-file is not provided')
    train_parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    
    # Tag command
    tag_parser = subparsers.add_parser('tag', help='Tag text with named entities')
    tag_parser.add_argument('--model-dir', help='Path to the trained model directory', default=None)
    tag_parser.add_argument('--input-file', help='Path to input text file (one sentence per line)')
    tag_parser.add_argument('--output-file', help='Path to save the tagged output')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_ner_model(
            train_file=args.train_file,
            output_dir=args.output_dir,
            val_file=args.val_file,
            test_size=args.test_size,
            random_state=args.random_state
        )
    
    elif args.command == 'tag':
        # Load the model
        if args.model_dir:
            tagger = NerTagger(model_path=args.model_dir)
        else:
            tagger = NerTagger()
        
        # Read input
        if args.input_file:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            print("Enter text to tag (press Ctrl+D when done):")
            texts = [line.strip() for line in sys.stdin if line.strip()]
    
        # Tag the text
        results = []
        for text in texts:
            entities = tagger.tag(text)
            results.append({
                'text': text,
                'entities': entities
            })
    
        # Output the results
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output_file}")
        else:
            for result in results:
                print(f"\nText: {result['text']}")
                if result['entities']:
                    print("Entities:")
                    for ent in result['entities']:
                        print(f"  {ent['word']} -> {ent['entity']}")
                else:
                    print("No named entities found.")
    
    else:
        parser.print_help()
    main()

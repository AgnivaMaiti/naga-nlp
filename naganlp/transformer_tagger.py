"""Transformer-based Part-of-Speech Tagger for Nagamese.

This module provides a POS tagging system using pre-trained transformer models
fine-tuned on Nagamese text. It supports both inference and training of new models.

Available Models:
    - Default model: 'agnivamaiti/naga-pos-tagger' (Hugging Face)
    - Additional models can be found at: https://huggingface.co/agnivamaiti/

Example:
    >>> from naganlp.transformer_tagger import PosTagger
    >>> # Initialize with default model
    >>> tagger = PosTagger()
    >>> # Tag a sentence
    >>> result = tagger.tag("moi school te jai")
    >>> print(result)
    [{'word': 'moi', 'score': 0.998, 'entity': 'PRON', 'index': 1, ...}]
    
    >>> # Train a new model
    >>> from naganlp.transformer_tagger import train_tagger
    >>> train_tagger("path/to/conll_file.txt", "your-username/naga-pos-model")
"""

import os
import json
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_metric
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    EvalPrediction,
    set_seed
)
from transformers.utils import logging as hf_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()

# Default configuration
DEFAULT_CONFIG = {
    'model_name': 'agnivamaiti/naga-pos-tagger',  # Default pre-trained model
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_seq_length': 128,
    'device': 0 if torch.cuda.is_available() else -1,
    'label_all_tokens': True,
    'ignore_mismatched_sizes': True,
}

# Available pre-trained models
PRETRAINED_MODELS = {
    'default': 'agnivamaiti/naga-pos-tagger',
    'base': 'agnivamaiti/naga-pos-tagger',
}

class PosTagger:
    """A transformer-based Part-of-Speech tagger for Nagamese.
    
    This class provides an easy-to-use interface for POS tagging using pre-trained
    transformer models. It handles tokenization, inference, and post-processing.
    
    Available models:
        - 'agnivamaiti/naga-pos-tagger' (default)
        - Custom model path or Hugging Face model ID
        
    Example:
        >>> # Initialize with default model
        >>> tagger = PosTagger()
        >>> 
        >>> # Initialize with custom model
        >>> custom_tagger = PosTagger("path/to/model")
        >>> 
        >>> # Tag a sentence
        >>> result = tagger.tag("moi school jai")
        >>> print(result)
        [
            {'word': 'moi', 'score': 0.998, 'entity': 'PRON', 'index': 1},
            {'word': 'school', 'score': 0.995, 'entity': 'NOUN', 'index': 2},
            {'word': 'jai', 'score': 0.997, 'entity': 'VERB', 'index': 3}
        ]
    """
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Union[int, str] = "auto",
        aggregation_strategy: str = "simple",
        use_auth_token: Optional[Union[bool, str]] = None,
        **kwargs
    ):
        """Initialize the POS tagger.
        
        Args:
            model_name_or_path: Path to a local model directory or Hugging Face model ID.
                               If None, uses the default model.
            device: Device to run the model on (-1 for CPU, 0 for GPU, 'auto' for auto-detect).
            aggregation_strategy: Strategy for aggregating subword tokens.
                                 Options: 'simple', 'first', 'average', 'max'.
            use_auth_token: Hugging Face authentication token for private models.
            **kwargs: Additional arguments passed to the token classification pipeline.
        """
        self.model_name_or_path = model_name_or_path or PRETRAINED_MODELS['default']
        self.aggregation_strategy = aggregation_strategy
        self.use_auth_token = use_auth_token
        self._initialize_device(device)
        self._initialize_model(**kwargs)
    
    def _initialize_device(self, device: Union[int, str]) -> None:
        """Initialize the device for model inference.
        
        Args:
            device: Device specification (int, 'auto', or 'cuda'/'cpu').
            
        Raises:
            ValueError: If device specification is invalid.
        """
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        elif isinstance(device, str) and device.startswith('cuda'):
            self.device = int(device.split(':')[-1]) if ':' in device else 0
        elif device in ('cpu', -1):
            self.device = -1
        else:
            try:
                self.device = int(device)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid device specification: {device}. "
                    "Expected 'auto', 'cpu', 'cuda', 'cuda:0', or an integer."
                ) from e
        
        # Log device information
        if self.device == -1:
            logger.info("Using CPU for inference")
        else:
            logger.info(f"Using GPU {self.device} for inference")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(self.device)}")
    
    def _initialize_model(self, **kwargs) -> None:
        """Initialize the token classification pipeline.
        
        Raises:
            RuntimeError: If model loading fails.
            ValueError: If model configuration is invalid.
        """
        try:
            logger.info(f"Loading model: {self.model_name_or_path}")
            
            # Load tokenizer with error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    use_auth_token=self.use_auth_token,
                    use_fast=True
                )
            except OSError as e:
                logger.warning(
                    f"Could not load tokenizer for {self.model_name_or_path}. "
                    "Using default tokenizer."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    PRETRAINED_MODELS['default'],
                    use_auth_token=self.use_auth_token
                )
            
            # Load model configuration
            try:
                config = AutoConfig.from_pretrained(
                    self.model_name_or_path,
                    use_auth_token=self.use_auth_token
                )
                
                # Ensure the model is for token classification
                if not hasattr(config, 'id2label') or not config.id2label:
                    logger.warning(
                        "Model config does not contain id2label mapping. "
                        "Using default POS tags."
                    )
                    config.id2label = {
                        0: "ADJ", 1: "ADP", 2: "ADV", 3: "AUX", 4: "CCONJ",
                        5: "DET", 6: "INTJ", 7: "NOUN", 8: "NUM", 9: "PART",
                        10: "PRON", 11: "PROPN", 12: "PUNCT", 13: "SCONJ",
                        14: "SYM", 15: "VERB", 16: "X"
                    }
                    config.label2id = {v: k for k, v in config.id2label.items()}
                
                # Load model with config
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name_or_path,
                    config=config,
                    use_auth_token=self.use_auth_token,
                    ignore_mismatched_sizes=True
                )
                
                # Update id2label and label2id from the model's config
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
                
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
            
            # Initialize pipeline
            self.pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy=self.aggregation_strategy,
                framework="pt",
                **kwargs
            )
            
            logger.info(f"Successfully loaded model: {self.model_name_or_path}")
            logger.info(f"Available labels: {list(self.label2id.keys())}")
            
        except Exception as e:
            error_msg = (
                f"Failed to load model '{self.model_name_or_path}'. "
                f"Error: {str(e)}\n"
                "Please ensure the model exists and is compatible with the token-classification task.\n"
                f"Available pre-trained models: {', '.join(PRETRAINED_MODELS.keys())}"
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def tag(
        self,
        text: Union[str, List[str]],
        batch_size: int = 32,
        return_offsets: bool = False,
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Tag text with part-of-speech labels.
        
        Args:
            text: Input text or list of texts to tag.
            batch_size: Number of texts to process in parallel.
            return_offsets: Whether to include character offsets in the output.
            **kwargs: Additional arguments passed to the pipeline.
            
        Returns:
            For a single text: List of dictionaries containing word, entity, score, etc.
            For multiple texts: List of lists of dictionaries (one list per text).
            
        Raises:
            ValueError: If input is invalid.
            RuntimeError: If tagging fails.
            
        Example:
            >>> tagger = PosTagger()
            >>> # Tag a single sentence
            >>> result = tagger.tag("moi school jai")
            >>> # Tag multiple sentences
            >>> results = tagger.tag(["moi school jai", "tumi kile aha nai?"])
        """
        if not text:
            return [] if isinstance(text, str) else [[]]
            
        if not isinstance(text, (str, list)):
            raise ValueError(
                f"text must be a string or list of strings, got {type(text).__name__}"
            )
            
        is_batch = isinstance(text, list)
        if is_batch and not all(isinstance(t, str) for t in text):
            raise ValueError("All items in the text list must be strings")
            
        try:
            # Process the text
            results = self.pipeline(
                text,
                batch_size=batch_size,
                **kwargs
            )
            
            # Post-process results
            if is_batch:
                if not results or not isinstance(results[0], list):
                    results = [results] if results else []
                
                # Ensure consistent structure for batch results
                processed_results = []
                for result in results:
                    if not result:
                        processed_results.append([])
                        continue
                    
                    # Convert to list if needed
                    result_list = result if isinstance(result, list) else [result]
                    
                    # Add index if missing
                    for i, item in enumerate(result_list):
                        if 'index' not in item:
                            item['index'] = i + 1
                    
                    processed_results.append(result_list)
                
                return processed_results
            
            else:  # Single text input
                if not results:
                    return []
                
                # Convert to list if needed
                result_list = results if isinstance(results, list) else [results]
                
                # Add index if missing
                for i, item in enumerate(result_list):
                    if 'index' not in item:
                        item['index'] = i + 1
                
                return result_list
                
        except Exception as e:
            error_msg = f"Failed to tag text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Provide more helpful error messages for common issues
            if "CUDA out of memory" in str(e):
                error_msg += (
                    "\nOut of GPU memory. Try reducing batch_size or using CPU.\n"
                    "Example: tagger = PosTagger(device='cpu')"
                )
            elif "Connection error" in str(e):
                error_msg += (
                    "\nFailed to download model. Check your internet connection.\n"
                    "If using a private model, ensure you're authenticated with Hugging Face Hub."
                )
            
            raise RuntimeError(error_msg) from e
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike]) -> None:
        """Save the model and tokenizer to a directory.
        
        Args:
            save_directory: Directory to save the model and tokenizer to.
            
        Raises:
            OSError: If the directory cannot be created.
            RuntimeError: If saving fails.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save config
            config = {
                'model_type': 'pos_tagger',
                'model_name_or_path': str(self.model_name_or_path),
                'id2label': self.id2label,
                'label2id': self.label2id,
                'framework': 'pytorch',
                'pipeline_tag': 'token-classification',
            }
            
            with open(save_path / 'config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            error_msg = f"Failed to save model to {save_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[str] = None,
        **kwargs
    ) -> 'PosTagger':
        """Load a pre-trained POS tagger.
        
        This is a convenience method that creates a new instance of PosTagger
        with the specified pre-trained model.
        
        Args:
            model_name_or_path: Model name or path. If None, uses the default model.
            **kwargs: Additional arguments passed to the PosTagger constructor.
            
        Returns:
            A new instance of PosTagger.
            
        Example:
            >>> from naganlp import PosTagger
            >>> # Load default model
            >>> tagger = PosTagger.from_pretrained()
            >>> # Load custom model
            >>> custom_tagger = PosTagger.from_pretrained("path/to/model")
        """
        return cls(model_name_or_path=model_name_or_path, **kwargs)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Alias for the tag method to make the tagger callable.
        
        Example:
            >>> tagger = PosTagger()
            >>> result = tagger("moi school jai")  # Same as tagger.tag()
        """
        return self.tag(text, **kwargs)


def read_conll(
    file_path: Union[str, os.PathLike],
    delimiter: str = '\t',
    encoding: str = 'utf-8',
    sentence_delimiter: str = '-DOCSTART- -X- O',
    max_length: Optional[int] = None
) -> Dataset:
    """Read a CoNLL-formatted file into a Hugging Face Dataset.
    
    Args:
        file_path: Path to the CoNLL file.
        delimiter: Column delimiter in the file.
        encoding: File encoding.
        sentence_delimiter: String that marks the start of a new document.
        max_length: Maximum number of examples to read (for testing).
        
    Returns:
        A Dataset with 'tokens' and 'pos_tags' columns.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                line = line.strip()
                
                # Skip empty lines and document markers
                if not line or line.startswith(sentence_delimiter):
                    if current_tokens:  # End of sentence
                        tokens.append(current_tokens)
                        labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                    continue
                
                # Parse token and label
                parts = line.split(delimiter)
                if len(parts) < 2:
                    logger.warning(f"Skipping malformed line {i+1}: {line}")
                    continue
                
                token = parts[0].strip()
                label = parts[-1].strip()  # Assume label is the last column
                
                current_tokens.append(token)
                current_labels.append(label)
                
                # Early stopping for testing
                if max_length and len(tokens) >= max_length:
                    break
            
            # Add the last sentence if file doesn't end with a blank line
            if current_tokens:
                tokens.append(current_tokens)
                labels.append(current_labels)
        
        # Create dataset
        dataset_dict = {
            'tokens': tokens,
            'pos_tags': labels
        }
        
        return Dataset.from_dict(dataset_dict)
    
    except Exception as e:
        raise ValueError(
            f"Error reading CoNLL file {file_path} at line {i+1}: {str(e)}"
        ) from e


def train_tagger(
    train_file: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    eval_file: Optional[Union[str, os.PathLike]] = None,
    test_size: float = 0.1,
    config: Optional[Dict[str, Any]] = None,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    **training_kwargs
) -> Tuple[PosTagger, Dict[str, float]]:
    """Train a new POS tagger model.
    
    Args:
        train_file: Path to training data in CoNLL format.
        output_dir: Directory to save the trained model.
        eval_file: Optional path to evaluation data in CoNLL format.
        test_size: Fraction of training data to use for evaluation if eval_file is not provided.
        config: Training configuration (overrides defaults).
        push_to_hub: Whether to push the trained model to the Hugging Face Hub.
        hub_model_id: Model ID for the Hub (required if push_to_hub is True).
        **training_kwargs: Additional training arguments.
        
    Returns:
        A tuple of (trained_model, metrics).
        
    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If training fails.
        
    Example:
        >>> from naganlp.transformer_tagger import train_tagger
        >>> # Train with default settings
        >>> model, metrics = train_tagger(
        ...     train_file="data/train.conll",
        ...     output_dir="models/pos-tagger"
        ... )
        >>> # Train with custom config
        >>> config = {
        ...     'model_name': 'bert-base-multilingual-cased',
        ...     'batch_size': 32,
        ...     'num_train_epochs': 5
        ... }
        >>> model, metrics = train_tagger(
        ...     train_file="data/train.conll",
        ...     output_dir="models/pos-tagger-custom",
        ...     config=config
        ... )
    """
    # Merge config with defaults
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Set random seed for reproducibility
    seed = training_kwargs.pop('seed', 42)
    set_seed(seed)
    
    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load training data
        logger.info(f"Loading training data from {train_file}")
        train_dataset = read_conll(train_file)
        
        # Load or create evaluation dataset
        if eval_file:
            logger.info(f"Loading evaluation data from {eval_file}")
            eval_dataset = read_conll(eval_file)
        else:
            # Split training data
            train_eval = train_dataset.train_test_split(
                test_size=test_size,
                seed=seed
            )
            train_dataset = train_eval['train']
            eval_dataset = train_eval['test']
            logger.info(f"Split training data into {len(train_dataset)} train "
                       f"and {len(eval_dataset)} validation examples")
        
        # Get label list
        all_labels = set()
        for labels in train_dataset['pos_tags'] + eval_dataset['pos_tags']:
            all_labels.update(labels)
        
        label_list = sorted(all_labels)
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for label, i in label2id.items()}
        
        # Save label mappings
        label_map = {
            'label2id': label2id,
            'id2label': id2label,
            'labels': label_list
        }
        with open(output_dir / 'label_map.json', 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        
        # Tokenize the data
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            use_fast=True
        )
        
        def tokenize_and_align_labels(examples):
            """Tokenize and align labels for NER/pos tagging."""
            tokenized_inputs = tokenizer(
                examples['tokens'],
                truncation=True,
                is_split_into_words=True,
                max_length=config['max_seq_length'],
                padding='max_length',
                return_offsets_mapping=True
            )
            
            # Map tokens to their word IDs
            word_ids = []
            for i in range(len(examples['tokens'])):
                word_ids.append(tokenized_inputs.word_ids(batch_index=i))
            
            # Align labels
            labels = []
            for i, label_seq in enumerate(examples['pos_tags']):
                word_idx_map = {}
                current_word_idx = -1
                
                # Create mapping from word index to label
                for word_idx, label in zip(word_ids[i], label_seq):
                    if word_idx is None:
                        continue
                    if word_idx != current_word_idx:
                        current_word_idx = word_idx
                        word_idx_map[word_idx] = label
                
                # Create label sequence
                label_ids = []
                for word_idx in word_ids[i]:
                    if word_idx is None:
                        label_ids.append(-100)  # Special token
                    else:
                        label_ids.append(label2id.get(word_idx_map.get(word_idx, 'O'), -100))
                
                labels.append(label_ids)
            
            tokenized_inputs['labels'] = labels
            return tokenized_inputs
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        tokenized_eval = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_vals=eval_dataset.column_names
        )
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(
            config['model_name'],
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="epoch",
            learning_rate=config['learning_rate'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            num_train_epochs=config['num_train_epochs'],
            weight_decay=config['weight_decay'],
            warmup_ratio=config['warmup_ratio'],
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=seed,
            **training_kwargs
        )
        
        # Compute metrics
        def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
            """Compute accuracy, precision, recall, and F1 score."""
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            
            # Remove ignored index (special tokens)
            true_predictions = [
                [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            
            # Flatten the lists
            flat_predictions = [p for sublist in true_predictions for p in sublist]
            flat_labels = [l for sublist in true_labels for l in sublist]
            
            # Calculate metrics
            accuracy = accuracy_score(flat_labels, flat_predictions)
            f1 = f1_score(
                flat_labels,
                flat_predictions,
                average='weighted',
                zero_division=0
            )
            
            # Get classification report
            report = classification_report(
                flat_labels,
                flat_predictions,
                output_dict=True,
                zero_division=0
            )
            
            # Prepare metrics
            metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall']
            }
            
            # Add per-class metrics
            for label in label_list:
                if label in report:
                    metrics[f"{label}_f1"] = report[label]['f1-score']
            
            return metrics
        
        # Initialize trainer
        data_collator = DataCollatorForTokenClassification(tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        with open(output_dir / 'eval_results.json', 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2)
        
        # Push to Hub if requested
        if push_to_hub:
            if not hub_model_id:
                raise ValueError("hub_model_id must be provided when push_to_hub is True")
            
            logger.info(f"Pushing model to the Hub: {hub_model_id}")
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
        
        # Create and return the tagger
        tagger = PosTagger(str(output_dir))
        
        return tagger, eval_results
    
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def main():
    """Command-line interface for training and using the POS tagger."""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Train or use a transformer-based POS tagger for Nagamese.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new POS tagger')
    train_parser.add_argument(
        'train_file',
        help='Path to training data in CoNLL format'
    )
    train_parser.add_argument(
        'output_dir',
        help='Directory to save the trained model'
    )
    train_parser.add_argument(
        '--eval-file',
        help='Path to evaluation data in CoNLL format (optional)'
    )
    train_parser.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Fraction of training data to use for evaluation if eval_file is not provided'
    )
    train_parser.add_argument(
        '--model-name',
        default='bert-base-multilingual-cased',
        help='Pre-trained model name or path'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training and evaluation'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push the trained model to the Hugging Face Hub'
    )
    train_parser.add_argument(
        '--hub-model-id',
        help='Model ID for the Hub (required if --push-to-hub is set)'
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Tag command
    tag_parser = subparsers.add_parser('tag', help='Tag text with POS labels')
    tag_parser.add_argument(
        '--model',
        default=PRETRAINED_MODELS['default'],
        help='Path to a trained model or Hugging Face model ID'
    )
    tag_parser.add_argument(
        '--text',
        help='Text to tag (or read from stdin if not provided)'
    )
    tag_parser.add_argument(
        '--input-file',
        help='Path to a text file to tag (one sentence per line)'
    )
    tag_parser.add_argument(
        '--output-file',
        help='Path to save the tagged output (default: print to stdout)'
    )
    tag_parser.add_argument(
        '--device',
        default='auto',
        help="Device to run inference on ('cpu', 'cuda', or 'auto')"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'train':
        # Prepare config
        config = {
            'model_name': args.model_name,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_train_epochs': args.num_epochs,
        }
        
        # Check if pushing to Hub
        if args.push_to_hub and not args.hub_model_id:
            parser.error("--hub-model-id is required when --push-to-hub is set")
        
        # Train the model
        try:
            tagger, metrics = train_tagger(
                train_file=args.train_file,
                output_dir=args.output_dir,
                eval_file=args.eval_file,
                test_size=args.test_size,
                config=config,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
                seed=args.seed
            )
            
            # Print metrics
            print("\nTraining complete!")
            print(f"Model saved to: {args.output_dir}")
            print("\nEvaluation results:")
            for key, value in metrics.items():
                if not key.endswith('_f1'):
                    print(f"  {key}: {value:.4f}")
            
            # Print per-class F1 scores
            print("\nPer-class F1 scores:")
            for key, value in metrics.items():
                if key.endswith('_f1'):
                    print(f"  {key}: {value:.4f}")
            
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == 'tag':
        # Check input source
        if args.input_file and args.text:
            parser.error("Cannot specify both --text and --input-file")
        
        # Load the tagger
        try:
            tagger = PosTagger(
                model_name_or_path=args.model,
                device=args.device
            )
            
            # Prepare input text
            if args.input_file:
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
                results = tagger.tag(texts)
            elif args.text:
                results = tagger.tag([args.text])
            else:
                # Read from stdin
                print("Enter text to tag (Ctrl+D to finish):")
                texts = [line.strip() for line in sys.stdin if line.strip()]
                if not texts:
                    print("No input provided", file=sys.stderr)
                    sys.exit(1)
                results = tagger.tag(texts)
            
            # Format and output results
            output = []
            for i, result in enumerate(results):
                if args.input_file or not args.text:
                    output.append(f"\nText {i+1}:")
                
                for token in result:
                    output.append(f"{token['word']}\t{token['entity']}")
            
            # Write output
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output) + '\n')
                print(f"Output written to {args.output_file}")
            else:
                print('\n'.join(output))
        
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

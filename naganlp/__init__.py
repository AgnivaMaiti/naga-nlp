# file: naganlp/__init__.py

"""
NagaNLP: A Natural Language Processing Toolkit for Nagamese.

This package provides tools for various NLP tasks in the Nagamese language,
including part-of-speech tagging, machine translation, and more.

Author: Agniva Maiti
Email: maitiagniva@gmail.com
Repository: https://github.com/AgnivaMaiti/naga-nlp
Documentation: https://naganlp.readthedocs.io/
"""

__version__ = "0.1.0"
__all__ = [
    'PosTagger',
    'Translator',
    'NltkPosTagger',
    'SubwordTokenizer',
    'train_sentencepiece_model',
    'train_and_save_nltk_tagger'
]

import os
import warnings
from typing import Dict, List, Optional, Union
from importlib.resources import files

# Import main components
try:
    from .nmt_translator import Translator
    from .nltk_tagger import NltkPosTagger, train_and_save_nltk_tagger
    from .subword_tokenizer import SubwordTokenizer, train_sentencepiece_model
    from .ner_tagger import NERTagger
    from .word_aligner import WordAligner
    from .transformer_tagger import TransformerTagger
    
    # For backward compatibility
    from transformers import pipeline
except ImportError as e:
    warnings.warn(f"Failed to import some dependencies: {e}. Some functionality may be limited.")

# Package data paths
PACKAGE_DIR = files('naganlp')
DATA_DIR = PACKAGE_DIR / 'data'

class PosTagger:
    """A unified interface for POS tagging models."""
    
    def __init__(self, model_id: str = "agnivamaiti/naganlp-pos-tagger", use_nltk: bool = False):
        """Initialize the POS tagger.
        
        Args:
            model_id: Hugging Face model ID or path to local model
            use_nltk: If True, use NLTK-based tagger instead of transformer
        """
        if use_nltk:
            self.tagger = NltkPosTagger(model_id)
        else:
            self.tagger = pipeline("token-classification", model=model_id)
    
    def tag(self, text: str) -> list:
        """Tag a sentence with part-of-speech tags.
        
        Args:
            text: Input text to tag
            
        Returns:
            List of dictionaries containing entity information
        """
        if hasattr(self.tagger, 'predict'):  # NLTK tagger
            words = text.split()
            tags = self.tagger.predict(words)
            return [{"entity": tag, "word": word} for word, tag in tags]
        else:  # Transformers pipeline
            return self.tagger(text)

"""
NagaNLP - A Natural Language Processing toolkit for the Nagamese language.

This package provides the following main components:

1. POS Tagging:
   - PosTagger: High-accuracy transformer-based tagger (recommended)
     Usage: `from naganlp import PosTagger`
   - NltkPosTagger: Lightweight NLTK-based tagger (faster, less accurate)
     Usage: `from naganlp import NltkPosTagger`

2. Machine Translation:
   - Translator: Neural Machine Translation model for Nagamese-English
     Usage: `from naganlp import Translator`

3. Tokenization:
   - SubwordTokenizer: Subword tokenizer for Nagamese text
     Usage: `from naganlp import SubwordTokenizer`
"""

# Expose the main, user-facing classes for easy importing
from .transformer_tagger import PosTagger  # noqa: F401
from .nltk_tagger import NltkPosTagger  # noqa: F401
from .nmt_translator import Translator  # noqa: F401
from .subword_tokenizer import SubwordTokenizer  # noqa: F401

__all__ = [
    'PosTagger',
    'NltkPosTagger',
    'Translator',
    'SubwordTokenizer',
    'train_and_save_nltk_tagger',
    'train_sentencepiece_model',
    'PACKAGE_DIR',
    'DATA_DIR'
]
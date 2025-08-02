# file: naganlp/__init__.py

"""
naganlp: A Natural Language Processing Toolkit for Nagamese.

Author: Agniva Maiti
Email: agnivamaiti.official@gmail.com
Repository: https://github.com/AgnivaMaiti/naga-nlp
"""

__version__ = "0.1.0"

"""NagaNLP - A Natural Language Processing toolkit for the Nagamese language."""

# Expose the main, user-facing classes for easy importing
# This allows 'from naganlp import PosTagger'
from .transformer_tagger import PosTagger  # noqa: F401
from .nltk_tagger import NltkPosTagger  # noqa: F401
from .nmt_translator import Translator  # noqa: F401
from .subword_tokenizer import SubwordTokenizer  # noqa: F401

__all__ = [
    'PosTagger',
    'NltkPosTagger',
    'Translator',
    'SubwordTokenizer',
]
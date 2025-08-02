# file: naganlp/__init__.py

"""
naganlp: A Natural Language Processing Toolkit for Nagamese.

Author: Agniva Maiti
Email: agnivamaiti.official@gmail.com
Repository: https://github.com/AgnivaMaiti/naga-nlp
"""

__version__ = "0.1.0"

# Expose the main, user-facing classes for easy importing
# This allows 'from naganlp import PosTagger'
from .transformer_tagger import PosTagger
from .nltk_tagger import NltkPosTagger
from .nmt_translator import Translator
from .subword_tokenizer import SubwordTokenizer
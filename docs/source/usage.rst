.. _usage:

Usage
=====

This section provides examples of how to use NagaNLP for various NLP tasks.

Part-of-Speech Tagging
----------------------

.. code-block:: python

    from naganlp import PosTagger
    
    # Initialize the POS tagger
    tagger = PosTagger()
    
    # Tag a sentence
    result = tagger.tag("moi school te jai")
    print(result)

Machine Translation
------------------

.. code-block:: python

    from naganlp import Translator
    
    # Initialize the translator
    translator = Translator()
    
    # Translate a sentence
    translation = translator.translate("moi school te jai")
    print(translation)

Subword Tokenization
-------------------

.. code-block:: python

    from naganlp import SubwordTokenizer
    
    # Initialize the tokenizer
    tokenizer = SubwordTokenizer()
    
    # Tokenize a sentence
    tokens = tokenizer.tokenize("moi school te jai")
    print(tokens)
    
    # Convert tokens back to text
    text = tokenizer.detokenize(tokens)
    print(text)

NLTK Tagger
-----------

.. code-block:: python

    from naganlp import NltkPosTagger
    
    # Initialize the NLTK tagger
    tagger = NltkPosTagger()
    
    # Tag a sentence
    result = tagger.tag("moi school te jai")
    print(result)

Configuration
-------------

You can configure various aspects of NagaNLP by passing parameters to the constructors:

.. code-block:: python

    # Custom model path for POS tagging
    tagger = PosTagger(model_name_or_path="path/to/your/model")
    
    # Custom device (CPU/GPU) for translation
    translator = Translator(device="cuda")  # or "cpu"
    
    # Custom vocabulary size for subword tokenization
    tokenizer = SubwordTokenizer(vocab_size=8000)

For more advanced usage, refer to the :ref:`api` documentation.

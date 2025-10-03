"""
NagaNLP Demo Script
------------------
This script demonstrates the core functionality of the NagaNLP package.
"""

# Import the main components
from naganlp import PosTagger, NltkPosTagger, Translator, SubwordTokenizer

# Initialize the POS tagger (using the default transformer-based model)
print("Initializing POS Tagger...")
pos_tagger = PosTagger()

# Example text in Nagamese
text = "moi school jai"
print(f"\nTagging text: {text}")
tags = pos_tagger.tag(text)
for tag in tags:
    print(f"Word: {tag['word']} - Tag: {tag['entity']}")

# Initialize the NLTK-based POS tagger (lighter but less accurate)
print("\nInitializing NLTK-based POS Tagger...")
nltk_tagger = NltkPosTagger()
tags = nltk_tagger.predict(text.split())
print(f"NLTK Tags: {tags}")

# Initialize the Translator
print("\nInitializing Translator...")
translator = Translator()

# Translate from Nagamese to English
translation = translator.translate(text, source_lang="nagamese", target_lang="english")
print(f"Translation: {translation}")

# Initialize the Subword Tokenizer
print("\nInitializing Subword Tokenizer...")
tokenizer = SubwordTokenizer()
tokens = tokenizer.tokenize(text)
print(f"Subword tokens: {tokens}")

print("\nDemo completed successfully!")

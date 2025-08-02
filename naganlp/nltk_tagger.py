# file: nltk_tagger.py

import nltk
import random
import pickle
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
import os

def read_conll_for_nltk(path: str):
    """Reads a CoNLL file into a list of tagged sentences for NLTK."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The CoNLL file was not found at: {path}")

    tagged_sents = []
    with open(path, encoding='utf-8') as f:
        sent = []
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    tagged_sents.append(sent)
                    sent = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token, tag = parts[0], parts[-1]
                    sent.append((token, tag))
        if sent:
            tagged_sents.append(sent)
    return tagged_sents

def train_and_save_nltk_tagger(conll_path: str, model_path: str):
    """Trains and saves an NLTK backoff tagger."""
    tagged_sentences = read_conll_for_nltk(conll_path)
    random.seed(42)
    random.shuffle(tagged_sentences)

    # Simple 90/10 split for training and testing
    split_idx = int(len(tagged_sentences) * 0.9)
    train_sents = tagged_sentences[:split_idx]
    test_sents = tagged_sentences[split_idx:]

    # Build the backoff tagger chain
    default_tagger = DefaultTagger('NOUN') # Default to NOUN if unknown
    unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
    bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
    trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger)

    print("--- Evaluating NLTK Tagger ---")
    accuracy = trigram_tagger.accuracy(test_sents)
    print(f"Trigram Backoff Tagger Accuracy: {accuracy:.2%}")

    # Save the trained tagger using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(trigram_tagger, f)
    print(f"NLTK model saved to '{model_path}'")
    return trigram_tagger


class NltkPosTagger:
    """A POS Tagger for Nagamese using a pickled NLTK tagger object."""
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tagger model file not found at: {model_path}")
        with open(model_path, 'rb') as f:
            self.tagger = pickle.load(f)

    def predict(self, tokens: list[str]):
        """
        Tags a list of tokens.

        Args:
            tokens (list[str]): A list of pre-tokenized words.

        Returns:
            list[tuple[str, str]]: A list of (word, tag) tuples.
        """
        return self.tagger.tag(tokens)


if __name__ == '__main__':
    conll_file = 'nagamese_manual_enriched.conll'
    nltk_model_file = 'nagamese_nltk_tagger.pkl'

    # --- Step 1: Train and save the NLTK tagger ---
    train_and_save_nltk_tagger(conll_file, nltk_model_file)

    # --- Step 2: Load the tagger and perform inference ---
    print("\n--- Loading NLTK Tagger for Inference ---")
    try:
        nltk_tagger = NltkPosTagger(nltk_model_file)
        test_tokens = ['moi', 'ghor', 'te', 'jai', 'ase']
        tagged_sentence = nltk_tagger.predict(test_tokens)

        print(f"\nTest Tokens: {test_tokens}")
        print(f"Predicted POS Tags: {tagged_sentence}")

    except FileNotFoundError as e:
        print(f"\nError loading the NLTK model: {e}")
import os
import pandas as pd
from naganlp.nltk_tagger import train_and_save_nltk_tagger, NltkPosTagger
from naganlp.subword_tokenizer import train_sentencepiece_model, load_data_for_spm

def train_pos_tagger():
    """Train and save the NLTK POS Tagger"""
    print("\n=== Training NLTK POS Tagger ===")
    conll_file = 'nagamese_manual_enriched.conll'
    model_path = 'nagamese_nltk_tagger.pkl'
    
    if not os.path.exists(conll_file):
        print(f"Error: CoNLL file not found at {conll_file}")
        return
    
    try:
        train_and_save_nltk_tagger(conll_file, model_path)
        print("NLTK POS Tagger training completed successfully!")
    except Exception as e:
        print(f"Error training NLTK POS Tagger: {e}")

def train_tokenizer():
    """Train and save the SentencePiece tokenizer"""
    print("\n=== Training SentencePiece Tokenizer ===")
    # Create a small parallel corpus for training
    data = {
        'nagamese': [
            "moi school te jai ase",
            "tumar naam ki",
            "bhat khaise neki",
            "aaj din bhal thakise",
            "eitu kiman hoi"
        ],
        'english': [
            "I am going to school",
            "what is your name",
            "have you eaten rice",
            "how was your day today",
            "how much is this"
        ]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Clean the text (simple cleaning for this example)
    df['nagamese_cleaned'] = df['nagamese'].str.lower()
    df['english_cleaned'] = df['english'].str.lower()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the tokenizer with smaller vocabulary size
    try:
        train_sentencepiece_model(df, 'models/nagamese_english_spm', vocab_size=200)
        print("SentencePiece Tokenizer training completed successfully!")
        
        # Copy the model file to the root directory for backward compatibility
        import shutil
        shutil.copy('models/nagamese_english_spm.model', 'nagamese_english_spm.model')
        print("Model file copied to root directory.")
        
    except Exception as e:
        print(f"Error training SentencePiece Tokenizer: {e}")
        print("This might be due to insufficient training data. The NLTK POS Tagger should still work.")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Train models
    train_pos_tagger()
    train_tokenizer()
    
    print("\nTraining complete! You can now run test.py to test the models.")

import sys
from naganlp.nmt_translator import Translator

def main():
    # Model ID from Hugging Face Hub
    model_id = "agnivamaiti/naganlp-nmt-en"
    
    # Test sentences
    test_sentences = [
        "moi school te jai",
        "apuni laga naam ki?",
        "bhat khaise neki?",
        "aaj din bhal thakise"
    ]

    try:
        print(f"Loading model '{model_id}' from Hugging Face Hub...")
        translator = Translator(model_id=model_id)
        print("Model loaded successfully!")
        
        print("\nTesting translations:")
        print("-" * 50)
        
        for sent in test_sentences:
            translation = translator.translate(sent)
            print(f"Nagamese: {sent}")
            print(f"English:  {translation}")
            print("-" * 50)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Make sure you have the latest version of huggingface-hub")
        print("   (install with: pip install --upgrade huggingface-hub)")
        print("3. If you're behind a proxy, set the HTTP_PROXY/HTTPS_PROXY environment variables")
        print("4. Try logging in to Hugging Face: huggingface-cli login")
        sys.exit(1)

if __name__ == "__main__":
    main()

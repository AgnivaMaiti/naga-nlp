import sys
from naganlp import Translator, PosTagger

def test_translator():
    print("Testing Translator import from package...")
    try:
        # Initialize translator with the correct model ID
        translator = Translator("agnivamaiti/naganlp-nmt-en")
        print("[OK] Translator imported successfully!")
        
        # Test translation
        test_sentence = "moi school te jai"
        translation = translator.translate(test_sentence)
        print("\nTest Translation:")
        print(f"Nagamese: {test_sentence}")
        print(f"English:  {translation}")
        return True
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

def test_pos_tagger():
    print("\nTesting PosTagger import from package...")
    try:
        # Initialize POS tagger
        tagger = PosTagger()
        print("[OK] PosTagger imported successfully!")
        
        # Test POS tagging
        test_sentence = "moi school te jai"
        tags = tagger.tag(test_sentence)
        print(f"\nTest POS Tagging for: {test_sentence}")
        if isinstance(tags, list) and len(tags) > 0:
            # Handle both dictionary and object formats
            if isinstance(tags[0], dict):
                for tag in tags:
                    if 'entity' in tag and 'word' in tag:
                        print(f"Word: {tag['word']}, Tag: {tag['entity']}")
                    elif 'label' in tag and 'token' in tag:
                        print(f"Word: {tag['token']}, Tag: {tag['label']}")
                    else:
                        print(f"Unexpected tag format: {tag}")
            else:
                print(f"Raw tags output: {tags}")
        else:
            print(f"No tags returned. Output: {tags}")
            return False
        return True
            
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

if __name__ == "__main__":
    # Set UTF-8 encoding for Windows
    if sys.platform == "win32":
        import io, sys
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Run tests
    success = []
    success.append(test_translator())
    success.append(test_pos_tagger())
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Translator: {'PASS' if success[0] else 'FAIL'}")
    print(f"PosTagger:  {'PASS' if success[1] else 'FAIL'}")
    
    if all(success):
        print("\nAll tests passed! Your NagaNLP package is working correctly.")
        print("You can now use it in your code with:")
        print("from naganlp import Translator, PosTagger")
    else:
        print("\nSome tests failed. Please check the error messages above.")

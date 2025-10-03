# file: main.py

import argparse
from getpass import getpass
from huggingface_hub import HfApi, create_repo

# Import all developer-facing training functions from the package
from naganlp.transformer_tagger import train_and_upload_tagger
from naganlp.nmt_translator import train_and_upload_translator
from naganlp.nltk_tagger import train_and_save_nltk_tagger
from naganlp.subword_tokenizer import train_sentencepiece_model, load_data_for_spm
from naganlp.word_aligner import align_corpus, load_data_for_aligner

def main():
    """Main function for the naga-nlp developer command-line interface."""
    parser = argparse.ArgumentParser(description="naga-nlp: Developer Toolkit CLI")
    # Use subparsers for a more robust CLI that can handle different commands with different arguments
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available developer commands")

    # --- Command to setup repos on the Hub ---
    parser_setup = subparsers.add_parser('setup-hub', help='Create model repositories on the Hugging Face Hub.')

    # --- Command to train the Transformer POS tagger ---
    parser_train_tagger = subparsers.add_parser('train-tagger', help='Train and upload the Transformer POS tagger.')
    parser_train_tagger.add_argument('--hub-id', type=str, required=True, help='HF Hub ID (e.g., your-name/naganlp-pos-tagger)')
    parser_train_tagger.add_argument('--conll-file', type=str, default='nagamese_manual_enriched.conll', help='Path to the training data.')

    # --- Command to train the NMT model ---
    parser_train_nmt = subparsers.add_parser('train-translator', help='Train and upload the NMT model.')
    parser_train_nmt.add_argument('--hub-id', type=str, required=True, help='HF Hub ID (e.g., your-name/naganlp-nmt-en)')
    parser_train_nmt.add_argument('--csv-file', type=str, default='merged.csv', help='Path to the main parallel corpus.')
    parser_train_nmt.add_argument('--gloss-file', type=str, default='nagamese_gloss.csv', help='Optional glossary CSV for pre-training.')

    # --- Command to train the NLTK tagger (local save only) ---
    parser_train_nltk = subparsers.add_parser('train-nltk-tagger', help='Train and save the NLTK POS Tagger locally.')
    parser_train_nltk.add_argument('--conll-file', type=str, default='nagamese_manual_enriched.conll')
    parser_train_nltk.add_argument('--model-path', type=str, default='nagamese_nltk_tagger.pkl')

    args = parser.parse_args()

    # --- Execute the chosen command ---
    if args.command == 'setup-hub':
        print("This utility will create the necessary repositories on the Hugging Face Hub.")
        username = input("Enter your Hugging Face username: ")
        hf_token = getpass("Enter your Hugging Face token (with write permissions): ")
        
        pos_tagger_id = f"{username}/naganlp-pos-tagger"
        nmt_model_id = f"{username}/naganlp-nmt-en"
        
        print(f"\nCreating repo: {pos_tagger_id}")
        create_repo(pos_tagger_id, token=hf_token, exist_ok=True)
        
        print(f"Creating repo: {nmt_model_id}")
        create_repo(nmt_model_id, token=hf_token, exist_ok=True)
        
        print("\nSetup complete. You can now train and upload your models.")
        print("Remember to log in with 'huggingface-cli login' before training.")

    elif args.command == 'train-tagger':
        train_and_upload_tagger(args.conll_file, args.hub_id)

    elif args.command == 'train-translator':
        train_and_upload_translator(
            csv_path=args.csv_file,
            hub_model_id=args.hub_id,
            gloss_path=args.gloss_file
        )
        
    elif args.command == 'train-nltk-tagger':
        train_and_save_nltk_tagger(args.conll_file, args.model_path)

if __name__ == '__main__':
    main()
# file: main.py

import argparse
from getpass import getpass
from huggingface_hub import HfApi, create_repo

# Import your training functions
from naganlp.transformer_tagger import train_and_upload_tagger
# from naganlp.nmt_translator import train_and_upload_translator # etc.

def main():
    parser = argparse.ArgumentParser(description="naga-nlp: Developer Toolkit CLI")
    parser.add_argument("command", choices=['train-tagger', 'setup-hub'], help="Action to perform.")
    parser.add_argument('--hub-id', type=str, help="Your Hugging Face Hub model ID (e.g., your-username/naganlp-pos-tagger)")
    parser.add_argument('--conll-file', type=str, default='nagamese_manual_enriched.conll', help='Path to training data.')

    args = parser.parse_args()

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

    elif args.command == 'train-tagger':
        if not args.hub_id:
            print("Error: Please provide a Hub ID with --hub-id (e.g., your-username/naganlp-pos-tagger)")
            return
        # You would need to be logged in: huggingface-cli login
        train_and_upload_tagger(args.conll_file, args.hub_id)

if __name__ == '__main__':
    main()
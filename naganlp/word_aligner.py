# file: word_aligner.py

import pandas as pd
import os
import re
import subprocess

def load_data_for_aligner(filepath: str):
    """Loads and cleans data for the word aligner."""
    if not os.path.exists(filepath):
        print(f"Error: The file at {filepath} was not found.")
        return None
    df = pd.read_csv(filepath).dropna(subset=['english', 'nagamese'])
    def clean_text(text):
        if not isinstance(text, str): return ""
        return re.sub(r'\s+', ' ', text).strip()

    df['english_cleaned'] = df['english'].apply(clean_text)
    df['nagamese_cleaned'] = df['nagamese'].apply(clean_text)
    return df

def align_corpus(df, output_file='alignments.txt'):
    """
    Runs awesome-align on the parallel corpus to generate word alignments.

    Args:
        df (pd.DataFrame): DataFrame with 'nagamese_cleaned' and 'english_cleaned' columns.
        output_file (str): The file to save the alignments to.
    """
    # 1. Prepare the input file for awesome-align
    # --- CORRECTION 1: The format must be tab-separated ('\t') ---
    input_file = 'aligner_input.txt'
    with open(input_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['english_cleaned']}\t{row['nagamese_cleaned']}\n")

    print(f"Input file for aligner created at '{input_file}'")

    # 2. Run the awesome-align command
    # --- CORRECTION 2: The path to the script is at the root of the cloned repo ---
    align_script_path = 'awesome-align/run_align.py'
    if not os.path.exists(align_script_path):
        print(f"Error: Alignment script not found at '{align_script_path}'.")
        print("Please ensure you have run the setup cell to clone and install awesome-align correctly.")
        return

    model_name = 'bert-base-multilingual-cased'
    command = [
        'python3', align_script_path,
        '--model_name_or_path', model_name,
        '--data_file', input_file,
        '--output_file', output_file,
        '--extraction', 'softmax',
        '--batch_size', '32'
    ]

    print("\n--- Starting Word Alignment (this may take several minutes) ---")
    try:
        # Using subprocess to run the command
        process = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        print(process.stdout) # Print the output from the script
        print(f"--- Alignment Complete. Results saved to '{output_file}' ---")
    except subprocess.CalledProcessError as e:
        print("--- An error occurred during alignment. ---")
        print(f"Return Code: {e.returncode}")
        print("----- STDOUT -----")
        print(e.stdout)
        print("----- STDERR -----")
        print(e.stderr)

if __name__ == '__main__':
    # --- 1. Load Data ---
    dataframe = load_data_for_aligner('merged.csv')

    if dataframe is not None:
        # --- 2. Generate Alignments ---
        # We align a smaller subset for a quick demonstration.
        # To run on the full dataset, use: align_corpus(dataframe)
        align_corpus(dataframe.head(100), output_file='alignments_sample.txt')

        # --- 3. Display Sample Alignments ---
        print("\n--- Sample of Generated Alignments ---")
        try:
            with open('alignments_sample.txt', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5: break
                    print(line.strip())
        except FileNotFoundError:
            print("Alignment file not found. The alignment process may have failed.")
# file: naganlp/transformer_tagger.py

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import classification_report, accuracy_score
import os

def read_conll(path: str) -> Dataset:
    """Reads a CoNLL-formatted file and returns a Hugging Face Dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The CoNLL file was not found at: {path}")
    sentences, tags = [], []
    with open(path, encoding='utf-8') as f:
        sent, sent_tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    tags.append(sent_tags)
                    sent, sent_tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    sent.append(parts[0])
                    sent_tags.append(parts[-1])
        if sent:
            sentences.append(sent)
            tags.append(sent_tags)
    return Dataset.from_dict({'tokens': sentences, 'pos_tags': tags})

class PosTagger:
    """
    A high-accuracy Part-of-Speech tagger for Nagamese.

    This class uses a fine-tuned Transformer model. On first use, it will
    download the model from the Hugging Face Hub and cache it locally.

    Example:
        >>> tagger = PosTagger()
        >>> result = tagger.tag("মই স্কুললৈ যাওঁ")
        >>> print(result)
        [{'entity_group': 'PRON', 'word': 'মই', ...}]
    """

    def __init__(self, model_name_or_path: str = "your-username/naganlp-pos-tagger"):
        """
        Initialize the POS Tagger with a pre-trained model.

        Args:
            model_name_or_path (str, optional): The model identifier from the Hugging Face Hub or a
                                              path to a local directory. Defaults to "your-username/naganlp-pos-tagger".

        Raises:
            ValueError: If the model_name_or_path is invalid or empty.
            OSError: If there's an issue downloading or loading the model.
            RuntimeError: If the model fails to initialize.
        """
        if not model_name_or_path or not isinstance(model_name_or_path, str):
            raise ValueError("model_name_or_path must be a non-empty string")

        self.model_name_or_path = model_name_or_path
        self.tagger = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the token classification pipeline."""
        try:
            from transformers import pipeline
            import torch
            
            device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
            
            self.tagger = pipeline(
                task="token-classification",
                model=self.model_name_or_path,
                aggregation_strategy="simple",
                device=device,
                framework="pt"
            )
            
            # Verify the model is properly loaded
            if not hasattr(self.tagger, 'model') or not self.tagger.model:
                raise RuntimeError("Failed to initialize the model properly.")
                
        except ImportError as e:
            raise ImportError(
                "Required dependencies not found. Please install them using: "
                "pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_name_or_path}'. "
                "Please ensure it's a valid Hugging Face model ID or local path."
            ) from e

    def tag(self, text: str) -> list[dict]:
        """
        Tag a Nagamese sentence with part-of-speech labels.

        Args:
            text (str): The input text to tag. Should be a single sentence or phrase.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains:
                - word (str): The token/word
                - entity_group (str): The predicted POS tag
                - score (float): The confidence score (0-1)
                - start (int): Start position in the input text
                - end (int): End position in the input text

        Raises:
            ValueError: If the input is not a non-empty string.
            RuntimeError: If the tagging fails for any reason.

        Example:
            >>> tagger = PosTagger()
            >>> result = tagger.tag("মই স্কুললৈ যাওঁ")
            >>> print(result)
            [
                {'word': 'মই', 'entity_group': 'PRON', 'score': 0.99, 'start': 0, 'end': 2},
                {'word': 'স্কুললৈ', 'entity_group': 'NOUN', 'score': 0.98, 'start': 3, 'end': 9},
                {'word': 'যাওঁ', 'entity_group': 'VERB', 'score': 0.97, 'start': 10, 'end': 13}
            ]
        """
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text).__name__}")
        if not text.strip():
            return []

        try:
            # Ensure the model is loaded
            if self.tagger is None:
                self._initialize_model()
                
            results = self.tagger(text)
            
            # Convert to list if a single result is returned
            if not isinstance(results, list):
                results = [results]
                
            return results
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to tag text. Error: {str(e)}"
            ) from e
            
    def __repr__(self) -> str:
        """Return a string representation of the tagger."""
        return f"{self.__class__.__name__}(model_name_or_path='{self.model_name_or_path}')"

def train_and_upload_tagger(conll_path: str, hub_model_id: str):
    """
    (Developer function) Trains a POS tagger and uploads it to the Hugging Face Hub.
    You must be logged in via `huggingface-cli login` to use this.
    """
    print("--- Preparing Dataset ---")
    dataset = read_conll(conll_path)

    # --- 1. Data Preparation Logic (No longer a placeholder) ---
    unique_tags = sorted({tag for tag_list in dataset['pos_tags'] for tag in tag_list})
    label2id = {label: i for i, label in enumerate(unique_tags)}
    id2label = {i: label for i, label in enumerate(unique_tags)}
    
    checkpoint = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_and_align_labels(examples):
        """Aligns tokens with their labels for the model."""
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["pos_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100) # Ignore special tokens and subwords
                else:
                    label_ids.append(label2id[label[word_idx]])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True)
    split = tokenized_ds.train_test_split(test_size=0.1, seed=42)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=len(unique_tags), id2label=id2label, label2id=label2id
    )

    # --- 2. Metrics Calculation Logic (No longer a placeholder) ---
    def compute_metrics(p):
        """Computes F1 score, and accuracy for evaluation."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        # Flatten the lists
        flat_true_predictions = [item for sublist in true_predictions for item in sublist]
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        results = classification_report(flat_true_labels, flat_true_predictions, output_dict=True, zero_division=0)
        return {
            "f1": results["weighted avg"]["f1-score"],
            "accuracy": accuracy_score(flat_true_labels, flat_true_predictions)
        }

    # --- 3. Training Configuration ---
    print("--- Configuring Training ---")
    training_args = TrainingArguments(
        output_dir=hub_model_id.split("/")[-1], # e.g., 'naganlp-pos-tagger'
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        push_to_hub=True, # This is the key argument for uploading
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 4. Start Training and Upload ---
    print(f"--- Starting Training & Uploading to {hub_model_id} ---")
    trainer.train()
    trainer.push_to_hub(commit_message="End of training")
    print("--- Model successfully trained and uploaded! ---")
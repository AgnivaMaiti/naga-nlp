# NagaNLP: Natural Language Processing for Nagamese

[![PyPI](https://img.shields.io/pypi/v/naganlp)](https://pypi.org/project/naganlp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/naganlp)](https://pypi.org/project/naganlp/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Issues](https://img.shields.io/github/issues/AgnivaMaiti/naga-nlp)](https://github.com/AgnivaMaiti/naga-nlp/issues)
[![GitHub Stars](https://img.shields.io/github/stars/AgnivaMaiti/naga-nlp)](https://github.com/AgnivaMaiti/naga-nlp/stargazers)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/agnivamaiti)

NagaNLP is a comprehensive Natural Language Processing toolkit for the Nagamese language, featuring state-of-the-art models for various NLP tasks. It provides simple and efficient tools for working with Nagamese text data.

## Features

- **Part-of-Speech Tagging**: Transformer-based and NLTK-based POS taggers
- **Neural Machine Translation**: Seq2Seq model for Nagamese to English translation
- **Named Entity Recognition**: Pre-trained BERT models for entity recognition
- **Word Alignment**: Tools for parallel corpus alignment
- **Subword Tokenization**: Support for handling out-of-vocabulary words
- **Easy Integration**: Simple Python API for all functionalities
- **Pre-trained Models**: Ready-to-use models for quick deployment
- **Extensible**: Easy to customize and extend for specific use cases

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Install from PyPI (recommended)
```bash
pip install naganlp
```

### Install from source
```bash
git clone https://github.com/AgnivaMaiti/naga-nlp.git
cd naga-nlp
pip install -e .
```

### Install with optional dependencies
```bash
# For development with testing and documentation
pip install "naganlp[dev]"

# For using GPU acceleration (requires CUDA-compatible GPU)
pip install "naganlp[gpu]"
```

## Pre-trained Models

All models are available on Hugging Face:

1. **Machine Translation**: [agnivamaiti/naganlp-nmt-en](https://huggingface.co/agnivamaiti/naganlp-nmt-en)
2. **Named Entity Recognition**: [agnivamaiti/naganlp-ner-crf-tagger](https://huggingface.co/agnivamaiti/naganlp-ner-crf-tagger)
3. **POS Tagger**: [agnivamaiti/naganlp-pos-tagger](https://huggingface.co/agnivamaiti/naganlp-pos-tagger)

## Quick Start

### Part-of-Speech Tagging

#### Using the Transformer-based Tagger (Recommended)
```python
from naganlp import PosTagger

# Initialize the tagger (automatically downloads the model on first use)
tagger = PosTagger()

# Tag a sentence
sentence = "moi school te jai"  # "I go to school" in Nagamese
result = tagger.tag(sentence)

# Print the results
for token in result:
    print(f"{token['word']}: {token['entity_group']}")
```

#### Using the NLTK-based Tagger (Lightweight)
```python
from naganlp import NltkPosTagger

# Load the pre-trained NLTK model
tagger = NltkPosTagger("naga_pos_model.pkl")

# Tag a list of tokens
result = tagger.predict(["moi", "school", "te", "jai"])
print(result)
# Output: [('moi', 'PRON'), ('school', 'NOUN'), ('te', 'ADP'), ('jai', 'VERB')]
```

### Machine Translation

```python
from naganlp import Translator

# Initialize the translator (uses 'agnivamaiti/naganlp-nmt-en' by default)
translator = Translator()

# Translate from Nagamese to English
translation = translator.translate("moi school te jai")
print(f"Translation: {translation}")
# Output: "I go to school"

# Get translation with token IDs
translation, token_ids = translator.translate("tumi kiman din ahiba?", return_tokens=True)
print(f"Translation: {translation}")
print(f"Token IDs: {token_ids}")
```

### Named Entity Recognition

```python
from naganlp import NERTagger

# Initialize the NER tagger with the pre-trained model
ner_tagger = NERTagger(model_id="agnivamaiti/naganlp-ner-crf-tagger")

# Extract named entities from text
entities = ner_tagger.tag("Agniva is going to Guwahati tomorrow")
for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.2f})")

# Training a new NER model (example)
# python main.py train-ner --data-file path/to/ner_data.json --hub-id your-username/naganlp-ner-crf-tagger
```

### Word Alignment

```python
from naganlp import align_parallel_texts

# Align parallel sentences
df = align_parallel_texts(
    source_texts=["moi school jai", "tumi kiman din te ahibo?"],
    target_texts=["I go to school", "In how many days will you come?"],
    config={"alignment_method": "fast_align"}  # or 'simalign' for better quality
)

print(df[['source', 'target', 'alignment']])
```

### Subword Tokenization

```python
from naganlp import SubwordTokenizer

# Initialize tokenizer with a pre-trained model
tokenizer = SubwordTokenizer()

# Tokenize a sentence
tokens = tokenizer.tokenize("moi school te jai")
print(f"Tokens: {tokens}")

# Convert tokens to IDs
ids = tokenizer.encode("moi school te jai")
print(f"Token IDs: {ids}")
```

## API Reference

### PosTagger
```python
class PosTagger(model_name: str = 'agnivamaiti/naganlp-pos-tagger', use_nltk: bool = False)
```
A part-of-speech tagger for Nagamese text.

**Parameters**:
- `model_name`: Hugging Face model ID or path to local model
- `use_nltk`: If True, uses NLTK-based tagger instead of transformer

**Methods**:
- `tag(text: Union[str, List[str]])`: Tag a single sentence or list of sentences
- `__call__`: Alias for `tag`

### NltkPosTagger
```python
class NltkPosTagger(model_path: str)
```
Lightweight POS tagger using NLTK's CRF implementation.

**Parameters**:
- `model_path`: Path to the trained NLTK model file

**Methods**:
- `predict(tokens: List[str])`: Tag a list of tokens
- `evaluate(test_data)`: Evaluate the model on test data

### Translator
```python
class Translator(model_id: str = 'agnivamaiti/naganlp-nmt-en', device: str = None)
```
Neural Machine Translation model for Nagamese to English.

**Parameters**:
- `model_id`: Hugging Face model ID or path to local model
- `device`: Device to run the model on ('cuda' or 'cpu')

**Methods**:
- `translate(text: str, beam_size: int = 5, max_len: int = 50, length_penalty: float = 0.7, return_tokens: bool = False)`: Translate text
- `batch_translate(texts: List[str], **kwargs)`: Translate a batch of texts

### NERTagger
```python
class NERTagger(model_name: str = 'agnivamaiti/naganlp-ner-crf-tagger')
```
Named Entity Recognition for Nagamese text.

**Parameters**:
- `model_name`: Hugging Face model ID or path to local model

**Methods**:
- `tag(text: str)`: Extract named entities from text
- `batch_tag(texts: List[str])`: Process multiple texts in a batch

### SubwordTokenizer
```python
class SubwordTokenizer(model_path: str = None, vocab_size: int = 8000)
```
Subword tokenizer for Nagamese text.

**Parameters**:
- `model_path`: Path to SentencePiece model
- `vocab_size`: Vocabulary size (only used when training a new model)

**Methods**:
- `tokenize(text: str)`: Tokenize text into subwords
- `encode(text: str)`: Convert text to token IDs
- `decode(ids: List[int])`: Convert token IDs back to text
- `train(input_file: str, model_prefix: str)`: Train a new tokenizer

## Pre-trained Models

NagaNLP provides several pre-trained models available on Hugging Face Hub:

1. **Machine Translation (Nagamese to English)**
   - Model ID: `agnivamaiti/naganlp-nmt-en`
   - Type: Seq2Seq Transformer
   - Training Data: Parallel Nagamese-English corpus
   - Usage: `translator = Translator(model_id="agnivamaiti/naganlp-nmt-en")`

2. **Named Entity Recognition**
   - Model ID: `agnivamaiti/naganlp-ner-crf-tagger`
   - Type: CRF-based NER Tagger
   - Training Data: Annotated Nagamese text
   - Usage: `ner_tagger = NERTagger(model_id="agnivamaiti/naganlp-ner-crf-tagger")`

3. **Part-of-Speech Tagger**
   - Model ID: `agnivamaiti/naganlp-pos-tagger`
   - Type: Transformer-based POS Tagger
   - Training Data: An annotated Nagamese text
   - Usage: `pos_tagger = PosTagger(model_id="agnivamaiti/naganlp-pos-tagger")`

## Command Line Interface
{{ ... }}
NagaNLP provides a convenient CLI for common tasks:

```bash
# Tag text with POS tags
naganlp pos-tag --text "moi school te jai"

# Translate text
naganlp translate --text "tumi kiman din ahiba?" --target-lang en

# Train a new model
naganlp train-pos-tagger --train-file train.conll --output-dir ./model

# Evaluate a model
naganlp evaluate --model agnivamaiti/naga-pos-tagger --eval-file test.conll
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to the project.

### Setting up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/AgnivaMaiti/naga-nlp.git
   cd naga-nlp
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Run code formatting and linting:
   ```bash
   black .
   flake8 .
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use NagaNLP in your research, please cite it as follows:

```bibtex
@software{naganlp2023,
  author = {Agniva Maiti},
  title = {NagaNLP: Natural Language Processing for Nagamese},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AgnivaMaiti/naga-nlp}}
}
```

## Contact

For questions, suggestions, or support:
- Open an issue on [GitHub Issues](https://github.com/AgnivaMaiti/naga-nlp/issues)
- Email: [Agniva Maiti](mailto:agnivamaiti.official@gmail.com)
- Twitter: [@AgnivaMaiti](https://twitter.com/AgnivaMaiti)

```python
from naganlp import NltkPosTagger

# First train and save the model (only needed once)
from naganlp.nltk_tagger import train_and_save_nltk_tagger
train_and_save_nltk_tagger("path/to/your/conll/file.conll", "naga_pos_model.pkl")

# Then load and use the trained model
tagger = NltkPosTagger("naga_pos_model.pkl")

# Tag a list of pre-tokenized words
result = tagger.predict(["moi", "school", "te", "jai"])
print(result)
# Output: [('moi', 'PRON'), ('school', 'NOUN'), ('te', 'ADP'), ('jai', 'VERB')]
```

### Translation

```python
from naganlp import Translator

# Initialize the translator with the pre-trained model from Hugging Face
translator = Translator(model_id="agnivamaiti/naganlp-nmt-en")

# Translate from Nagamese to English
translation = translator.translate("moi school te jai")
print(translation)
# Output: "I go to school"
```

## Documentation

### Data Requirements

- For POS Tagging: CONLL-formatted file with token and POS tag columns
- For Translation: Parallel corpus in CSV format with 'nagamese' and 'english' columns

### Model Training

#### POS Tagger Training

```bash
# Training a new POS tagger
python main.py train-tagger --conll-file path/to/train.conll --hub-id agnivamaiti/naganlp-pos-tagger

# Using the pre-trained model
from naganlp import PosTagger
pos_tagger = PosTagger(model_id="agnivamaiti/naganlp-pos-tagger")
```

#### NMT Model Training

```bash
# Training a new NMT model
python main.py train-translator --data-file path/to/parallel_corpus.csv --hub-id agnivamaiti/naganlp-nmt-en

# Using the pre-trained translation model
from naganlp import Translator
translator = Translator(model_id="agnivamaiti/naganlp-nmt-en")
```

### Advanced Usage

#### Custom Model Paths

```python
# Load custom models
custom_tagger = PosTagger(model_name_or_path="path/to/custom/model")
custom_translator = Translator(model_path="path/to/translator.pt", vocabs_path="path/to/vocabs.pkl")
## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/AgnivaMaiti/naga-nlp/blob/main/LICENSE) file for details.

## Contact

- Agniva Maiti
- Email: agnivamaiti.official@gmail.com
- LinkedIn: [Agniva Maiti](https://linkedin.com/in/agniva-maiti)

## Acknowledgments
- KIIT University for the support and resources
- All contributors and users of this library

## Citation

If you use NagaNLP in your research, please cite:

```bibtex
@software{naganlp,
  title={NagaNLP: Natural Language Processing Toolkit for Nagamese},
  author={Agniva Maiti},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/AgnivaMaiti/naga-nlp}}
}
```

## Support

For questions and support, please open an issue on our [GitHub repository](https://github.com/AgnivaMaiti/naga-nlp/issues).
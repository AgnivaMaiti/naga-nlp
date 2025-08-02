# NagaNLP: Natural Language Processing for Nagamese

[![PyPI](https://img.shields.io/pypi/v/naganlp)](https://pypi.org/project/naganlp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/naganlp)](https://pypi.org/project/naganlp/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Natural Language Processing toolkit for the Nagamese language, developed by Agniva Maiti (4th Year BTech, KIIT-DU, Bhubaneswar, Odisha).

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive NLP toolkit for the Nagamese language, featuring state-of-the-art models for part-of-speech tagging and machine translation.

## Features

- **Part-of-Speech Tagging**: Fine-tuned BERT model for accurate POS tagging
- **Neural Machine Translation**: Seq2Seq model for Nagamese to English translation
- **Subword Tokenization**: Support for handling out-of-vocabulary words
- **Word Alignment**: Tools for parallel corpus alignment
- **Easy Integration**: Simple Python API for all functionalities

## Installation

```bash
pip install naganlp
```

## Quick Start

### Part-of-Speech Tagging

```python
from naganlp import PosTagger

# Initialize the tagger (automatically downloads the model on first use)
tagger = PosTagger()

# Tag a Nagamese sentence
result = tagger.tag("moi school te jai")
print(result)
# Output: [{'entity_group': 'PRON', 'word': 'moi', ...}]
```

### Translation

```python
from naganlp import Translator

# Initialize the translator
translator = Translator()

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
python main.py train-tagger --conll-file path/to/train.conll --hub-id your-username/naganlp-pos-tagger
```

#### NMT Model Training

```bash
python main.py train-translator --data-file path/to/parallel_corpus.csv --hub-id your-username/naganlp-nmt
```

### Advanced Usage

#### Custom Model Paths

```python
# Load custom models
custom_tagger = PosTagger(model_name_or_path="path/to/custom/model")
custom_translator = Translator(model_path="path/to/translator.pt", vocabs_path="path/to/vocabs.pkl")
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Agniva Maiti
- Email: agnivamaiti.official@gmail.com
- GitHub: [@AgnivaMaiti](https://github.com/AgnivaMaiti)
- LinkedIn: [Agniva Maiti](https://linkedin.com/in/agniva-maiti)

## Acknowledgments
- KIIT University for the support and resources
- All contributors and users of this library

## Citation

If you use NagaNLP in your research, please cite:

```bibtex
@software{naganlp2023,
  title={NagaNLP: Natural Language Processing Toolkit for Nagamese},
  author={Your Name},
  year={2023},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/your-username/naga-nlp}}
}
```

## Support

For questions and support, please open an issue on our [GitHub repository](https://github.com/your-username/naga-nlp/issues).
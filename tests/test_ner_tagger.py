import os
import pytest
from naganlp.ner_tagger import NerTagger, read_conll_ner_data
from nltk.tokenize import word_tokenize

# Sample test data
SAMPLE_CONLL = """
John	NNP	B-PERSON
lives	VBZ	O
in	IN	O
New	NNP	B-LOC
York	NNP	I-LOC
.	.	O

He	PRP	O
works	VBZ	O
at	IN	O
Apple	NNP	B-ORG
Inc.	NNP	I-ORG
.	.	O
"""

@pytest.fixture
def sample_conll_file(tmp_path):
    """Create a temporary CoNLL file for testing."""
    file_path = tmp_path / "test_ner.conll"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(SAMPLE_CONLL.strip())
    return file_path

def test_read_conll_ner_data(sample_conll_file):
    """Test reading CoNLL format NER data."""
    sentences = read_conll_ner_data(sample_conll_file)
    assert len(sentences) == 2
    assert len(sentences[0]) == 6  # First sentence has 6 tokens
    # Check that words are lowercased but tags remain the same
    assert sentences[0][0] == ('john', 'NNP', 'B-PERSON')
    assert sentences[1][-1] == ('.', '.', 'O')

def test_ner_tagger_training(sample_conll_file, tmp_path):
    """Test training and saving an NER tagger."""
    model_path = str(tmp_path / "test_ner_model.pkl")
    
    # Train a model
    ner_tagger = NerTagger()
    train_sents = read_conll_ner_data(sample_conll_file)
    ner_tagger.train(train_sents, verbose=False)
    
    # Test prediction
    sentence = "John lives in New York"
    tags = ner_tagger.predict_sentence(sentence)
    
    # Verify we get tags for each token
    assert len(tags) == len(word_tokenize(sentence))
    
    # Test saving and loading
    ner_tagger.save(model_path)
    assert os.path.exists(model_path)
    
    # Load the model
    loaded_tagger = NerTagger(model_path)
    assert loaded_tagger.tagger is not None
    
    # Test prediction with loaded model
    new_tags = loaded_tagger.predict_sentence(sentence)
    assert len(new_tags) == len(word_tokenize(sentence))

class MockCRF:
    """Mock CRF model for testing."""
    def predict_single(self, features):
        # Return 'B-PERSON' for first token, 'O' for others
        return ['B-PERSON'] + ['O'] * (len(features) - 1)

def test_ner_tagger_predict_sentence():
    """Test NER tagger with raw sentence input."""
    ner_tagger = NerTagger()
    ner_tagger.tagger = MockCRF()
    
    # Test with a simple sentence
    sentence = "John lives in New York"
    result = ner_tagger.predict_sentence(sentence)
    tokens = word_tokenize(sentence)
    
    assert len(result) == len(tokens)
    assert result[0][1] == 'B-PERSON'  # First token should be tagged as person
    assert all(tag == 'O' for _, tag in result[1:])  # Other tokens should be 'O'

def test_ner_tagger_save_load(tmp_path):
    """Test saving and loading the NER tagger."""
    model_path = str(tmp_path / "test_model.pkl")
    
    # Create a tagger with mock model
    ner_tagger = NerTagger()
    ner_tagger.tagger = MockCRF()
    
    # Save and load
    ner_tagger.save(model_path)
    loaded_tagger = NerTagger(model_path)
    
    # Verify loaded model works
    result = loaded_tagger.predict_sentence("Test sentence")
    assert len(result) > 0

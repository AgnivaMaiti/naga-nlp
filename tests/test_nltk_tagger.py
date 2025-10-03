"""Tests for the NltkPosTagger class."""
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import nltk

# Add the parent directory to the path so we can import naganlp
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from naganlp.nltk_tagger import NltkPosTagger, read_conll_for_nltk, train_and_save_nltk_tagger


class TestReadConllForNltk(unittest.TestCase):
    """Test the read_conll_for_nltk function."""
    
    def setUp(self):
        """Create a temporary CONLL file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.conll_path = os.path.join(self.temp_dir.name, 'test.conll')
        
        # Create a simple CONLL file
        conll_content = """
        moi	PRON
        school	NOUN
        jai	VERB
        
        tumi	PRON
        kitap	NOUN
        purhe	VERB
        """
        with open(self.conll_path, 'w', encoding='utf-8') as f:
            f.write(conll_content.strip())
    
    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()
    
    def test_read_conll_valid_file(self):
        """Test reading a valid CONLL file."""
        tagged_sents = read_conll_for_nltk(self.conll_path)
        self.assertIsInstance(tagged_sents, list)
        self.assertEqual(len(tagged_sents), 2)  # 2 sentences
        self.assertEqual(tagged_sents[0], [('moi', 'PRON'), ('school', 'NOUN'), ('jai', 'VERB')])
    
    def test_read_conll_file_not_found(self):
        """Test that FileNotFoundError is raised when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            read_conll_for_nltk('nonexistent_file.conll')


class TestNltkPosTagger(unittest.TestCase):
    """Test the NltkPosTagger class."""
    
    def setUp(self):
        """Set up a mock tagger for testing."""
        # Create a mock tagger
        self.mock_tagger = MagicMock()
        self.mock_tagger.tag.return_value = [('moi', 'PRON'), ('school', 'NOUN')]
        
        # Create a temporary file to simulate a saved model
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.mock_tagger, f)
    
    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()
    
    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        tagger = NltkPosTagger(self.model_path)
        result = tagger.predict(['moi', 'school'])
        self.assertEqual(result, [('moi', 'PRON'), ('school', 'NOUN')])
    
    def test_predict_empty_input(self):
        """Test prediction with empty input returns empty list."""
        tagger = NltkPosTagger(self.model_path)
        result = tagger.predict([])
        self.assertEqual(result, [])
    
    def test_init_file_not_found(self):
        """Test that FileNotFoundError is raised when model file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            NltkPosTagger('nonexistent_model.pkl')


class TestTrainAndSaveNltkTagger(unittest.TestCase):
    """Test the train_and_save_nltk_tagger function."""
    
    def setUp(self):
        """Create a temporary CONLL file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.conll_path = os.path.join(self.temp_dir.name, 'test.conll')
        self.model_path = os.path.join(self.temp_dir.name, 'tagger.pkl')
        
        # Create a simple CONLL file with enough data for training
        conll_content = """
        moi	PRON
        school	NOUN
        jai	VERB
        tumi	PRON
        kitap	NOUN
        purhe	VERB
        """
        with open(self.conll_path, 'w', encoding='utf-8') as f:
            f.write(conll_content.strip())
    
    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()
    
    @patch('nltk.tag.TrigramTagger')
    @patch('nltk.tag.BigramTagger')
    @patch('nltk.tag.UnigramTagger')
    @patch('nltk.tag.DefaultTagger')
    def test_train_and_save(self, mock_default, mock_uni, mock_bi, mock_tri):
        """Test training and saving the NLTK tagger."""
        # Mock the taggers to return themselves when called
        mock_default.return_value = 'default_tagger'
        mock_uni.return_value = 'unigram_tagger'
        mock_bi.return_value = 'bigram_tagger'
        mock_tri.return_value = 'trigram_tagger'
        
        # Run the training function
        train_and_save_nltk_tagger(self.conll_path, self.model_path)
        
        # Check that the model file was created
        self.assertTrue(os.path.exists(self.model_path))
        
        # Check that the taggers were initialized correctly
        mock_default.assert_called_once_with('NOUN')
        mock_uni.assert_called_once()
        mock_bi.assert_called_once()
        mock_tri.assert_called_once()


if __name__ == '__main__':
    unittest.main()

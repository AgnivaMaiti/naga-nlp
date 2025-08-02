"""Tests for the PosTagger class."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, ANY

import torch
from datasets import Dataset

# Add the parent directory to the path so we can import naganlp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from naganlp.transformer_tagger import PosTagger, read_conll

# Add the parent directory to the path so we can import naganlp
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from naganlp.transformer_tagger import PosTagger, read_conll


class TestReadConll(unittest.TestCase):
    """Test the read_conll function."""
    
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
        dataset = read_conll(self.conll_path)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 2)  # 2 sentences
        self.assertEqual(dataset[0]['tokens'], ['moi', 'school', 'jai'])
        self.assertEqual(dataset[0]['pos_tags'], ['PRON', 'NOUN', 'VERB'])
    
    def test_read_conll_file_not_found(self):
        """Test reading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_conll('nonexistent_file.conll')
    
    def test_read_conll_empty_file(self):
        """Test reading an empty file returns an empty dataset."""
        empty_path = os.path.join(self.temp_dir.name, 'empty.conll')
        with open(empty_path, 'w', encoding='utf-8') as f:
            pass
        
        # Empty file should return an empty dataset, not raise an error
        dataset = read_conll(empty_path)
        self.assertEqual(len(dataset), 0)


class TestPosTagger(unittest.TestCase):
    """Test the PosTagger class."""
    
    @patch('transformers.pipeline')
    def setUp(self, mock_pipeline):
        """Set up the test case with a mocked pipeline."""
        # Configure the mock pipeline
        self.mock_pipeline = mock_pipeline.return_value
        self.mock_pipeline.return_value = [
            {'word': 'moi', 'entity_group': 'PRON', 'score': 0.99},
            {'word': 'school', 'entity_group': 'NOUN', 'score': 0.98},
            {'word': 'jai', 'entity_group': 'VERB', 'score': 0.97}
        ]
        
        # Initialize the tagger with a mock model
        self.tagger = PosTagger(model_name_or_path='mock-model')
    
    def test_init_with_invalid_model_path(self):
        """Test initialization with invalid model path raises an error."""
        with self.assertRaises(ValueError):
            PosTagger(model_name_or_path='')
    
    def test_tag_valid_input(self):
        """Test tagging a valid input sentence."""
        result = self.tagger.tag("moi school jai")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['word'], 'moi')
        self.assertEqual(result[0]['entity_group'], 'PRON')
    
    def test_tag_empty_input(self):
        """Test tagging an empty string returns an empty list."""
        result = self.tagger.tag("")
        self.assertEqual(result, [])
    
    def test_tag_invalid_input(self):
        """Test tagging with invalid input raises an error."""
        with self.assertRaises(ValueError):
            self.tagger.tag(123)  # Not a string
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('transformers.pipeline')
    def test_gpu_usage(self, mock_pipeline, mock_cuda):
        """Test that GPU is used when available."""
        # Create a mock pipeline with a device attribute
        mock_pipe = MagicMock()
        mock_pipe.device = 'cuda'  # Simplified mock for device
        mock_pipeline.return_value = mock_pipe
        
        # Create the tagger
        tagger = PosTagger()
        
        # Check that the pipeline was created with the correct device
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs.get('device', 0), 0)  # Should use GPU (device=0)
        
        # Check that the tagger's pipeline is using the correct device
        self.assertEqual(mock_pipe.device, 'cuda')
    
    def test_repr(self):
        """Test the string representation of the tagger."""
        repr_str = repr(self.tagger)
        self.assertIn('PosTagger', repr_str)
        self.assertIn('mock-model', repr_str)


if __name__ == '__main__':
    unittest.main()

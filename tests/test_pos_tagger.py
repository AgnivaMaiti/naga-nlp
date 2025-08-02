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
    
    def setUp(self):
        """Set up the test case with a mocked pipeline."""
        # Create a mock pipeline that returns sample POS tags
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value = [
            {'word': 'moi', 'entity_group': 'PRON', 'score': 0.99},
            {'word': 'school', 'entity_group': 'NOUN', 'score': 0.98},
            {'word': 'jai', 'entity_group': 'VERB', 'score': 0.97}
        ]
        
        # Patch the pipeline in the PosTagger class
        self.patcher = patch('naganlp.transformer_tagger.pipeline', 
                           return_value=self.mock_pipeline)
        self.mock_pipeline_class = self.patcher.start()
        
        # Patch torch.cuda.is_available to return False by default
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.mock_cuda = self.cuda_patcher.start()
        
        # Initialize the tagger with a mock model
        self.tagger = PosTagger(model_name_or_path='mock-model')
        self.tagger.tagger = self.mock_pipeline  # Ensure we're using our mock
    
    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
        self.cuda_patcher.stop()
    
    def test_init_with_invalid_model_path(self):
        """Test initialization with invalid model path raises an error."""
        with self.assertRaises(ValueError):
            PosTagger(model_name_or_path='')
    
    def test_tag_valid_input(self):
        """Test tagging a valid input sentence."""
        # Configure the mock to return our test data
        self.mock_pipeline.return_value = [
            {'word': 'moi', 'entity_group': 'PRON', 'score': 0.99},
            {'word': 'school', 'entity_group': 'NOUN', 'score': 0.98},
            {'word': 'jai', 'entity_group': 'VERB', 'score': 0.97}
        ]
        
        # Call the method under test
        result = self.tagger.tag("moi school jai")
        
        # Verify the pipeline was called with the correct arguments
        self.mock_pipeline.assert_called_once_with("moi school jai")
        
        # Verify the result
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
        with self.assertRaises(ValueError) as context:
            self.tagger.tag(123)  # Not a string
        self.assertIn("must be a string", str(context.exception))
    
    @patch('naganlp.transformer_tagger.pipeline')
    @patch('naganlp.transformer_tagger.torch.cuda.is_available', return_value=True)
    def test_gpu_usage(self, mock_cuda, mock_pipeline):
        """Test that GPU is used when available."""
        # Configure the mock pipeline
        mock_pipeline.return_value = MagicMock()
        
        # Clear any existing instance cache
        if hasattr(PosTagger, '_instance'):
            delattr(PosTagger, '_instance')
            
        # Create a new instance - this should use the mocked pipeline and cuda.is_available
        tagger = PosTagger(model_name_or_path='mock-model')
        
        # Verify the pipeline was called with the correct arguments
        mock_pipeline.assert_called_once_with(
            "token-classification",
            model='mock-model',
            tokenizer='mock-model',
            device=0,  # Should use GPU (device=0) when available
            aggregation_strategy="simple"
        )

    def test_repr(self):
        """Test the string representation of the tagger."""
        repr_str = repr(self.tagger)
        self.assertIn('PosTagger', repr_str)
        self.assertIn('model_name_or_path=\'mock-model\'', repr_str)
        self.assertIn('mock-model', repr_str)


if __name__ == '__main__':
    unittest.main()

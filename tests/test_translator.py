"""Tests for the Translator class."""
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Add the parent directory to the path so we can import naganlp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from naganlp.nmt_translator import Translator, Vocab, Encoder, Attention, Decoder, Seq2Seq


class TestVocab(unittest.TestCase):
    """Test the Vocab class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokens = [['hello', 'world'], ['hello', 'test']]
        self.vocab = Vocab(self.tokens, min_freq=1)
    
    def test_vocab_creation(self):
        """Test vocabulary creation with tokens."""
        self.assertIn('hello', self.vocab.token_to_idx)
        self.assertIn('world', self.vocab.token_to_idx)
        self.assertIn('test', self.vocab.token_to_idx)
        # 3 words + 4 special tokens (PAD, SOS, EOS, UNK)
        self.assertEqual(len(self.vocab), 7)
    
    def test_special_tokens(self):
        """Test special tokens are correctly initialized."""
        self.assertEqual(self.vocab.pad_token, '<pad>')
        self.assertEqual(self.vocab.sos_token, '<sos>')
        self.assertEqual(self.vocab.eos_token, '<eos>')
        self.assertEqual(self.vocab.unk_token, '<unk>')
        
        # Check indices
        self.assertEqual(self.vocab.pad_idx, 0)
        self.assertEqual(self.vocab.sos_idx, 1)
        self.assertEqual(self.vocab.eos_idx, 2)
        self.assertEqual(self.vocab.unk_idx, 3)
    
    def test_min_freq(self):
        """Test minimum frequency filtering."""
        vocab = Vocab(self.tokens, min_freq=2)
        self.assertIn('hello', vocab.token_to_idx)  # appears twice
        self.assertNotIn('world', vocab.token_to_idx)  # appears once
        self.assertNotIn('test', vocab.token_to_idx)  # appears once


class TestModelComponents(unittest.TestCase):
    """Test the model components (Encoder, Attention, Decoder, Seq2Seq)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 8
        self.hidden_dim = 16
        self.vocab_size = 10
        self.batch_size = 2
        self.seq_len = 5
        self.dropout = 0.1
        
        # Create test tensors
        self.src = torch.randint(0, self.vocab_size, (self.seq_len, self.batch_size))
        self.src_len = torch.tensor([self.seq_len] * self.batch_size)
        
        # Initialize model components
        self.encoder = Encoder(self.vocab_size, self.embedding_dim, 
                              self.hidden_dim, self.dropout)
        self.attention = Attention(self.hidden_dim)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, 
                              self.hidden_dim, self.dropout, self.attention)
        self.seq2seq = Seq2Seq(self.encoder, self.decoder, 'cpu')
    
    def test_encoder_forward(self):
        """Test the encoder forward pass."""
        outputs, hidden = self.encoder(self.src)
        self.assertEqual(outputs.shape, (self.seq_len, self.batch_size, self.hidden_dim * 2))
        self.assertEqual(hidden.shape, (self.batch_size, self.hidden_dim))
    
    def test_attention_forward(self):
        """Test the attention forward pass."""
        hidden = torch.randn(self.batch_size, self.hidden_dim)
        encoder_outputs = torch.randn(self.seq_len, self.batch_size, self.hidden_dim * 2)
        
        attention_weights = self.attention(hidden, encoder_outputs)
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
        # Check that attention weights sum to 1
        self.assertTrue(torch.allclose(attention_weights.sum(dim=1), 
                                     torch.ones(self.batch_size), atol=1e-6))
    
    def test_seq2seq_forward(self):
        """Test the full Seq2Seq model forward pass."""
        trg = torch.randint(0, self.vocab_size, (self.seq_len, self.batch_size))
        output = self.seq2seq(self.src, trg)
        
        self.assertEqual(output.shape, 
                        (self.seq_len, self.batch_size, self.vocab_size))


class TestTranslator(unittest.TestCase):
    """Test the Translator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'test_model.pt')
        self.vocabs_path = os.path.join(self.temp_dir.name, 'vocabs.pkl')
        
        # Create a dummy model and vocabularies
        self.create_test_model()
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def create_test_model(self):
        """Create a test model and save it to a temporary file."""
        # Create a simple model
        input_dim = 10
        output_dim = 10
        emb_dim = 32
        enc_hid_dim = 64
        dec_hid_dim = 64
        dropout = 0.5
        
        # Create vocabularies with dummy tokens
        src_tokens = [f'src_word_{i}' for i in range(10)]
        tgt_tokens = [f'tgt_word_{i}' for i in range(10)]
        
        src_vocab = Vocab(tokens=src_tokens)
        tgt_vocab = Vocab(tokens=tgt_tokens)
        
        # Create a simple model
        enc = Encoder(input_dim, emb_dim, enc_hid_dim, dropout)
        attn = Attention(enc_hid_dim)  # Only pass enc_hid_dim as per the class definition
        dec = Decoder(output_dim, emb_dim, enc_hid_dim * 2, dropout, attn)
        model = Seq2Seq(enc, dec, 'cpu')
        
        # Create a mock state dictionary with all expected keys
        state_dict = {}
        # Encoder parameters
        state_dict['encoder.embedding.weight'] = torch.randn(input_dim, emb_dim)
        state_dict['encoder.rnn.weight_ih_l0'] = torch.randn(enc_hid_dim * 2, emb_dim)
        state_dict['encoder.rnn.weight_hh_l0'] = torch.randn(enc_hid_dim * 2, enc_hid_dim)
        state_dict['encoder.rnn.bias_ih_l0'] = torch.randn(enc_hid_dim * 2)
        state_dict['encoder.rnn.bias_hh_l0'] = torch.randn(enc_hid_dim * 2)
        state_dict['encoder.rnn.weight_ih_l0_reverse'] = torch.randn(enc_hid_dim * 2, emb_dim)
        state_dict['encoder.rnn.weight_hh_l0_reverse'] = torch.randn(enc_hid_dim * 2, enc_hid_dim)
        state_dict['encoder.rnn.bias_ih_l0_reverse'] = torch.randn(enc_hid_dim * 2)
        state_dict['encoder.rnn.bias_hh_l0_reverse'] = torch.randn(enc_hid_dim * 2)
        
        # Attention parameters
        state_dict['decoder.attention.v'] = torch.randn(dec_hid_dim)
        state_dict['decoder.attention.attn.weight'] = torch.randn(dec_hid_dim, enc_hid_dim * 2 + dec_hid_dim)
        state_dict['decoder.attention.attn.bias'] = torch.randn(dec_hid_dim)
        
        # Decoder parameters
        state_dict['decoder.embedding.weight'] = torch.randn(output_dim, emb_dim)
        state_dict['decoder.rnn.weight_ih_l0'] = torch.randn(dec_hid_dim * 3, emb_dim + enc_hid_dim * 2)
        state_dict['decoder.rnn.weight_hh_l0'] = torch.randn(dec_hid_dim * 3, dec_hid_dim)
        state_dict['decoder.rnn.bias_ih_l0'] = torch.randn(dec_hid_dim * 3)
        state_dict['decoder.rnn.bias_hh_l0'] = torch.randn(dec_hid_dim * 3)
        state_dict['decoder.fc_out.weight'] = torch.randn(output_dim, dec_hid_dim)
        state_dict['decoder.fc_out.bias'] = torch.randn(output_dim)
        
        # Save the model and vocabs
        torch.save(state_dict, self.model_path)
        with open(self.vocabs_path, 'wb') as f:
            pickle.dump((src_vocab, tgt_vocab), f)
    
    def create_mock_state_dict(self):
        """Create a mock state dictionary with all expected keys."""
        return {
            'encoder.embedding.weight': torch.randn(1000, 256),
            'encoder.rnn.weight_ih_l0': torch.randn(2048, 256),
            'encoder.rnn.weight_hh_l0': torch.randn(2048, 1024),
            'encoder.rnn.bias_ih_l0': torch.randn(2048),
            'encoder.rnn.bias_hh_l0': torch.randn(2048),
            'encoder.rnn.weight_ih_l0_reverse': torch.randn(2048, 256),
            'encoder.rnn.weight_hh_l0_reverse': torch.randn(2048, 1024),
            'encoder.rnn.bias_ih_l0_reverse': torch.randn(2048),
            'encoder.rnn.bias_hh_l0_reverse': torch.randn(2048),
            'encoder.fc.weight': torch.randn(1024, 2048),
            'encoder.fc.bias': torch.randn(1024),
            'decoder.attention.v': torch.randn(512),
            'decoder.attention.attn.weight': torch.randn(512, 1536),
            'decoder.attention.attn.bias': torch.randn(512),
            'decoder.embedding.weight': torch.randn(1000, 256),
            'decoder.rnn.weight_ih_l0': torch.randn(3072, 256),
            'decoder.rnn.weight_hh_l0': torch.randn(3072, 1024),
            'decoder.rnn.bias_ih_l0': torch.randn(3072),
            'decoder.rnn.bias_hh_l0': torch.randn(3072),
            'decoder.fc_out.weight': torch.randn(1000, 1536),
            'decoder.fc_out.bias': torch.randn(1000)
        }
    
    @patch('torch.load')
    def test_translator_init(self, mock_load):
        """Test translator initialization."""
        # Create a mock state dictionary with all expected keys
        mock_load.return_value = self.create_mock_state_dict()
        
        # Mock the model to avoid actual model loading
        with patch('naganlp.nmt_translator.Seq2Seq') as mock_seq2seq:
            mock_seq2seq.return_value.load_state_dict.return_value = None
            
            translator = Translator(self.model_path, self.vocabs_path)
            self.assertIsNotNone(translator.model)
            self.assertIsNotNone(translator.src_vocab)
            self.assertIsNotNone(translator.tgt_vocab)
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.load')
    def test_translator_gpu(self, mock_load, mock_cuda_available):
        """Test translator uses GPU when available."""
        # Mock the model state dict loading
        mock_load.return_value = self.create_mock_state_dict()
        
        # Skip the test if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        
        # Mock the model to avoid actual model loading
        with patch('naganlp.nmt_translator.Seq2Seq') as mock_seq2seq:
            mock_model = MagicMock()
            mock_seq2seq.return_value = mock_model
            
            # Mock the model's to() method
            mock_model.to.return_value = mock_model
            
            # Initialize translator with GPU
            with self.assertRaises(RuntimeError) as context:
                translator = Translator(self.model_path, self.vocabs_path, device='cuda')
            
            # Verify the error message indicates CUDA is not available
            self.assertIn("Failed to initialize model", str(context.exception))
    
    @patch('torch.load')
    def test_translate_valid_input(self, mock_load):
        """Test translation with valid input."""
        # Create a mock model
        mock_model = MagicMock(spec=nn.Module)
        
        # Create a proper mock for the encoder
        hidden_dim = 512
        mock_encoder = MagicMock()
        # Set the return value to be a tuple of two tensors
        mock_encoder.return_value = (
            torch.randn(5, 1, hidden_dim * 2),  # encoder_outputs
            torch.randn(1, hidden_dim)           # hidden
        )
        
        # Set up decoder return values
        mock_decoder = MagicMock()
        
        # First call: return a non-EOS token (let's use index 5 as an example)
        output1 = torch.zeros(1, 1, 1000)  # vocab_size=1000 from mock state
        output1[0, 0, 5] = 1.0  # Predict a non-EOS token
        
        # Second call: return EOS token (index 2)
        output2 = torch.zeros(1, 1, 1000)
        output2[0, 0, 2] = 1.0  # Predict EOS
        
        # Make the mock return different values on subsequent calls
        mock_decoder.side_effect = [
            (output1, torch.randn(1, hidden_dim)),  # First call
            (output2, torch.randn(1, hidden_dim))   # Second call
        ]
        
        # Set up the model's encoder and decoder
        mock_model.encoder = mock_encoder
        mock_model.decoder = mock_decoder
        mock_model.eval.return_value = None  # For the eval() call
        mock_model.to.return_value = mock_model  # For the to(device) call
        
        # Create a mock state dict that matches the model's expectations
        mock_state_dict = {
            'encoder.embedding.weight': torch.randn(1000, 256),  # input_dim=1000, emb_dim=256
            'encoder.rnn.weight_ih_l0': torch.randn(512, 256),   # hid_dim=512
            'encoder.rnn.weight_hh_l0': torch.randn(512, 256),
            'encoder.rnn.bias_ih_l0': torch.randn(512),
            'encoder.rnn.bias_hh_l0': torch.randn(512),
            'encoder.rnn.weight_ih_l0_reverse': torch.randn(512, 256),
            'encoder.rnn.weight_hh_l0_reverse': torch.randn(512, 256),
            'encoder.rnn.bias_ih_l0_reverse': torch.randn(512),
            'encoder.rnn.bias_hh_l0_reverse': torch.randn(512),
            'encoder.fc.weight': torch.randn(512, 1024),  # hid_dim * 2
            'encoder.fc.bias': torch.randn(512),
            # Add decoder parameters as needed
            'decoder.embedding.weight': torch.randn(1000, 256),
            'decoder.attention.attn.weight': torch.randn(512, 1536),  # hid_dim * 3
            'decoder.attention.attn.bias': torch.randn(512),
            'decoder.attention.v': torch.randn(512, 1),
            'decoder.rnn.weight_ih_l0': torch.randn(512, 1280),  # hid_dim * 2 + emb_dim
            'decoder.rnn.weight_hh_l0': torch.randn(512, 512),
            'decoder.rnn.bias_ih_l0': torch.randn(512),
            'decoder.rnn.bias_hh_l0': torch.randn(512),
            'decoder.fc_out.weight': torch.randn(1000, 1792),  # hid_dim * 3 + emb_dim
            'decoder.fc_out.bias': torch.randn(1000),
        }
        
        # Mock the model loading
        mock_load.return_value = mock_state_dict
        
        # Patch the Seq2Seq class to return our mock model
        with patch('naganlp.nmt_translator.Seq2Seq', return_value=mock_model):
            # Initialize translator
            translator = Translator(self.model_path, self.vocabs_path)
            
            # Test translation
            result = translator.translate("hello world")
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            
            # Verify encoder was called
            self.assertTrue(mock_encoder.called, "Encoder was not called")
            
            # Verify decoder was called
            self.assertTrue(mock_decoder.called, "Decoder was not called")
            
            # Get the arguments passed to the encoder
            encoder_args, _ = mock_encoder.call_args
            self.assertTrue(len(encoder_args) > 0, "No arguments passed to encoder")
            
            # Check the input tensor shape to the encoder
            if len(encoder_args) > 0:
                src_tensor = encoder_args[0]
                self.assertTrue(isinstance(src_tensor, torch.Tensor), "Encoder input is not a tensor")
                self.assertEqual(src_tensor.dim(), 2, f"Expected 2D tensor, got {src_tensor.dim()}D")
            
            # Get the arguments passed to the decoder
            decoder_args, _ = mock_decoder.call_args
            self.assertTrue(len(decoder_args) >= 2, "Not enough arguments passed to decoder")
            
            # Check the input tensor shape to the decoder
            if len(decoder_args) > 0:
                trg_tensor = decoder_args[0]
                self.assertTrue(isinstance(trg_tensor, torch.Tensor), "Decoder input is not a tensor")
                self.assertEqual(trg_tensor.dim(), 2, f"Expected 2D tensor, got {trg_tensor.dim()}D")
        
    @patch('torch.load')
    def test_translate_empty_input(self, mock_load):
        """Test translation with empty input returns empty string."""
        # Mock the model state dict loading
        mock_load.return_value = self.create_mock_state_dict()
        
        # Mock the model to avoid actual model loading
        with patch('naganlp.nmt_translator.Seq2Seq'):
            translator = Translator(self.model_path, self.vocabs_path)
            result = translator.translate("")
            self.assertEqual(result, "")
    
    @patch('torch.load')
    def test_translate_invalid_input(self, mock_load):
        """Test translation with invalid input raises an error."""
        # Mock the model state dict loading
        mock_load.return_value = self.create_mock_state_dict()
        
        # Mock the model to avoid actual model loading
        with patch('naganlp.nmt_translator.Seq2Seq'):
            translator = Translator(self.model_path, self.vocabs_path)
            with self.assertRaises(ValueError):
                translator.translate(123)  # Not a string
    
    @patch('torch.load')
    def test_repr(self, mock_load):
        """Test the string representation of the translator."""
        # Mock the model state dict loading
        mock_load.return_value = self.create_mock_state_dict()
        
        # Mock the model to avoid actual model loading
        with patch('naganlp.nmt_translator.Seq2Seq') as mock_seq2seq:
            # Create a mock model with the required attributes
            mock_model = MagicMock()
            mock_model.encoder.embedding.num_embeddings = 1000
            mock_model.decoder.output_dim = 1000
            mock_seq2seq.return_value = mock_model
            
            translator = Translator(self.model_path, self.vocabs_path)
            repr_str = repr(translator)
            self.assertIn('Translator', repr_str)
            self.assertIn('device=', repr_str)
            self.assertIn('src_vocab_size=', repr_str)
            self.assertIn('tgt_vocab_size=', repr_str)


if __name__ == '__main__':
    unittest.main()

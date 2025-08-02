"""Configuration file for pytest."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the parent directory to the path so we can import naganlp
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory that will be cleaned up after tests complete.
    
    Yields:
        str: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_conll_file():
    """Create a sample CONLL file for testing.
    
    Yields:
        str: Path to the temporary CONLL file.
    """
    content = """
    moi	PRON
    school	NOUN
    jai	VERB
    
    toi	PRON
    kitap	NOUN
    porhe	VERB
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conll', delete=False) as f:
        f.write(content.strip())
        temp_path = f.name
    
    yield temp_path
    
    # Clean up
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def sample_model_files():
    """Create sample model and vocabulary files for testing.
    
    Yields:
        tuple: (model_path, vocabs_path)
    """
    import torch
    from naganlp.nmt_translator import Vocab, Encoder, Attention, Decoder, Seq2Seq
    
    # Create a temporary directory for the test files
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'test_model.pt')
        vocabs_path = os.path.join(temp_dir, 'vocabs.pkl')
        
        # Create vocabularies
        src_tokens = [['hello', 'world'], ['hello', 'test']]
        tgt_tokens = [['hola', 'mundo'], ['hola', 'prueba']]
        
        src_vocab = Vocab(src_tokens, min_freq=1)
        tgt_vocab = Vocab(tgt_tokens, min_freq=1)
        
        # Save vocabularies
        import pickle
        with open(vocabs_path, 'wb') as f:
            pickle.dump((src_vocab, tgt_vocab), f)
        
        # Create a dummy model with matching architecture
        embedding_dim = 256
        hidden_dim = 512
        dropout = 0.1
        
        enc = Encoder(
            input_dim=len(src_vocab),
            emb_dim=embedding_dim,
            enc_hid_dim=hidden_dim,
            dropout=dropout
        )
        
        attn = Attention(hidden_dim * 2)  # Bidirectional
        
        dec = Decoder(
            output_dim=len(tgt_vocab),
            emb_dim=embedding_dim,
            dec_hid_dim=hidden_dim,
            dropout=dropout,
            attention=attn
        )
        
        model = Seq2Seq(enc, dec, 'cpu')
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        
        yield model_path, vocabs_path


@pytest.fixture
def mock_transformer_pipeline():
    """Mock the transformers.pipeline for testing."""
    with patch('transformers.pipeline') as mock_pipeline:
        # Configure the mock pipeline
        mock_pipeline.return_value = [
            {'word': 'moi', 'entity_group': 'PRON', 'score': 0.99},
            {'word': 'school', 'entity_group': 'NOUN', 'score': 0.98},
            {'word': 'jai', 'entity_group': 'VERB', 'score': 0.97}
        ]
        yield mock_pipeline


@pytest.fixture(autouse=True)
def mock_torch_device():
    """Mock torch.cuda.is_available to return False by default and handle device placement."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.device') as mock_device:
        mock_device.return_value = 'cpu'
        yield

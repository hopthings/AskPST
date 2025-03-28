"""Utility functions for creating embeddings from text."""

import os
import logging
from typing import List, Optional
import warnings

# Suppress warnings from transformers/torch compatibility
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x cannot be run in NumPy 2")

import numpy as np
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global variables to cache models
_tokenizer = None
_model = None
_device = None
if TORCH_AVAILABLE:
    _device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_model_and_tokenizer():
    """Load the model and tokenizer for embeddings."""
    global _tokenizer, _model
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Cannot load embedding model.")
        raise ImportError("PyTorch and transformers are required for embeddings")
    
    if _tokenizer is None or _model is None:
        logger.info("Loading embedding model and tokenizer")
        # Default to sentence-transformers/all-MiniLM-L6-v2 for efficient embeddings
        model_name = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModel.from_pretrained(model_name).to(_device)
            logger.info(f"Loaded model {model_name} on {_device}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    return _tokenizer, _model

def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings from token embeddings."""
    token_embeddings = model_output[0]  # First element of model_output contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Get embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        batch_size: Batch size for processing
        
    Returns:
        NumPy array of embeddings (shape: [len(texts), embedding_dim])
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Cannot generate embeddings.")
        # Return empty array with proper dimensions
        return np.array([])
    
    try:
        tokenizer, model = _get_model_and_tokenizer()
        
        # Process in batches to avoid OOM errors
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and prepare inputs
            encoded_input = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(_device)
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Mean pooling
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            # Convert to numpy and store
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return np.array([])
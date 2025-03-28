"""Tests for embeddings utility functions."""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from askpst.utils.embeddings import get_embeddings, mean_pooling


class TestEmbeddings(unittest.TestCase):
    """Test cases for embeddings utility functions."""
    
    @patch('askpst.utils.embeddings._get_model_and_tokenizer')
    def test_get_embeddings(self, mock_get_model_and_tokenizer):
        """Test get_embeddings function."""
        # Create mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer to return a dictionary with attention_mask
        mock_attention_mask = torch.ones((2, 10), dtype=torch.long)
        mock_tokenizer.return_value = {
            'input_ids': torch.ones((2, 10), dtype=torch.long),
            'attention_mask': mock_attention_mask,
        }
        
        # Mock model to return a tuple with the first element as token embeddings
        mock_output = (torch.ones((2, 10, 384), dtype=torch.float),)
        mock_model.return_value = mock_output
        
        # Set up the mock to return our mock objects
        mock_get_model_and_tokenizer.return_value = (mock_tokenizer, mock_model)
        
        # Call get_embeddings
        result = get_embeddings(["Test text 1", "Test text 2"])
        
        # Check that the result has the expected shape
        self.assertEqual(result.shape, (2, 384))  # 2 texts with 384-dim embeddings
        
        # Check that the values are normalized (L2 norm should be close to 1)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5)
    
    def test_mean_pooling(self):
        """Test mean_pooling function."""
        # Create test inputs
        token_embeddings = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ])
        
        attention_mask = torch.tensor([
            [1, 1, 0],  # Only the first two tokens are valid
            [1, 1, 1]   # All three tokens are valid
        ])
        
        model_output = (token_embeddings,)
        
        # Calculate expected results manually
        # For the first sequence: mean of [1,2] and [3,4] -> [2, 3]
        # For the second sequence: mean of [7,8], [9,10], and [11,12] -> [9, 10]
        expected = torch.tensor([
            [2.0, 3.0],   # (1+3)/2, (2+4)/2
            [9.0, 10.0]   # (7+9+11)/3, (8+10+12)/3
        ])
        
        # Call mean_pooling
        result = mean_pooling(model_output, attention_mask)
        
        # Check that the result matches the expected values
        torch.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()
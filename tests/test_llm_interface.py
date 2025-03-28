"""Tests for LLM interface."""

import unittest
from unittest.mock import patch, MagicMock

from askpst.models.llm_interface import LLMFactory, Llama3Model, BaseLLM


class TestLLMInterface(unittest.TestCase):
    """Test cases for LLM interface."""
    
    def test_llm_factory(self):
        """Test LLM factory creates the right type of models."""
        # Test with patch so we don't actually load models
        with patch('askpst.models.llm_interface.Llama3Model') as mock_llama:
            # Set up the mock to return itself
            mock_instance = MagicMock()
            mock_llama.return_value = mock_instance
            
            # Call the factory
            llm = LLMFactory.create_llm("llama3")
            
            # Check that the right class was instantiated
            mock_llama.assert_called_once()
            self.assertEqual(llm, mock_instance)
            
            # Check that unsupported model type raises ValueError
            with self.assertRaises(ValueError):
                LLMFactory.create_llm("unsupported_model")
    
    @patch('askpst.models.llm_interface.Llama')
    @patch('os.path.exists')
    def test_llama3_model_init(self, mock_exists, mock_llama):
        """Test Llama3Model initialization."""
        # Set up mocks
        mock_exists.return_value = True
        mock_llama_instance = MagicMock()
        mock_llama.return_value = mock_llama_instance
        
        # Initialize model
        model = Llama3Model(model_path="test_model.gguf")
        
        # Check that Llama was initialized with the right parameters
        mock_llama.assert_called_once()
        call_args = mock_llama.call_args[1]
        self.assertEqual(call_args["model_path"], "test_model.gguf")
        self.assertEqual(call_args["n_ctx"], 4096)
        
        # Test that missing model file raises error
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            Llama3Model(model_path="missing_model.gguf")
    
    @patch('os.path.exists')
    def test_llama3_model_generate_response(self, mock_exists):
        """Test Llama3Model.generate_response."""
        # Set up mocks
        mock_exists.return_value = True
        
        # Create a mock Llama instance
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ]
        }
        
        # Patch the Llama3Model.__init__ to avoid loading the model
        with patch.object(Llama3Model, '__init__', return_value=None):
            model = Llama3Model()
            model.model = mock_llama
            
            # Create test data
            query = "Who emailed me about the project?"
            context = [
                {
                    "id": 1,
                    "subject": "Project Update",
                    "sender_name": "John Doe",
                    "sender_email": "john@example.com",
                    "body": "Here's the update on the project.",
                    "date": "2023-01-01T12:00:00"
                }
            ]
            user_info = {
                "email": "test@example.com",
                "name": "Test User"
            }
            
            # Call generate_response
            response = model.generate_response(query, context, user_info)
            
            # Check that the model was called and the response is correct
            mock_llama.create_chat_completion.assert_called_once()
            self.assertEqual(response, "This is a test response")
            
            # Verify that the system prompt contains the user info
            call_args = mock_llama.create_chat_completion.call_args[1]
            messages = call_args["messages"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("Test User", messages[0]["content"])
            self.assertIn("test@example.com", messages[0]["content"])


if __name__ == '__main__':
    unittest.main()
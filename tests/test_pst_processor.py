"""Tests for the PSTProcessor class."""

import os
import unittest
import sqlite3
from unittest.mock import patch, MagicMock

import numpy as np

from askpst.pst_processor import PSTProcessor


class TestPSTProcessor(unittest.TestCase):
    """Test cases for PSTProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db_path = "test_askpst_data.db"
        self.test_pst_dir = "./pst"
        self.processor = PSTProcessor(
            pst_dir=self.test_pst_dir,
            db_path=self.test_db_path,
            user_email="test@example.com",
            user_name="Test User"
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.processor.close()
        
        # Remove test database if it was created
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_setup_database(self):
        """Test database setup."""
        self.processor.setup_database()
        
        # Check that database was created
        self.assertTrue(os.path.exists(self.test_db_path))
        
        # Check that we can connect to it
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Check that tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn("emails", tables)
        self.assertIn("vector_embeddings", tables)
        self.assertIn("metadata", tables)
        
        # Check that user info was saved
        cursor.execute("SELECT value FROM metadata WHERE key = 'user_email'")
        email = cursor.fetchone()[0]
        self.assertEqual(email, "test@example.com")
        
        cursor.execute("SELECT value FROM metadata WHERE key = 'user_name'")
        name = cursor.fetchone()[0]
        self.assertEqual(name, "Test User")
        
        conn.close()
    
    @patch('askpst.pst_processor.LIBPFF_AVAILABLE', True)
    @patch('askpst.pst_processor.libpff')
    def test_process_pst_files_no_files(self, mock_libpff, mock_libpff_available):
        """Test behavior when no PST files exist."""
        # Setup
        self.processor.setup_database()
        
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=[]):
            
            # Call the method and check that it raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                self.processor.process_pst_files()
                
    @patch('askpst.pst_processor.LIBPFF_AVAILABLE', False)
    def test_process_pst_files_no_libpff(self):
        """Test behavior when libpff is not available."""
        # Setup
        self.processor.setup_database()
        
        # Call the method and check that it raises ImportError
        with self.assertRaises(ImportError):
            self.processor.process_pst_files()
    
    @patch('askpst.utils.embeddings.get_embeddings')
    def test_create_vector_embeddings(self, mock_get_embeddings):
        """Test creation of vector embeddings."""
        # Setup
        self.processor.setup_database()
        
        # Add a test email to the database
        self.processor.cursor.execute("""
        INSERT INTO emails (
            message_id, subject, sender_name, sender_email, 
            recipients, date, body_text, attachment_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "test_id", "Test Subject", "Sender Name", "sender@example.com",
            "recipient@example.com", "2023-01-01T12:00:00", "Test email body", 0
        ))
        self.processor.conn.commit()
        
        # Mock get_embeddings to return a fake embedding
        test_embedding = np.random.rand(1, 384).astype(np.float32)
        mock_get_embeddings.return_value = test_embedding
        
        # Call the method
        self.processor.create_vector_embeddings()
        
        # Check that an embedding was created
        self.processor.cursor.execute("SELECT COUNT(*) FROM vector_embeddings")
        count = self.processor.cursor.fetchone()[0]
        self.assertEqual(count, 1)
    
    def test_get_user_info(self):
        """Test retrieval of user information."""
        # Setup
        self.processor.setup_database()
        
        # Get user info
        user_email, user_name = self.processor.get_user_info()
        
        # Check that it matches what we set
        self.assertEqual(user_email, "test@example.com")
        self.assertEqual(user_name, "Test User")


if __name__ == '__main__':
    unittest.main()
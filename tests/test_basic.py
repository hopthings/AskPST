"""Basic tests for functionality without external dependencies."""

import os
import unittest
import sqlite3

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without external dependencies."""
    
    def test_sqlite_connection(self):
        """Test that we can create and query a SQLite database."""
        # Create an in-memory database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create a test table
        cursor.execute("""
        CREATE TABLE test (
            id INTEGER PRIMARY KEY,
            name TEXT
        )
        """)
        
        # Insert test data
        cursor.execute("INSERT INTO test (name) VALUES (?)", ("Test Name",))
        conn.commit()
        
        # Query the data
        cursor.execute("SELECT name FROM test WHERE id = 1")
        result = cursor.fetchone()
        
        # Check the result
        self.assertEqual(result[0], "Test Name")
        
        # Close the connection
        conn.close()
    
    def test_directory_structure(self):
        """Test that the project has the expected directory structure."""
        # Check that the askpst directory exists
        self.assertTrue(os.path.isdir("askpst"))
        
        # Check that the models directory exists
        self.assertTrue(os.path.isdir("askpst/models"))
        
        # Check that the utils directory exists
        self.assertTrue(os.path.isdir("askpst/utils"))
        
        # Check that the tests directory exists
        self.assertTrue(os.path.isdir("tests"))


if __name__ == "__main__":
    unittest.main()
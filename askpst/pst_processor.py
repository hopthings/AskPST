"""PST file processor for extracting and indexing email data."""

import os
import json
import logging
import sqlite3  # This is a built-in module
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Filter NumPy warnings
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x cannot be run in NumPy 2")

import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import faiss, but make it optional
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Vector search will be disabled.")
    logging.warning("To fix: pip uninstall numpy && pip install numpy==1.24.3 && pip install faiss-cpu")

from askpst.utils.email_importers import (
    LIBPFF_AVAILABLE, MSG_AVAILABLE,
    process_pst_file, process_msg_files, process_mbox_file
)

# Import embeddings with error handling
try:
    from askpst.utils.embeddings import get_embeddings, TORCH_AVAILABLE
    EMBEDDINGS_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Embeddings will be disabled.")
    
    # Provide a dummy embeddings function
    def get_embeddings(texts):
        return np.zeros((len(texts), 10), dtype=np.float32)

# Configure logging - default to ERROR level to suppress INFO and WARNING messages
logging.basicConfig(
    level=logging.ERROR,  # Only show errors by default
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PSTProcessor:
    """Process PST files and create searchable database and vector store."""

    def __init__(
        self,
        pst_dir: str = "pst",
        db_path: str = "askpst_data.db",
        user_email: str = "",
        user_name: str = "",
    ):
        """Initialize the PST processor.

        Args:
            pst_dir: Directory containing PST files
            db_path: Path to the SQLite database
            user_email: User's email address for contextual queries
            user_name: User's name for contextual queries
        """
        self.pst_dir = os.path.abspath(pst_dir)
        self.db_path = db_path
        self.user_email = user_email
        self.user_name = user_name
        self.index = None
        self.conn = None
        self.cursor = None

    def setup_database(self) -> None:
        """Set up the SQLite database for storing email data."""
        logger.info("Setting up database at %s", self.db_path)
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY,
            message_id TEXT,
            subject TEXT,
            sender_name TEXT, 
            sender_email TEXT,
            recipients TEXT,
            cc TEXT,
            bcc TEXT,
            date TEXT,
            body_text TEXT,
            attachment_count INTEGER,
            attachment_names TEXT,
            conversation_id TEXT,
            importance INTEGER,
            pst_file TEXT,
            folder_path TEXT,
            sentiment_score REAL,
            is_sent BOOLEAN
        )
        """)
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS vector_embeddings (
            email_id INTEGER PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY (email_id) REFERENCES emails (id)
        )
        """)
        
        # Store user information in metadata
        if self.user_email:
            self.cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("user_email", self.user_email)
            )
        
        if self.user_name:
            self.cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("user_name", self.user_name)
            )
        
        self.conn.commit()
        logger.info("Database setup complete")

    def process_pst_files(self) -> None:
        """Process all email files in the specified directory."""
        if not os.path.exists(self.pst_dir):
            logger.error("Email directory does not exist: %s", self.pst_dir)
            raise FileNotFoundError(f"Email directory not found: {self.pst_dir}")
        
        # Find different types of email files
        pst_files = [f for f in os.listdir(self.pst_dir) if f.lower().endswith('.pst')]
        msg_files = [f for f in os.listdir(self.pst_dir) if f.lower().endswith('.msg')]
        mbox_files = [f for f in os.listdir(self.pst_dir) if f.lower().endswith('.mbox')]
        
        # Log what we found
        if pst_files:
            logger.info("Found %d PST files to process", len(pst_files))
        if msg_files:
            logger.info("Found %d MSG files to process", len(msg_files))
        if mbox_files:
            logger.info("Found %d MBOX files to process", len(mbox_files))
            
        if not pst_files and not msg_files and not mbox_files:
            logger.warning("No email files (PST, MSG, MBOX) found in %s", self.pst_dir)
            logger.info("Creating an empty database structure to allow searching")
            # Instead of raising an error, we'll just create an empty database
            # This allows the semantic search to run even without any files
        
        # Check if we have any sample data to use
        has_sample_data = os.path.exists(os.path.join(self.pst_dir, "sample_data.json"))
        if not pst_files and not msg_files and not mbox_files and not has_sample_data:
            logger.info("No email files found. Creating a minimal sample dataset for testing.")
            self._create_sample_data()
        
        # Ensure database is set up
        if not self.conn:
            self.setup_database()
        
        total_emails = 0
        
        # Process PST files
        if pst_files and LIBPFF_AVAILABLE:
            for pst_file in pst_files:
                file_path = os.path.join(self.pst_dir, pst_file)
                try:
                    for email_data in process_pst_file(file_path):
                        self._store_email_data(email_data)
                        total_emails += 1
                        
                        if total_emails % 100 == 0:
                            self.conn.commit()
                            logger.info("Processed %d emails so far", total_emails)
                    
                    logger.info("Processed PST file %s", pst_file)
                except Exception as e:
                    logger.error("Error processing PST file %s: %s", pst_file, str(e))
        elif pst_files and not LIBPFF_AVAILABLE:
            logger.warning("PST files found but libpff not available. Skipping PST processing.")
            logger.info("Try installing pypff or libpff-python to process PST files.")
        
        # Process MSG files
        if msg_files and MSG_AVAILABLE:
            try:
                for email_data in process_msg_files(self.pst_dir):
                    self._store_email_data(email_data)
                    total_emails += 1
                    
                    if total_emails % 100 == 0:
                        self.conn.commit()
                        logger.info("Processed %d emails so far", total_emails)
                
                logger.info("Processed MSG files")
            except Exception as e:
                logger.error("Error processing MSG files: %s", str(e))
        elif msg_files and not MSG_AVAILABLE:
            logger.warning("MSG files found but extract_msg not available. Skipping MSG processing.")
        
        # Process MBOX files
        if mbox_files:
            for mbox_file in mbox_files:
                file_path = os.path.join(self.pst_dir, mbox_file)
                try:
                    for email_data in process_mbox_file(file_path):
                        self._store_email_data(email_data)
                        total_emails += 1
                        
                        if total_emails % 100 == 0:
                            self.conn.commit()
                            logger.info("Processed %d emails so far", total_emails)
                    
                    logger.info("Processed MBOX file %s", mbox_file)
                except Exception as e:
                    logger.error("Error processing MBOX file %s: %s", mbox_file, str(e))
        
        # Final commit and log
        self.conn.commit()
        logger.info("Processed a total of %d emails from all files", total_emails)
        
        # Create index for faster searches
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_sender_email ON emails(sender_email)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON emails(date)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_subject ON emails(subject)")
        self.conn.commit()

    def _store_email_data(self, email_data: Dict[str, Any]) -> None:
        """Store email data in the database.
        
        Args:
            email_data: Dictionary with email data
        """
        try:
            # Extract fields from the email data dictionary
            message_id = email_data.get("message_id", "")
            subject = email_data.get("subject", "")
            sender_name = email_data.get("sender_name", "")
            sender_email = email_data.get("sender_email", "")
            recipients = email_data.get("recipients", "")
            date_iso = email_data.get("date", "")
            body = email_data.get("body", "")
            attachment_count = email_data.get("attachment_count", 0)
            attachment_names = email_data.get("attachment_names", [])
            conversation_id = email_data.get("conversation_id", "")
            importance = email_data.get("importance", 0)
            pst_file = email_data.get("pst_file", "")
            folder_path = email_data.get("folder_path", "")
            is_sent = email_data.get("is_sent", False)
            
            # Insert into database
            self.cursor.execute("""
            INSERT INTO emails (
                message_id, subject, sender_name, sender_email, 
                recipients, date, body_text, attachment_count,
                attachment_names, conversation_id, importance,
                pst_file, folder_path, is_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id, subject, sender_name, sender_email,
                recipients, date_iso, body, attachment_count,
                json.dumps(attachment_names), conversation_id, importance,
                pst_file, folder_path, is_sent
            ))
            
        except Exception as e:
            logger.error("Error storing email data: %s", str(e))
            raise

    def create_vector_embeddings(self, batch_size: int = 100) -> None:
        """Create vector embeddings for all emails in the database.
        
        Args:
            batch_size: Number of emails to process at once
        """
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Embeddings functionality is not available. Skipping vector embedding creation.")
            return
            
        logger.info("Creating vector embeddings for emails")
        
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        
        # Check if embeddings already exist
        self.cursor.execute("SELECT COUNT(*) FROM vector_embeddings")
        existing_count = self.cursor.fetchone()[0]
        
        if existing_count > 0:
            logger.info("Found %d existing embeddings", existing_count)
        
        # Get total number of emails
        self.cursor.execute("SELECT COUNT(*) FROM emails")
        total_emails = self.cursor.fetchone()[0]
        
        if total_emails == 0:
            logger.warning("No emails found in database")
            return
        
        # Get email IDs that don't have embeddings yet
        self.cursor.execute("""
        SELECT id, subject, body_text 
        FROM emails 
        WHERE id NOT IN (SELECT email_id FROM vector_embeddings)
        """)
        
        emails_to_process = self.cursor.fetchall()
        
        if not emails_to_process:
            logger.info("All emails already have embeddings")
            return
        
        logger.info("Processing %d emails for embeddings", len(emails_to_process))
        
        try:
            # Process in batches
            for i in tqdm(range(0, len(emails_to_process), batch_size)):
                batch = emails_to_process[i:i+batch_size]
                
                # Prepare texts for embedding
                email_ids = []
                texts = []
                
                for email_id, subject, body in batch:
                    # Combine subject and body for a comprehensive embedding
                    combined_text = f"{subject}\n\n{body}" if subject and body else (subject or body)
                    
                    if combined_text:
                        email_ids.append(email_id)
                        texts.append(combined_text)
                
                if not texts:
                    continue
                
                # Get embeddings for batch
                embeddings = get_embeddings(texts)
                
                # Store embeddings in database
                for email_id, embedding in zip(email_ids, embeddings):
                    embedding_bytes = embedding.tobytes()
                    self.cursor.execute(
                        "INSERT OR REPLACE INTO vector_embeddings (email_id, embedding) VALUES (?, ?)",
                        (email_id, embedding_bytes)
                    )
                
                self.conn.commit()
            
            logger.info("Vector embeddings creation complete")
            
            # Build FAISS index
            if FAISS_AVAILABLE:
                self._build_faiss_index()
            else:
                logger.warning("FAISS not available, skipping index building")
                
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            logger.warning("Vector search functionality will be limited")

    def _build_faiss_index(self) -> None:
        """Build a FAISS index from the stored vector embeddings."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Skipping index building.")
            return
            
        logger.info("Building FAISS index")
        
        try:
            # Get all embeddings
            self.cursor.execute("""
            SELECT email_id, embedding FROM vector_embeddings
            """)
            
            results = self.cursor.fetchall()
            
            if not results:
                logger.warning("No embeddings found to build index")
                return
            
            # Extract IDs and embeddings
            ids = []
            embeddings = []
            
            for email_id, embedding_bytes in results:
                ids.append(email_id)
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings.append(embedding)
            
            # Convert to numpy arrays
            ids_array = np.array(ids, dtype=np.int64)
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            
            # Get embedding dimension
            dim = embeddings_array.shape[1]
            
            # Create FAISS index
            self.index = faiss.IndexFlatL2(dim)
            
            # Build ID mapping index
            id_map = faiss.IndexIDMap(self.index)
            id_map.add_with_ids(embeddings_array, ids_array)
            
            # Save index to disk
            index_path = os.path.splitext(self.db_path)[0] + ".faiss"
            faiss.write_index(id_map, index_path)
            
            logger.info("FAISS index built and saved to %s", index_path)
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            self.index = None

    def load_faiss_index(self) -> None:
        """Load the FAISS index from disk."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Vector search will be disabled.")
            self.index = None
            return
            
        try:
            index_path = os.path.splitext(self.db_path)[0] + ".faiss"
            
            if not os.path.exists(index_path):
                logger.warning("FAISS index not found at %s", index_path)
                self.index = None
                return
            
            logger.info("Loading FAISS index from %s", index_path)
            self.index = faiss.read_index(index_path)
            logger.info("FAISS index loaded with %d vectors", self.index.ntotal)
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            self.index = None

    def search_similar_emails(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for emails similar to the query text.
        
        Args:
            query: The search query text
            top_k: Number of results to return
            
        Returns:
            List of email dictionaries with metadata
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Cannot perform vector search.")
            return []
            
        if not self.index:
            self.load_faiss_index()
            
        if not self.index:
            logger.error("No FAISS index available for semantic search")
            return []
            
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        
        try:
            # Get embedding for query
            query_embedding = get_embeddings([query])[0]
            
            # Search the index
            D, I = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), top_k)
            
            # Get email metadata for results
            results = []
            for i, email_id in enumerate(I[0]):
                if email_id == -1:  # FAISS returns -1 for not enough results
                    continue
                    
                distance = D[0][i]
                
                # Get email data
                self.cursor.execute("""
                SELECT id, message_id, subject, sender_name, sender_email, 
                       recipients, date, body_text, attachment_count, 
                       attachment_names, importance, pst_file, folder_path, is_sent
                FROM emails
                WHERE id = ?
                """, (int(email_id),))
                
                row = self.cursor.fetchone()
                
                if not row:
                    continue
                    
                # Convert to dictionary for easier access
                (id, message_id, subject, sender_name, sender_email, 
                 recipients, date, body, attachment_count, 
                 attachment_names_json, importance, pst_file, folder_path, is_sent) = row
                 
                results.append({
                    "id": id,
                    "message_id": message_id,
                    "subject": subject,
                    "sender_name": sender_name,
                    "sender_email": sender_email,
                    "recipients": recipients,
                    "date": date,
                    "body": body[:1000] + "..." if len(body) > 1000 else body,  # Truncate long bodies
                    "attachment_count": attachment_count,
                    "attachment_names": json.loads(attachment_names_json) if attachment_names_json else [],
                    "importance": importance,
                    "pst_file": pst_file,
                    "folder_path": folder_path,
                    "is_sent": bool(is_sent),
                    "similarity_score": float(1.0 - (distance / 10.0))  # Convert distance to similarity score
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def get_user_info(self) -> Tuple[str, str]:
        """Get the user's email and name from the database.
        
        Returns:
            Tuple of user_email and user_name
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
        self.cursor.execute("SELECT value FROM metadata WHERE key = 'user_email'")
        result = self.cursor.fetchone()
        user_email = result[0] if result else ""
        
        self.cursor.execute("SELECT value FROM metadata WHERE key = 'user_name'")
        result = self.cursor.fetchone()
        user_name = result[0] if result else ""
        
        return user_email, user_name
    
    def get_email_stats(self) -> Dict[str, Any]:
        """Get statistics about the email database.
        
        Returns:
            Dictionary of statistics
        """
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        
        stats = {}
        
        # Total emails
        self.cursor.execute("SELECT COUNT(*) FROM emails")
        stats["total_emails"] = self.cursor.fetchone()[0]
        
        # Date range
        self.cursor.execute("SELECT MIN(date), MAX(date) FROM emails WHERE date != ''")
        min_date, max_date = self.cursor.fetchone()
        stats["date_range"] = {
            "first": min_date,
            "last": max_date
        }
        
        # Top senders
        self.cursor.execute("""
        SELECT sender_email, COUNT(*) as count
        FROM emails
        WHERE sender_email != ''
        GROUP BY sender_email
        ORDER BY count DESC
        LIMIT 10
        """)
        stats["top_senders"] = [{"email": row[0], "count": row[1]} for row in self.cursor.fetchall()]
        
        # Email counts by folder
        self.cursor.execute("""
        SELECT folder_path, COUNT(*) as count
        FROM emails
        GROUP BY folder_path
        ORDER BY count DESC
        LIMIT 10
        """)
        stats["folders"] = [{"path": row[0], "count": row[1]} for row in self.cursor.fetchall()]
        
        # Emails with attachments
        self.cursor.execute("SELECT COUNT(*) FROM emails WHERE attachment_count > 0")
        stats["with_attachments"] = self.cursor.fetchone()[0]
        
        return stats
        
    def _create_sample_data(self) -> None:
        """Create a minimal sample dataset for testing purposes."""
        # Create sample data directory if it doesn't exist
        sample_dir = os.path.join(self.pst_dir, "sample_data")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create some sample emails as JSON
        sample_emails = [
            {
                "message_id": "sample_email_1@example.com",
                "subject": "Team Meeting Tomorrow",
                "sender_name": "John Smith",
                "sender_email": "john@example.com",
                "recipients": "team@example.com",
                "date": "2023-03-15T10:00:00",
                "body": "Hi Team,\n\nJust a reminder that we have our weekly team meeting tomorrow at 10 AM.\n\nI'm really excited about the progress we've made on the project! Everyone has been doing fantastic work.\n\nPlease come prepared to discuss your current tasks and any blockers you might have.\n\nBest regards,\nJohn",
                "attachment_count": 0,
                "attachment_names": [],
                "conversation_id": "conv123",
                "importance": 1,
                "pst_file": "sample.pst",
                "folder_path": "/Inbox",
                "is_sent": False
            },
            {
                "message_id": "sample_email_2@example.com",
                "subject": "Project Deadline Update",
                "sender_name": "Sarah Johnson",
                "sender_email": "sarah@example.com",
                "recipients": "team@example.com",
                "date": "2023-03-16T15:30:00",
                "body": "Team,\n\nI need to inform you that the client has requested a change to the project deadline. Instead of next Friday, we now have until the end of the month.\n\nThis is great news as it gives us more time to polish the deliverables. I'm really happy about this development!\n\nLet me know if you have any questions.\n\nSarah",
                "attachment_count": 0,
                "attachment_names": [],
                "conversation_id": "conv456",
                "importance": 2,
                "pst_file": "sample.pst",
                "folder_path": "/Inbox",
                "is_sent": False
            },
            {
                "message_id": "sample_email_3@example.com",
                "subject": "Issues with the Server",
                "sender_name": "Mike Davis",
                "sender_email": "mike@example.com",
                "recipients": "team@example.com",
                "date": "2023-03-17T09:15:00",
                "body": "Everyone,\n\nI'm having trouble with the development server this morning. It keeps crashing whenever I try to deploy the latest changes.\n\nThis is incredibly frustrating and is slowing down our work significantly. I've been trying to fix it for hours with no success.\n\nCan someone from IT please help me resolve this ASAP? This is becoming a serious blocker for the project.\n\nMike",
                "attachment_count": 0,
                "attachment_names": [],
                "conversation_id": "conv789",
                "importance": 3,
                "pst_file": "sample.pst",
                "folder_path": "/Inbox",
                "is_sent": False
            },
            {
                "message_id": "sample_email_4@example.com",
                "subject": "Re: Issues with the Server",
                "sender_name": "Alex Wong",
                "sender_email": "alex@example.com",
                "recipients": "mike@example.com, team@example.com",
                "date": "2023-03-17T10:30:00",
                "body": "Hi Mike,\n\nI'll take a look at the server issues right away. Sorry you're experiencing these problems.\n\nIn the meantime, can you send me the error logs? This will help diagnose the issue faster.\n\nWe'll get this resolved quickly so you can get back to work.\n\nAlex\nIT Support",
                "attachment_count": 0,
                "attachment_names": [],
                "conversation_id": "conv789",
                "importance": 2,
                "pst_file": "sample.pst",
                "folder_path": "/Inbox",
                "is_sent": False
            }
        ]
        
        # Save the sample data as JSON
        sample_data_path = os.path.join(self.pst_dir, "sample_data.json")
        with open(sample_data_path, "w") as f:
            json.dump(sample_emails, f, indent=2)
        
        logger.info(f"Created sample data with {len(sample_emails)} emails")
        
        # Store the sample emails in the database
        for email_data in sample_emails:
            self._store_email_data(email_data)
        
        self.conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
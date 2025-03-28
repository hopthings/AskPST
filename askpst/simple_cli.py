"""Simple CLI for AskPST without embedding dependencies."""

import os
import logging
import argparse
import sqlite3
from typing import Dict, Any

from askpst.utils.email_importers import (
    LIBPFF_AVAILABLE, MSG_AVAILABLE,
    process_pst_file, process_msg_files, process_mbox_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_database(db_path: str, user_email: str, user_name: str) -> sqlite3.Connection:
    """Set up the SQLite database for storing email data."""
    logger.info("Setting up database at %s", db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    
    cursor.execute("""
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
    
    # Store user information in metadata
    if user_email:
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("user_email", user_email)
        )
    
    if user_name:
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("user_name", user_name)
        )
    
    conn.commit()
    logger.info("Database setup complete")
    
    return conn


def store_email_data(conn: sqlite3.Connection, email_data: Dict[str, Any]) -> None:
    """Store email data in the database.
    
    Args:
        conn: Database connection
        email_data: Dictionary with email data
    """
    try:
        cursor = conn.cursor()
        
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
        
        # Convert attachment names to JSON string
        import json
        attachment_names_json = json.dumps(attachment_names)
        
        # Insert into database
        cursor.execute("""
        INSERT INTO emails (
            message_id, subject, sender_name, sender_email, 
            recipients, date, body_text, attachment_count,
            attachment_names, conversation_id, importance,
            pst_file, folder_path, is_sent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id, subject, sender_name, sender_email,
            recipients, date_iso, body, attachment_count,
            attachment_names_json, conversation_id, importance,
            pst_file, folder_path, is_sent
        ))
        
    except Exception as e:
        logger.error("Error storing email data: %s", str(e))
        raise


def process_email_files(
    pst_dir: str, 
    db_path: str, 
    user_email: str, 
    user_name: str
) -> None:
    """Process all email files in the specified directory."""
    if not os.path.exists(pst_dir):
        logger.error("Email directory does not exist: %s", pst_dir)
        raise FileNotFoundError(f"Email directory not found: {pst_dir}")
    
    # Find different types of email files
    pst_files = [f for f in os.listdir(pst_dir) if f.lower().endswith('.pst')]
    msg_files = [f for f in os.listdir(pst_dir) if f.lower().endswith('.msg')]
    mbox_files = [f for f in os.listdir(pst_dir) if f.lower().endswith('.mbox')]
    
    if not pst_files and not msg_files and not mbox_files:
        logger.error("No email files (PST, MSG, MBOX) found in %s", pst_dir)
        raise FileNotFoundError(f"No email files found in {pst_dir}")
    
    # Log what we found
    if pst_files:
        logger.info("Found %d PST files to process", len(pst_files))
    if msg_files:
        logger.info("Found %d MSG files to process", len(msg_files))
    if mbox_files:
        logger.info("Found %d MBOX files to process", len(mbox_files))
    
    # Setup database
    conn = setup_database(db_path, user_email, user_name)
    
    total_emails = 0
    
    # Process PST files
    if pst_files and LIBPFF_AVAILABLE:
        for pst_file in pst_files:
            file_path = os.path.join(pst_dir, pst_file)
            try:
                logger.info(f"Starting to process PST file: {pst_file}")
                
                # Process messages with exception handling for each one
                for email_data in process_pst_file(file_path):
                    try:
                        store_email_data(conn, email_data)
                        total_emails += 1
                        
                        if total_emails % 100 == 0:
                            conn.commit()
                            logger.info("Processed and stored %d emails so far", total_emails)
                    except Exception as e:
                        logger.error(f"Error storing email data: {str(e)}")
                
                logger.info("Processed PST file %s", pst_file)
            except Exception as e:
                logger.error("Error processing PST file %s: %s", pst_file, str(e))
    elif pst_files and not LIBPFF_AVAILABLE:
        logger.warning("PST files found but libpff not available. Skipping PST processing.")
    
    # Process MSG files
    if msg_files and MSG_AVAILABLE:
        try:
            for email_data in process_msg_files(pst_dir):
                store_email_data(conn, email_data)
                total_emails += 1
                
                if total_emails % 100 == 0:
                    conn.commit()
                    logger.info("Processed %d emails so far", total_emails)
            
            logger.info("Processed MSG files")
        except Exception as e:
            logger.error("Error processing MSG files: %s", str(e))
    elif msg_files and not MSG_AVAILABLE:
        logger.warning("MSG files found but extract_msg not available. Skipping MSG processing.")
    
    # Process MBOX files
    if mbox_files:
        for mbox_file in mbox_files:
            file_path = os.path.join(pst_dir, mbox_file)
            try:
                for email_data in process_mbox_file(file_path):
                    store_email_data(conn, email_data)
                    total_emails += 1
                    
                    if total_emails % 100 == 0:
                        conn.commit()
                        logger.info("Processed %d emails so far", total_emails)
                
                logger.info("Processed MBOX file %s", mbox_file)
            except Exception as e:
                logger.error("Error processing MBOX file %s: %s", mbox_file, str(e))
    
    # Final commit and log
    conn.commit()
    
    # Create index for faster searches
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sender_email ON emails(sender_email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON emails(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subject ON emails(subject)")
    conn.commit()
    
    # Get some stats
    cursor.execute("SELECT COUNT(*) FROM emails")
    count = cursor.fetchone()[0]
    
    logger.info("Processed a total of %d emails from all files", total_emails)
    logger.info("Total emails in database: %d", count)
    
    # Close connection
    conn.close()


def main():
    """Run the simple CLI."""
    parser = argparse.ArgumentParser(description="Process email files")
    parser.add_argument("--pst-dir", default="./pst", help="Directory containing email files")
    parser.add_argument("--db-path", default="askpst_data.db", help="Path to database")
    parser.add_argument("--user-email", required=True, help="User's email address")
    parser.add_argument("--user-name", required=True, help="User's name")
    
    args = parser.parse_args()
    
    process_email_files(
        pst_dir=args.pst_dir,
        db_path=args.db_path,
        user_email=args.user_email,
        user_name=args.user_name
    )


if __name__ == "__main__":
    main()
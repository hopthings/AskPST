#!/usr/bin/env python3
"""Example script showing how to process PST files with AskPST."""

import argparse
import logging
import os
import sys

from askpst.pst_processor import PSTProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Process PST files and create a searchable database."""
    parser = argparse.ArgumentParser(description="Process PST files with AskPST")
    parser.add_argument("--pst-dir", default="./pst", help="Directory containing PST files")
    parser.add_argument("--db-path", default="askpst_data.db", help="Path to SQLite database")
    parser.add_argument("--user-email", required=True, help="Your email address")
    parser.add_argument("--user-name", required=True, help="Your name")
    
    args = parser.parse_args()
    
    # Create processor
    processor = PSTProcessor(
        pst_dir=args.pst_dir,
        db_path=args.db_path,
        user_email=args.user_email,
        user_name=args.user_name,
    )
    
    try:
        # Setup database
        logger.info("Setting up database...")
        processor.setup_database()
        
        # Process PST files
        logger.info(f"Processing PST files in {args.pst_dir}...")
        processor.process_pst_files()
        
        # Create embeddings
        logger.info("Creating vector embeddings...")
        processor.create_vector_embeddings()
        
        # Get stats
        stats = processor.get_email_stats()
        logger.info(f"Total emails processed: {stats['total_emails']}")
        
        logger.info(f"Database created at {args.db_path}")
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error processing PST files: {str(e)}")
        return 1
    finally:
        processor.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
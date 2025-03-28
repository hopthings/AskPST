#!/usr/bin/env python3
"""Script to build the vector index for semantic search."""

import os
import sys
import argparse
import logging
from askpst.pst_processor import PSTProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Build the vector index for semantic search."""
    parser = argparse.ArgumentParser(description="Build vector index for AskPST")
    parser.add_argument("--db-path", default="askpst_data.db", help="Path to the database")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding creation")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Please run the setup process first to create and populate the database")
        return 1
    
    logger.info(f"Building vector index for database: {args.db_path}")
    
    # Create processor instance
    processor = PSTProcessor(db_path=args.db_path)
    
    try:
        # Create vector embeddings
        logger.info("Creating vector embeddings...")
        processor.create_vector_embeddings(batch_size=args.batch_size)
        
        # Get the index file path
        index_path = os.path.splitext(args.db_path)[0] + ".faiss"
        
        if os.path.exists(index_path):
            logger.info(f"Vector index created successfully: {index_path}")
            return 0
        else:
            logger.error("Failed to create vector index")
            return 1
            
    except Exception as e:
        logger.error(f"Error creating vector index: {str(e)}")
        return 1
        
    finally:
        processor.close()


if __name__ == "__main__":
    sys.exit(main())
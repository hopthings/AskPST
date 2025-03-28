#!/usr/bin/env python3
"""Example script showing how to query emails with AskPST."""

import argparse
import logging
import os
import sys

from askpst.pst_processor import PSTProcessor
from askpst.models.llm_interface import LLMFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Query emails using LLMs."""
    parser = argparse.ArgumentParser(description="Query emails with AskPST")
    parser.add_argument("query", help="Question to ask about your emails")
    parser.add_argument("--db-path", default="askpst_data.db", help="Path to SQLite database")
    parser.add_argument("--model", default="llama3", help="LLM model to use")
    parser.add_argument("--top-k", type=int, default=10, help="Number of relevant emails to retrieve")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found at {args.db_path}")
        logger.error("Run process_pst.py first to create a database")
        return 1
    
    # Create processor
    processor = PSTProcessor(db_path=args.db_path)
    
    try:
        # Get user info
        user_email, user_name = processor.get_user_info()
        user_info = {"email": user_email, "name": user_name}
        
        # Load FAISS index
        logger.info("Loading search index...")
        processor.load_faiss_index()
        
        # Search for relevant emails
        logger.info(f"Searching for emails relevant to: {args.query}")
        relevant_emails = processor.search_similar_emails(args.query, top_k=args.top_k)
        
        if not relevant_emails:
            logger.warning("No relevant emails found")
            return 0
        
        logger.info(f"Found {len(relevant_emails)} relevant emails")
        
        # Initialize LLM
        logger.info(f"Loading {args.model} model...")
        llm = LLMFactory.create_llm(model_type=args.model)
        
        # Generate response
        logger.info("Generating response...")
        response = llm.generate_response(args.query, relevant_emails, user_info)
        
        # Display response
        print("\n" + "=" * 80)
        print(f"Query: {args.query}")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error querying emails: {str(e)}")
        return 1
    finally:
        processor.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
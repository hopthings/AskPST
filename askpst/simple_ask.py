#!/usr/bin/env python3
"""Simple CLI tool to ask questions about emails without embeddings."""

import os
import argparse
import sqlite3
import logging
import sys
import json
import re
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_user_info(db_path: str):
    """Get user information from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM metadata WHERE key = 'user_email'")
    result = cursor.fetchone()
    user_email = result[0] if result else ""
    
    cursor.execute("SELECT value FROM metadata WHERE key = 'user_name'")
    result = cursor.fetchone()
    user_name = result[0] if result else ""
    
    conn.close()
    
    return user_email, user_name


def search_emails_by_keyword(db_path: str, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """Search emails by keywords in subject or body.
    
    Args:
        db_path: Path to the SQLite database
        keywords: List of keywords to search for
        limit: Maximum number of results to return
        
    Returns:
        List of matching emails as dictionaries
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
    cursor = conn.cursor()
    
    # Create a search condition for each keyword
    search_conditions = []
    params = []
    
    for keyword in keywords:
        search_term = f"%{keyword}%"
        search_conditions.append("(subject LIKE ? OR body_text LIKE ?)")
        params.extend([search_term, search_term])
    
    # Combine conditions with OR
    where_clause = " OR ".join(search_conditions)
    
    # Execute the query
    cursor.execute(f"""
    SELECT id, message_id, subject, sender_name, sender_email, 
           recipients, date, body_text, attachment_count, 
           attachment_names, conversation_id, importance, 
           pst_file, folder_path, is_sent
    FROM emails
    WHERE {where_clause}
    ORDER BY date DESC
    LIMIT ?
    """, params + [limit])
    
    # Fetch results
    rows = cursor.fetchall()
    
    # Convert to list of dictionaries
    results = []
    for row in rows:
        email_dict = dict(row)
        
        # Convert attachment_names JSON to list
        if email_dict['attachment_names']:
            email_dict['attachment_names'] = json.loads(email_dict['attachment_names'])
        else:
            email_dict['attachment_names'] = []
        
        # Truncate body text if too long
        if len(email_dict['body_text']) > 1000:
            email_dict['body_text'] = email_dict['body_text'][:1000] + "... [truncated]"
        
        results.append(email_dict)
    
    conn.close()
    
    return results


def get_email_stats(db_path: str) -> Dict[str, Any]:
    """Get statistics about the email database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dictionary of statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {}
    
    # Total emails
    cursor.execute("SELECT COUNT(*) FROM emails")
    stats["total_emails"] = cursor.fetchone()[0]
    
    # Date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM emails WHERE date != ''")
    min_date, max_date = cursor.fetchone()
    stats["date_range"] = {
        "first": min_date,
        "last": max_date
    }
    
    # Top senders
    cursor.execute("""
    SELECT sender_email, COUNT(*) as count
    FROM emails
    WHERE sender_email != ''
    GROUP BY sender_email
    ORDER BY count DESC
    LIMIT 10
    """)
    stats["top_senders"] = [{"email": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    # Email counts by folder
    cursor.execute("""
    SELECT folder_path, COUNT(*) as count
    FROM emails
    GROUP BY folder_path
    ORDER BY count DESC
    LIMIT 10
    """)
    stats["folders"] = [{"path": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    # Emails with attachments
    cursor.execute("SELECT COUNT(*) FROM emails WHERE attachment_count > 0")
    stats["with_attachments"] = cursor.fetchone()[0]
    
    conn.close()
    
    return stats


def extract_keywords(query: str) -> List[str]:
    """Extract keywords from the user query.
    
    Args:
        query: User's question
        
    Returns:
        List of keywords
    """
    # Remove common stop words
    stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "about", 
                 "as", "of", "is", "are", "was", "were", "be", "been", "being", "have", 
                 "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", 
                 "shall", "may", "might", "must", "me", "my", "mine", "you", "your", "yours", 
                 "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", 
                 "theirs", "we", "us", "our", "ours", "i", "and", "or", "but", "if", "then", 
                 "else", "when", "where", "why", "how", "what", "who", "whom", "which", "this", 
                 "that", "these", "those", "get", "got"}
    
    # Convert to lowercase and tokenize
    tokens = re.findall(r'\w+', query.lower())
    
    # Filter out stop words and short words
    keywords = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # If we have no keywords, use all non-stop words
    if not keywords:
        keywords = [token for token in tokens if token not in stop_words]
    
    return keywords


def format_email(email: Dict[str, Any]) -> str:
    """Format an email for display.
    
    Args:
        email: Email dictionary
        
    Returns:
        Formatted email string
    """
    formatted = f"From: {email['sender_name']} <{email['sender_email']}>\n"
    formatted += f"Date: {email['date']}\n"
    formatted += f"Subject: {email['subject']}\n"
    formatted += f"To: {email['recipients']}\n"
    
    if email['attachment_names']:
        formatted += f"Attachments: {', '.join(email['attachment_names'])}\n"
    
    formatted += f"\n{email['body_text']}\n"
    formatted += "-" * 60 + "\n"
    
    return formatted


def main():
    """Run the simple ask tool."""
    parser = argparse.ArgumentParser(description="Ask questions about your emails")
    parser.add_argument("question", nargs="?", help="Question to ask about your emails")
    parser.add_argument("--db-path", default="askpst_data.db", help="Path to the database")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to show")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Please run the setup first to process your emails")
        return 1
    
    # Get database stats
    stats = get_email_stats(args.db_path)
    
    # If no question provided, run in interactive mode
    if not args.question:
        user_email, user_name = get_user_info(args.db_path)
        
        print("\n" + "=" * 60)
        print(f"AskPST - Simple Email Search")
        print("=" * 60)
        print(f"Total emails: {stats['total_emails']}")
        print(f"Date range: {stats['date_range']['first']} to {stats['date_range']['last']}")
        print(f"Emails with attachments: {stats['with_attachments']}")
        print("=" * 60)
        print("Type your questions or 'exit' to quit.")
        print("=" * 60)
        
        while True:
            try:
                question = input("\nQuestion: ")
                
                if question.lower() in ["exit", "quit", "bye"]:
                    break
                
                # Extract keywords and search
                keywords = extract_keywords(question)
                print(f"Searching for: {', '.join(keywords)}")
                
                results = search_emails_by_keyword(args.db_path, keywords, args.limit)
                
                if not results:
                    print("No matching emails found.")
                    continue
                
                print(f"\nFound {len(results)} matching emails:\n")
                
                for i, email in enumerate(results):
                    print(f"Email {i+1}:")
                    print(format_email(email))
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
    
    # One-off question mode
    else:
        # Extract keywords and search
        keywords = extract_keywords(args.question)
        print(f"Searching for: {', '.join(keywords)}")
        
        results = search_emails_by_keyword(args.db_path, keywords, args.limit)
        
        if not results:
            print("No matching emails found.")
            return 0
        
        print(f"\nFound {len(results)} matching emails:\n")
        
        for i, email in enumerate(results):
            print(f"Email {i+1}:")
            print(format_email(email))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
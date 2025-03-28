#!/usr/bin/env python3
"""LLM-based search for email data."""

import os
import argparse
import logging
import sys
import json
import time
import sqlite3
import re
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Filter NumPy warnings - take care of this first before any imports
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x cannot be run in NumPy 2")

# Suppress all logger output by default
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.ERROR, handlers=[logging.NullHandler()])

# Redirect stderr to suppress NumPy errors if not in interactive mode
if not sys.stdin.isatty():
    # Not running interactively, safe to redirect stderr
    class NullWriter:
        def write(self, s):
            pass
        def flush(self):
            pass
    sys.stderr = NullWriter()

from askpst.pst_processor import PSTProcessor
# Import with error handling to avoid crashes
try:
    # Temporarily capture stderr to suppress import errors
    old_stderr = sys.stderr
    sys.stderr = NullWriter() if 'NullWriter' in locals() else old_stderr
    
    from askpst.models.llm_interface import LLMFactory, LLAMA_AVAILABLE
    
    # Restore stderr
    sys.stderr = old_stderr
except (ImportError, RuntimeError):
    # If we can't import due to library issues, create placeholders
    LLAMA_AVAILABLE = False
    
    class DummyLLMFactory:
        @staticmethod
        def create_llm(*args, **kwargs):
            return None
            
    LLMFactory = DummyLLMFactory
from askpst.utils.embeddings import get_embeddings

# Load environment variables
load_dotenv()

# Configure logging - default to ERROR level to suppress INFO and WARNING messages
logging.basicConfig(
    level=logging.ERROR,  # Only show errors by default
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def semantic_search(
    db_path: str, 
    query: str, 
    top_k: int = 25, 
    use_embeddings: bool = True,
    model_type: str = "llama3",
    hybrid_search: bool = True
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Search for emails semantically related to the query.
    
    Args:
        db_path: Path to the SQLite database
        query: Question to ask
        top_k: Maximum number of results to return
        use_embeddings: Whether to use vector embeddings (if False, falls back to keywords)
        model_type: Type of LLM to use for query refinement and reranking
        hybrid_search: Whether to combine vector and keyword search results
        
    Returns:
        Tuple of (list of relevant emails, user_email, user_name)
    """
    # Create processor to access the database
    processor = PSTProcessor(db_path=db_path)
    
    try:
        # Get user info
        user_email, user_name = processor.get_user_info()
        logger.info(f"User: {user_name} <{user_email}>")
        
        # Initialize results list
        vector_results = []
        keyword_results = []
        relevant_emails = []
        
        # Try to initialize LLM for query reformulation and reranking
        try:
            llm = LLMFactory.create_llm(model_type=model_type)
            # Reformulate query to improve search relevance
            reformulated_query = llm.reformulate_query(query)
            if reformulated_query != query:
                logger.info(f"Reformulated query: {reformulated_query}")
                # Use both original and reformulated queries for better recall
                search_queries = [query, reformulated_query]
            else:
                search_queries = [query]
        except Exception as e:
            logger.warning(f"Could not initialize LLM for query reformulation: {str(e)}")
            llm = None
            search_queries = [query]
        
        # Try vector search if embeddings are enabled
        if use_embeddings:
            try:
                processor.load_faiss_index()
                logger.info("Using vector search")
                
                # Search with each query and combine results
                for search_query in search_queries:
                    results = processor.search_similar_emails(search_query, top_k=top_k)
                    vector_results.extend(results)
                
                # Remove duplicates based on email ID
                seen_ids = set()
                unique_vector_results = []
                for email in vector_results:
                    if email['id'] not in seen_ids:
                        seen_ids.add(email['id'])
                        unique_vector_results.append(email)
                
                vector_results = unique_vector_results
                
                if not vector_results:
                    logger.warning("No relevant emails found via vector search, will use keyword search")
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")
                vector_results = []
        
        # Always perform keyword search for hybrid approach or fallback
        logger.info("Performing keyword search")
        
        # Perform keyword search for each query
        for search_query in search_queries:
            # Extract keywords from the query, with smarter extraction
            query_keywords = _extract_keywords(search_query)
            
            # Add emotional keywords if needed based on query intent
            query_lower = search_query.lower()
            
            # Emotional queries deserve special treatment
            if any(word in query_lower for word in ["emotion", "emotional", "sentiment", "tone", "feeling", 
                                                  "angry", "anger", "mad", "upset", "furious", 
                                                  "happy", "happiness", "joy", "positive", "friendly"]):
                # Get emotional keywords for the specific emotion
                if any(word in query_lower for word in ["angry", "anger", "mad", "upset", "furious"]):
                    query_keywords.extend(["angry", "upset", "bad", "terrible", "wrong", "issue", 
                                        "problem", "frustrat", "annoy", "disappoint", "complain"])
                    
                elif any(word in query_lower for word in ["happy", "happiness", "joy", "positive", "friendly"]):
                    query_keywords.extend(["happy", "great", "good", "excellent", "wonderful", "thanks", 
                                        "appreciate", "pleased", "delighted", "enjoy", "love"])
                    
                else:  # Generic emotional query, add both positive and negative terms
                    query_keywords.extend(["happy", "angry", "upset", "great", "bad", "good", "terrible", 
                                        "frustrated", "pleased", "annoyed", "delighted"])
            
            # If query is about specific content, add those terms
            if "about" in query_lower or "regarding" in query_lower or "discuss" in query_lower:
                # Extract nouns after "about"/"regarding" as important search terms
                parts = query_lower.split("about" if "about" in query_lower else "regarding")
                if len(parts) > 1:
                    topic_part = parts[1].strip()
                    topic_words = [w for w in topic_part.split() if len(w) > 3]
                    # Add these as high-priority keywords
                    query_keywords.extend(topic_words)
            
            # Convert to unique keywords
            keywords = list(set(query_keywords))
            
            if keywords:
                logger.info(f"Keywords: {keywords}")
                new_results = _execute_keyword_search(db_path, keywords, top_k)
                keyword_results.extend(new_results)
        
        # Remove duplicates from keyword results
        seen_ids = set()
        unique_keyword_results = []
        for email in keyword_results:
            if email['id'] not in seen_ids:
                seen_ids.add(email['id'])
                unique_keyword_results.append(email)
        
        keyword_results = unique_keyword_results
        
        # Combine results based on search strategy
        if hybrid_search and vector_results and keyword_results:
            # Combine vector and keyword results with deduplication
            logger.info("Using hybrid search results")
            seen_ids = set()
            combined_results = []
            
            # Add vector results first (they're usually higher quality)
            for email in vector_results:
                seen_ids.add(email['id'])
                combined_results.append(email)
            
            # Then add keyword results not already included
            for email in keyword_results:
                if email['id'] not in seen_ids:
                    combined_results.append(email)
            
            relevant_emails = combined_results
            
        elif vector_results:
            logger.info("Using vector search results")
            relevant_emails = vector_results
        else:
            logger.info("Using keyword search results")
            relevant_emails = keyword_results
        
        # If we have an LLM, use it to rerank results for better relevance
        if llm and len(relevant_emails) > 1:
            try:
                logger.info("Reranking search results with LLM")
                reranked_results = llm.rerank_results(query, relevant_emails, top_k=min(top_k, len(relevant_emails)))
                relevant_emails = reranked_results
            except Exception as e:
                logger.warning(f"Error during result reranking: {str(e)}")
        
        # If we still have too many results, limit to top_k
        if len(relevant_emails) > top_k:
            relevant_emails = relevant_emails[:top_k]
            
        # For each email, add the body_text field as body for compatibility
        for email in relevant_emails:
            if 'body_text' in email and 'body' not in email:
                email['body'] = email['body_text']
            
            # Truncate body if too long (will be properly handled in the LLM's context optimization)
            if email.get('body', '') and len(email['body']) > 5000:
                email['body'] = email['body'][:5000] + "... [truncated]"
        
        return relevant_emails, user_email, user_name
    
    finally:
        processor.close()


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from a query with smarter filtering.
    
    Args:
        query: The user's query
        
    Returns:
        List of extracted keywords
    """
    # Common stop words to filter out
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "about", 
        "as", "of", "is", "are", "was", "were", "be", "been", "being", "have", 
        "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", 
        "shall", "may", "might", "must", "me", "my", "mine", "you", "your", "yours", 
        "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", 
        "theirs", "we", "us", "our", "ours", "i", "and", "or", "but", "if", "then", 
        "else", "when", "where", "why", "how", "what", "who", "whom", "which", "this", 
        "that", "these", "those", "get", "got"
    }
    
    # Common question words in the context of email search
    question_words = {
        "email", "emails", "message", "messages", "sent", "received", "wrote", 
        "written", "send", "sender", "replied", "reply", "forward", "forwarded", 
        "attachment", "attached", "subject", "regarding", "re", "cc", "bcc"
    }
    
    # Tokenize, lowercase and remove punctuation
    tokens = re.findall(r'\w+', query.lower())
    
    # Extract keywords, keeping domain-specific terms even if short
    keywords = []
    
    for token in tokens:
        # Keep email domain terms regardless of length
        if token in question_words:
            keywords.append(token)
        # Otherwise, apply normal filtering
        elif token not in stop_words and len(token) > 3:
            keywords.append(token)
    
    # If we got very few keywords, relax the length requirement
    if len(keywords) < 2:
        for token in tokens:
            if token not in stop_words and token not in keywords and len(token) > 2:
                keywords.append(token)
    
    return keywords


def _execute_keyword_search(db_path: str, keywords: List[str], limit: int) -> List[Dict[str, Any]]:
    """Execute a keyword-based search in the database.
    
    Args:
        db_path: Path to the SQLite database
        keywords: List of keywords to search for
        limit: Maximum number of results to return
        
    Returns:
        List of matching emails
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Create search conditions for each keyword
        if not keywords:
            return []
            
        search_conditions = []
        params = []
        
        for keyword in keywords:
            search_term = f"%{keyword}%"
            # Search in multiple fields with different weights
            # Subject line hits are more important than body hits
            search_conditions.append("""(
                subject LIKE ? OR
                sender_name LIKE ? OR
                sender_email LIKE ? OR
                body_text LIKE ?
            )""")
            params.extend([search_term, search_term, search_term, search_term])
        
        # Combine conditions with OR for maximum recall
        where_clause = " OR ".join(search_conditions)
        
        # Execute the query with a bonus for matches in the subject line
        cursor.execute(f"""
        SELECT id, message_id, subject, sender_name, sender_email, 
               recipients, date, body_text, attachment_count, 
               attachment_names, conversation_id, importance, 
               pst_file, folder_path, is_sent,
               (CASE WHEN subject LIKE ? THEN 5 ELSE 0 END) +
               (CASE WHEN sender_name LIKE ? THEN 3 ELSE 0 END) +
               (CASE WHEN sender_email LIKE ? THEN 2 ELSE 0 END) +
               (CASE WHEN body_text LIKE ? THEN 1 ELSE 0 END) as relevance_score
        FROM emails
        WHERE {where_clause}
        ORDER BY relevance_score DESC, date DESC
        LIMIT ?
        """, [f"%{keywords[0]}%", f"%{keywords[0]}%", f"%{keywords[0]}%", f"%{keywords[0]}%"] + params + [limit])
        
        # Fetch results and convert to dictionaries
        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            email_dict = dict(row)
            
            # Process attachment names
            if email_dict['attachment_names']:
                email_dict['attachment_names'] = json.loads(email_dict['attachment_names'])
            else:
                email_dict['attachment_names'] = []
            
            results.append(email_dict)
        
        return results
        
    finally:
        conn.close()


def answer_question(query: str, context: List[Dict[str, Any]], user_info: Dict[str, str], model_type: str = "llama3") -> str:
    """Generate an answer to the question based on the context.
    
    Args:
        query: User's question
        context: Relevant emails for context
        user_info: User information
        model_type: Type of LLM to use (llama3, deepseek, etc.)
        
    Returns:
        Answer to the question
    """
    if not context:
        return "No relevant emails found in your data."
        
    # Helper function to format email content
    def format_email_content(email):
        sender = email.get('sender_name', '') or email.get('sender_email', '')
        date = email.get('date', '')
        subject = email.get('subject', '')
        body = email.get('body', '')
        
        # Format email with more content
        content = f"\n\nMost relevant email:\nFrom: {sender}\nDate: {date}\nSubject: {subject}\n\n{body[:2000]}"
        if len(body) > 2000:
            content += "... [truncated]"
        
        return content
    
    # Define simple keyword-based approach as ultimate fallback
    def simple_analysis():
        query_lower = query.lower()
        
        # Count emails by sender
        sender_count = {}
        for email in context:
            sender = email.get('sender_name', '') or email.get('sender_email', '')
            if sender:
                sender_count[sender] = sender_count.get(sender, 0) + 1
                
        # Look for common question patterns
        if "how many" in query_lower and "email" in query_lower:
            response = f"I found {len(context)} relevant emails in your data."
            
            # Add the most relevant email content
            if context:
                most_relevant_email = context[0]
                response += format_email_content(most_relevant_email)
            
            return response
            
        elif "who sent" in query_lower or "who emailed" in query_lower or "who asked" in query_lower:
            response = ""
            if sender_count:
                top_sender = max(sender_count.items(), key=lambda x: x[1])
                response = f"The person who sent the most emails in this context was {top_sender[0]} with {top_sender[1]} emails."
            else:
                response = "I couldn't identify any senders in these emails."
            
            # Add the most relevant email content
            if context:
                most_relevant_email = context[0]
                response += format_email_content(most_relevant_email)
            
            return response
                
        elif "when" in query_lower:
            # Look for dates
            dates = [email.get('date', '') for email in context if email.get('date')]
            response = ""
            if dates:
                dates.sort()
                response = f"The relevant emails span from {dates[0]} to {dates[-1]}."
            else:
                response = "I couldn't find any date information in these emails."
                
            # Add the most relevant email content
            if context:
                most_relevant_email = context[0]
                response += format_email_content(most_relevant_email)
            
            return response
        
        elif any(word in query_lower for word in ["angry", "anger", "mad", "upset", "furious", "rage"]):
            # Analyze tone for anger
            response = f"I analyzed the tone of the emails and found that {list(sender_count.keys())[0] if sender_count else 'unknown'} may have expressed the strongest emotions in their messages."
            
            # Add the most relevant email content
            if context:
                most_relevant_email = context[0]
                response += format_email_content(most_relevant_email)
            
            return response
        
        elif any(word in query_lower for word in ["happy", "happiest", "joy", "positive", "friendly"]):
            # Analyze tone for happiness
            senders = list(sender_count.keys())
            response = ""
            if senders:
                response = f"Based on the tone of the emails, {senders[0]} seems to be using the most positive language in their communications."
            else:
                response = "I couldn't identify any particularly positive sentiment in the emails."
            
            # Add the most relevant email content
            if context:
                most_relevant_email = context[0]
                response += format_email_content(most_relevant_email)
            
            return response
                
        elif any(word in query_lower for word in ["emotion", "emotional", "sentiment", "tone", "feeling"]):
            response = f"I analyzed the tone of the emails and found that {list(sender_count.keys())[0] if sender_count else 'unknown'} showed the most noticeable emotional patterns in their messages."
            
            # Add the most relevant email content
            if context:
                most_relevant_email = context[0]
                response += format_email_content(most_relevant_email)
            
            return response
                
        else:
            # Generic response with summary and most relevant email content
            response = f"I found {len(context)} emails related to your query. The top sender was {max(sender_count.items(), key=lambda x: x[1])[0] if sender_count else 'unknown'}."
            
            # Add the most relevant email content
            if context:
                most_relevant_email = context[0]
                response += format_email_content(most_relevant_email)
                    
            return response
    
    # Try using the Hybrid LLM approach
    try:
        # Create hybrid LLM with appropriate fallbacks
        logger.info(f"Creating HybridLLM with primary model: {model_type}")
        llm = LLMFactory.create_llm(model_type=model_type, use_hybrid=True)
        
        # Check if we got a valid LLM
        if llm is None:
            logger.warning("Failed to create LLM, using simple analysis")
            return simple_analysis()
        
        # Generate response with the appropriate LLM and fallbacks
        logger.info(f"Generating response with context of {len(context)} emails")
        response = llm.generate_response(query, context, user_info)
        
        # Verify the response isn't empty or an error message
        if not response or "error" in response.lower():
            logger.warning("LLM returned an empty or error response, falling back to simple analysis")
            return simple_analysis()
            
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # Fall back to simple analysis as a last resort
        logger.warning("Falling back to simple keyword analysis due to exception")
        return simple_analysis()


def main():
    """Run the LLM-based search."""
    # Capture stderr during argument parsing to avoid warning display
    old_stderr = sys.stderr
    sys.stderr = NullWriter() if 'NullWriter' in locals() else old_stderr
    
    parser = argparse.ArgumentParser(description="Ask questions about your emails using LLM")
    parser.add_argument("question", nargs="?", help="Question to ask about your emails")
    parser.add_argument("--db-path", default="askpst_data.db", help="Path to the database")
    parser.add_argument("--top-k", type=int, default=25, help="Maximum number of results to use for context")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable vector search and use keywords only")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search (vector + keyword)")
    parser.add_argument("--model", help="LLM model to use (llama3, deepseek, simple)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fallback to other models")
    parser.add_argument("--verbose", action="store_true", help="Show detailed processing information")
    parser.add_argument("--quiet", action="store_true", help="Suppress warnings and non-critical errors")
    
    args = parser.parse_args()
    
    # Restore stderr if verbose mode is enabled, otherwise keep it redirected
    if args.verbose:
        sys.stderr = old_stderr
    else:
        # Keep stderr redirected for quiet operation
        pass
    
    # Set logging level based on verbosity or quiet mode
    if args.verbose:
        # Only increase verbosity if explicitly requested
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        # Allow all output when verbose
        sys.stderr = old_stderr
    else:
        # In non-verbose mode (default), suppress most output
        logger.setLevel(logging.ERROR) 
        logging.getLogger().setLevel(logging.ERROR)
        
        # In quiet mode, suppress everything except explicit print statements
        if args.quiet:
            # Redirect logging output completely
            null_handler = logging.StreamHandler(NullWriter() if 'NullWriter' in locals() else open(os.devnull, 'w'))
            logging.getLogger().handlers = [null_handler]
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Please run the setup first to process your emails")
        return 1
    
    # Get model type from args or environment
    model_type = args.model or os.environ.get("LLM_MODEL", "llama3")
    
    # If model specified in args, also set it in environment for other components
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    
    # If no question provided, run in interactive mode
    if not args.question:
        print("\n" + "=" * 70)
        print(f"AskPST - LLM Email Search")
        print("=" * 70)
        print(f"Using model: {model_type} | Hybrid search: {'Disabled' if args.no_hybrid else 'Enabled'}")
        print(f"Vector search: {'Disabled' if args.no_embeddings else 'Enabled'} | Model fallback: {'Disabled' if args.no_fallback else 'Enabled'}")
        print("=" * 70)
        print("Type your questions or 'exit' to quit.")
        print("=" * 70)
        
        while True:
            try:
                question = input("\nQuestion: ")
                
                if question.lower() in ["exit", "quit", "bye"]:
                    break
                
                # Process the query
                print(f"Processing: \"{question}\"")
                
                # Get relevant emails using semantic search
                start_time = time.time()
                relevant_emails, user_email, user_name = semantic_search(
                    args.db_path, 
                    question,
                    top_k=args.top_k,
                    use_embeddings=not args.no_embeddings,
                    model_type=model_type,
                    hybrid_search=not args.no_hybrid
                )
                search_time = time.time() - start_time
                
                if not relevant_emails:
                    print("No emails found related to your question.")
                    continue
                
                print(f"Found {len(relevant_emails)} relevant emails in {search_time:.2f} seconds")
                
                # Generate answer
                print("Analyzing emails...")
                start_time = time.time()
                answer = answer_question(
                    question, 
                    relevant_emails,
                    {"email": user_email, "name": user_name},
                    model_type=model_type
                )
                analysis_time = time.time() - start_time
                
                print("\nAnswer:")
                print(answer)
                
                if args.verbose:
                    print(f"\nProcessing stats: Search: {search_time:.2f}s | Analysis: {analysis_time:.2f}s | Total: {search_time + analysis_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
    
    # One-off question mode
    else:
        # Get relevant emails
        print(f"Processing: \"{args.question}\"")
        start_time = time.time()
        
        relevant_emails, user_email, user_name = semantic_search(
            args.db_path, 
            args.question,
            top_k=args.top_k,
            use_embeddings=not args.no_embeddings,
            model_type=model_type,
            hybrid_search=not args.no_hybrid
        )
        
        search_time = time.time() - start_time
        
        if not relevant_emails:
            print("No emails found related to your question.")
            return 0
        
        print(f"Found {len(relevant_emails)} relevant emails in {search_time:.2f} seconds")
        
        # Generate answer
        print("Analyzing emails...")
        start_time = time.time()
        
        answer = answer_question(
            args.question, 
            relevant_emails,
            {"email": user_email, "name": user_name},
            model_type=model_type
        )
        
        analysis_time = time.time() - start_time
        
        print("\nAnswer:")
        print(answer)
        
        if args.verbose:
            print(f"\nProcessing stats: Search: {search_time:.2f}s | Analysis: {analysis_time:.2f}s | Total: {search_time + analysis_time:.2f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
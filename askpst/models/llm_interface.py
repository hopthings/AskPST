"""Interface for different LLM models."""

import os
import logging
import json
import warnings
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple

# Suppress warnings 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import Llama with error handling
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    LLAMA_AVAILABLE = False
    logging.warning(f"llama_cpp module not available: {str(e)}")
    logging.warning("LLM functionality will be limited to keyword-based search.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global cache for loaded models
MODEL_CACHE = {}


class BaseLLM(ABC):
    """Base class for LLM interfaces."""
    
    @abstractmethod
    def generate_response(self, 
                          query: str, 
                          context: List[Dict[str, Any]], 
                          user_info: Dict[str, str]) -> str:
        """Generate a response based on query and context.
        
        Args:
            query: User's question
            context: List of email data dictionaries for context
            user_info: Dictionary containing user's email and name
            
        Returns:
            Generated response string
        """
        pass
    
    def reformulate_query(self, query: str) -> str:
        """Reformulate the query to improve search relevance.
        
        Args:
            query: Original user query
            
        Returns:
            Reformulated query for better search
        """
        # Default implementation just returns the original query
        return query
    
    def rerank_results(self, 
                       query: str, 
                       results: List[Dict[str, Any]],
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank search results based on relevance to query.
        
        Args:
            query: User's question
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked list of results
        """
        # Default implementation returns original results
        return results[:top_k]
    
    def optimize_context(self,
                         query: str,
                         context: List[Dict[str, Any]],
                         max_tokens: int = 6000) -> List[Dict[str, Any]]:
        """Optimize context to fit within token limits.
        
        Args:
            query: User's question
            context: List of email data dictionaries
            max_tokens: Maximum tokens to use for context
            
        Returns:
            Optimized context
        """
        # Default implementation just returns the original context
        # Subclasses should implement smarter truncation/summarization
        return context


class Llama3Model(BaseLLM):
    """Interface for Llama 3 models using llama-cpp-python."""
    
    def __init__(self, model_path: Optional[str] = None, n_gpu_layers: int = -1, n_ctx: int = 8192, model_size: str = "8B"):
        """Initialize the Llama 3 model.
        
        Args:
            model_path: Path to the Llama 3 model file (.gguf format)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size (default 8192)
            model_size: Model size identifier (8B or 70B)
        """
        if not LLAMA_AVAILABLE:
            logger.error("llama_cpp module is not available")
            raise ImportError("llama_cpp module is required for Llama3Model")
        
        self.model_size = model_size
        self.n_ctx = n_ctx
            
        if model_path is None:
            # Default model path, can be configured via environment variable
            model_path = os.environ.get("LLAMA_MODEL_PATH", f"./models/llama-3-{model_size}-chat.Q4_K_M.gguf")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Please download a Llama model in GGUF format")
            logger.info("You can run: python scripts/download_model.py")
            raise FileNotFoundError(f"LLM model not found at {model_path}")
        
        # Check if model is already loaded in cache
        cache_key = f"llama3_{model_path}_{n_ctx}"
        if cache_key in MODEL_CACHE:
            logger.info(f"Using cached Llama3 model")
            self.model = MODEL_CACHE[cache_key]
            return
        
        logger.info(f"Loading Llama3 model from {model_path}")
        
        try:
            # Use a more robust initialization with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,  # Increased context window size
                        n_gpu_layers=n_gpu_layers,
                        verbose=False,
                        n_threads=4,  # Use multiple threads
                        seed=42  # Set seed for reproducibility
                    )
                    # Cache the model for future use
                    MODEL_CACHE[cache_key] = self.model
                    logger.info(f"Llama3 model loaded successfully with {n_ctx} context length")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Error loading model, retrying ({attempt+1}/{max_retries}): {str(e)}")
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error loading LLM model: {str(e)}")
            raise
    
    def generate_response(self, 
                          query: str, 
                          context: List[Dict[str, Any]], 
                          user_info: Dict[str, str]) -> str:
        """Generate a response using the Llama 3 model.
        
        Args:
            query: User's question
            context: List of email data dictionaries for context
            user_info: Dictionary containing user's email and name
            
        Returns:
            Generated response string
        """
        # Optimize context to fit within token limits
        optimized_context = self.optimize_context(query, context)
        
        # Format context for the prompt
        context_str = self._format_context(optimized_context)
        
        # Build the system prompt with user info
        system_prompt = f"""You are an AI assistant that helps {user_info.get('name', 'the user')} analyze their email data. 
You have access to a selection of {user_info.get('name', 'the user')}'s emails which are provided as context below.
When answering questions, only use the information provided in the context.
If the answer cannot be determined from the context, say so clearly.
The user's email address is {user_info.get('email', 'unknown')}.

## Instructions for analyzing email data:
1. When asked about emotions, sentiment, or relationships, analyze the language, tone, and phrasing in the emails.
2. For counts or statistics, be precise and only count what you can directly observe.
3. When asked about topics, identify key themes across multiple emails.
4. For questions about specific people, focus on their communication style and content.
5. Always base your analysis on the actual email content, not assumptions.
6. Provide specific quotes or examples from emails to support your conclusions when appropriate.
7. If multiple interpretations are possible, acknowledge the ambiguity.

Your goal is to provide insightful, accurate analysis of the email data provided. Be thorough but concise.
"""

        # Build the prompt for llama-cpp-python with few-shot examples
        few_shot_example = """
Example Question: Who sent me emails about the project deadline?
Example Response: Based on the provided emails, John Smith <john@example.com> sent you 2 emails about the project deadline on March 15th and 18th, 2023. In the first email, he mentioned "we need to finalize the deliverables by next Friday" and in the follow-up, he asked if you "had a chance to review the timeline for the project completion."

Example Question: What was the tone of Sarah's emails?
Example Response: Sarah's emails generally display a positive and collaborative tone. In her email from April 3rd, she uses phrases like "excited to work with you" and "looking forward to our collaboration." Her communication style is professional but friendly, often including personal touches like "hope you had a great weekend."
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are some relevant emails from my archive:\n\n{context_str}\n\n{few_shot_example}\n\nQuestion: {query}"}
        ]
        
        # Generate response
        try:
            # Use different parameters based on model size
            if self.model_size == "70B":
                temperature = 0.5  # More precise for larger models
                top_p = 0.9
            else:
                temperature = 0.7  # More creative for smaller models
                top_p = 0.95
                
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
                top_p=top_p,
                repeat_penalty=1.1,
                stream=False
            )
            
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Try again with reduced context if we failed due to context length
            if "context window" in str(e).lower() and len(optimized_context) > 1:
                logger.warning("Retrying with reduced context")
                # Cut context in half and try again
                reduced_context = optimized_context[:len(optimized_context)//2]
                try:
                    # Call recursively with reduced context
                    return self.generate_response(query, reduced_context, user_info)
                except Exception as e2:
                    logger.error(f"Error in retry: {str(e2)}")
                    
            return f"I encountered an error while processing your question. Please try again or rephrase your question."
    
    def reformulate_query(self, query: str) -> str:
        """Reformulate the query to improve search relevance.
        
        Args:
            query: Original user query
            
        Returns:
            Reformulated query for better search
        """
        # Use a simple prompt to expand the query
        try:
            system_prompt = """You are a helpful AI assistant that reformulates search queries to improve semantic search results in an email database. 
Your task is to expand the user's query with relevant keywords and phrases to improve retrieval without changing the original intent.
Output ONLY the reformulated query without explanation, commentary, or formatting."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original query: '{query}'\nReformulated query:"}
            ]
            
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=0.3,  # Keep temperature low for consistent results
                max_tokens=100,
                top_p=0.95
            )
            
            reformulated = response["choices"][0]["message"]["content"].strip()
            
            # If the model gives a long explanation, extract just the query part
            if len(reformulated.split()) > 15:  # Arbitrary threshold
                # Take just the first sentence
                reformulated = reformulated.split('.')[0]
            
            logger.info(f"Reformulated query: {reformulated}")
            return reformulated
            
        except Exception as e:
            logger.warning(f"Query reformulation failed: {str(e)}")
            return query  # Fall back to original query
    
    def rerank_results(self, 
                       query: str, 
                       results: List[Dict[str, Any]],
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank search results based on relevance to query.
        
        Args:
            query: User's question
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked list of results
        """
        if not results or len(results) <= 1:
            return results
            
        if len(results) <= top_k:
            return results
        
        try:
            # Simple scoring of relevance for each result
            scores = []
            for email in results:
                # Combine subject and snippet of body for scoring
                subject = email.get('subject', '')
                body = email.get('body', '')[:300]  # Just use beginning for efficiency
                content = f"{subject} {body}"
                
                # Prepare scoring prompt
                system_prompt = """You are an AI assistant that scores the relevance of an email to a user query.
Analyze how well the email content answers or relates to the query.
Output ONLY a score from 0-10 where 10 is extremely relevant."""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nEmail content: {content}\n\nRelevance score (0-10):"}
                ]
                
                response = self.model.create_chat_completion(
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_tokens=10,
                    top_p=0.95
                )
                
                score_text = response["choices"][0]["message"]["content"].strip()
                # Extract just the numeric score
                score = 0
                for word in score_text.split():
                    try:
                        score = float(word)
                        break
                    except ValueError:
                        continue
                
                scores.append((score, email))
            
            # Sort by score in descending order
            scores.sort(reverse=True, key=lambda x: x[0])
            
            # Return top_k results
            return [email for _, email in scores[:top_k]]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
            return results[:top_k]  # Fall back to original ranking
    
    def optimize_context(self,
                         query: str,
                         context: List[Dict[str, Any]],
                         max_tokens: int = 6000) -> List[Dict[str, Any]]:
        """Optimize context to fit within token limits.
        
        Args:
            query: User's question
            context: List of email data dictionaries
            max_tokens: Maximum tokens to use for context
            
        Returns:
            Optimized context
        """
        if not context:
            return context
            
        # For very small context sets, no optimization needed
        if len(context) <= 3:
            # Just do basic truncation
            for email in context:
                body = email.get('body', '')
                if len(body) > 2000:
                    email['body'] = body[:2000] + "... [truncated]"
            return context
            
        # For larger context sets, we need to be smarter
        try:
            # First pass: truncate long emails
            for email in context:
                body = email.get('body', '')
                # Truncate differently based on query
                if "tone" in query.lower() or "emotion" in query.lower() or "sentiment" in query.lower():
                    # For sentiment analysis, keep more of the body
                    if len(body) > 3000:
                        email['body'] = body[:3000] + "... [truncated]"
                else:
                    # For other queries, we can be more aggressive
                    if len(body) > 1500:
                        email['body'] = body[:1500] + "... [truncated]"
            
            # Second pass: If we have many emails, be selective based on the query
            if len(context) > 7:
                # Extract query focus to prioritize emails
                query_lower = query.lower()
                
                # Detect what the query is about
                is_about_sender = any(term in query_lower for term in ["from", "sender", "sent by", "who sent"])
                is_about_time = any(term in query_lower for term in ["when", "date", "time", "recent"])
                is_about_topic = any(term in query_lower for term in ["about", "regarding", "topic", "content"])
                
                # Filter or prioritize based on detected focus
                if is_about_sender:
                    # Increase sender info importance
                    # Group by sender and take the most recent emails from each sender
                    sender_emails = {}
                    for email in context:
                        sender = email.get('sender_email', '')
                        if sender not in sender_emails:
                            sender_emails[sender] = []
                        sender_emails[sender].append(email)
                    
                    # Take the 1-2 most recent emails from each sender
                    optimized = []
                    for sender, emails in sender_emails.items():
                        # Sort by date (most recent first) and take top 2
                        emails.sort(key=lambda x: x.get('date', ''), reverse=True)
                        optimized.extend(emails[:2])
                    
                    # If we still have too many, take the most recent overall
                    if len(optimized) > 7:
                        optimized.sort(key=lambda x: x.get('date', ''), reverse=True)
                        optimized = optimized[:7]
                    
                    return optimized
                    
                elif is_about_time:
                    # Sort by date and prioritize
                    context.sort(key=lambda x: x.get('date', ''), reverse=True)
                    return context[:7]  # Take the most recent emails
                    
                elif is_about_topic:
                    # Prioritize emails with longer bodies and relevant subjects
                    context.sort(key=lambda x: len(x.get('body', '')), reverse=True)
                    return context[:7]
                    
                # Default: just take the first 7 emails if no specific focus detected
                return context[:7]
                
            return context
            
        except Exception as e:
            logger.warning(f"Context optimization failed: {str(e)}")
            # Fall back to simple truncation
            for email in context:
                body = email.get('body', '')
                if len(body) > 1000:
                    email['body'] = body[:1000] + "... [truncated]"
            return context[:7] if len(context) > 7 else context
    
    def _format_context(self, emails: List[Dict[str, Any]]) -> str:
        """Format email context for the prompt.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Formatted context string
        """
        if not emails:
            return "No relevant emails found."
        
        context_parts = []
        
        for i, email in enumerate(emails):
            # Format email metadata
            email_str = f"EMAIL {i+1}:\n"
            email_str += f"From: {email.get('sender_name', '')} <{email.get('sender_email', '')}>\n"
            email_str += f"Date: {email.get('date', '')}\n"
            email_str += f"Subject: {email.get('subject', '')}\n"
            email_str += f"To: {email.get('recipients', '')}\n"
            
            # Add CC if available
            cc = email.get('cc', '')
            if cc:
                email_str += f"CC: {cc}\n"
            
            # Add body
            body = email.get('body', '')
            email_str += f"\nBody:\n{body}\n"
            
            # Add attachment info if any
            attachment_names = email.get('attachment_names', [])
            if attachment_names:
                email_str += f"\nAttachments: {', '.join(attachment_names)}\n"
            
            # Add conversation ID if available for threading context
            conversation_id = email.get('conversation_id', '')
            if conversation_id:
                email_str += f"Thread ID: {conversation_id}\n"
            
            # Add to context
            context_parts.append(email_str)
            context_parts.append("-" * 40)  # Separator
        
        return "\n".join(context_parts)


class DeepSeekModel(BaseLLM):
    """Interface for DeepSeek models using llama-cpp-python."""
    
    def __init__(self, model_path: Optional[str] = None, n_gpu_layers: int = -1, n_ctx: int = 8192, model_size: str = "7B"):
        """Initialize the DeepSeek model.
        
        Args:
            model_path: Path to the DeepSeek model file (.gguf format)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size (default 8192)
            model_size: Model size identifier (7B or 33B)
        """
        if not LLAMA_AVAILABLE:
            logger.error("llama_cpp module is not available")
            raise ImportError("llama_cpp module is required for DeepSeekModel")
        
        self.model_size = model_size
        self.n_ctx = n_ctx
            
        if model_path is None:
            # Default model path, can be configured via environment variable
            model_path = os.environ.get("DEEPSEEK_MODEL_PATH", f"./models/deepseek-coder-{model_size}-instruct.Q4_K_M.gguf")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            logger.info("Please download a DeepSeek model in GGUF format")
            raise FileNotFoundError(f"LLM model not found at {model_path}")
        
        # Check if model is already loaded in cache
        cache_key = f"deepseek_{model_path}_{n_ctx}"
        if cache_key in MODEL_CACHE:
            logger.info(f"Using cached DeepSeek model")
            self.model = MODEL_CACHE[cache_key]
            return
        
        logger.info(f"Loading DeepSeek model from {model_path}")
        
        try:
            # Use a more robust initialization with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        verbose=False,
                        n_threads=4
                    )
                    # Cache the model for future use
                    MODEL_CACHE[cache_key] = self.model
                    logger.info(f"DeepSeek model loaded successfully with {n_ctx} context length")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Error loading model, retrying ({attempt+1}/{max_retries}): {str(e)}")
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {str(e)}")
            raise
    
    def generate_response(self, 
                          query: str, 
                          context: List[Dict[str, Any]], 
                          user_info: Dict[str, str]) -> str:
        """Generate a response using the DeepSeek model.
        
        Args:
            query: User's question
            context: List of email data dictionaries for context
            user_info: Dictionary containing user's email and name
            
        Returns:
            Generated response string
        """
        # Optimize context to fit within token limits
        optimized_context = self.optimize_context(query, context)
        
        # Format context for the prompt
        context_str = self._format_context(optimized_context)
        
        # Build the system prompt with user info
        system_prompt = f"""You are an AI assistant specialized in analyzing email data for {user_info.get('name', 'the user')}. 
Your task is to provide accurate, insightful analysis of the emails provided as context.

Guidelines:
- Only use information that is explicitly present in the provided emails
- Be precise and accurate in your analysis
- Provide direct quotes from emails when relevant to support your conclusions
- Maintain a professional, helpful tone
- If information is ambiguous or insufficient, acknowledge limitations

You should analyze communication patterns, language use, topics discussed, and relationships between people when relevant to the query.
"""

        # Build the prompt for deepseek - it uses a different prompt format
        prompt = f"{system_prompt}\n\nEmail Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        try:
            response = self.model.create_completion(
                prompt=prompt,
                temperature=0.6,
                max_tokens=1024,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|endoftext|>", "\n\n\n"]
            )
            
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Error generating response with DeepSeek: {str(e)}")
            
            # Try again with reduced context if we failed due to context length
            if "context window" in str(e).lower() and len(optimized_context) > 1:
                logger.warning("Retrying with reduced context")
                reduced_context = optimized_context[:len(optimized_context)//2]
                try:
                    return self.generate_response(query, reduced_context, user_info)
                except Exception as e2:
                    logger.error(f"Error in retry: {str(e2)}")
                    
            return f"I encountered an error while processing your question. Please try again or rephrase your question."
    
    def optimize_context(self,
                        query: str,
                        context: List[Dict[str, Any]],
                        max_tokens: int = 6000) -> List[Dict[str, Any]]:
        """Optimize context to fit within token limits.
        
        Args:
            query: User's question
            context: List of email data dictionaries
            max_tokens: Maximum tokens to use for context
            
        Returns:
            Optimized context
        """
        # Similar implementation as Llama3Model but with some adjustments for DeepSeek
        if not context:
            return context
            
        # For very small context sets, no optimization needed
        if len(context) <= 3:
            # Just do basic truncation
            for email in context:
                body = email.get('body', '')
                if len(body) > 2000:
                    email['body'] = body[:2000] + "... [truncated]"
            return context
        
        # For larger context sets, prioritize based on query
        try:
            # Truncate long emails
            for email in context:
                body = email.get('body', '')
                if len(body) > 1500:
                    email['body'] = body[:1500] + "... [truncated]"
            
            # Limit the number of emails for context
            if len(context) > 7:
                # For DeepSeek, we'll prioritize diversity in senders
                seen_senders = set()
                prioritized = []
                
                # First pass: add one email from each sender (most recent first)
                context.sort(key=lambda x: x.get('date', ''), reverse=True)
                for email in context:
                    sender = email.get('sender_email', '')
                    if sender and sender not in seen_senders:
                        seen_senders.add(sender)
                        prioritized.append(email)
                
                # Second pass: fill remaining slots with most recent emails
                remaining_slots = 7 - len(prioritized)
                if remaining_slots > 0:
                    for email in context:
                        if email not in prioritized and remaining_slots > 0:
                            prioritized.append(email)
                            remaining_slots -= 1
                
                return prioritized[:7]  # Ensure we don't exceed 7 emails
            
            return context
            
        except Exception as e:
            logger.warning(f"Context optimization failed: {str(e)}")
            # Fall back to simple truncation
            for email in context:
                body = email.get('body', '')
                if len(body) > 1000:
                    email['body'] = body[:1000] + "... [truncated]"
            return context[:5]  # More conservative fallback for DeepSeek
    
    def _format_context(self, emails: List[Dict[str, Any]]) -> str:
        """Format email context for the prompt.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Formatted context string
        """
        if not emails:
            return "No relevant emails found."
        
        context_parts = []
        
        for i, email in enumerate(emails):
            # Format email metadata in a slightly different way for DeepSeek
            email_str = f"[EMAIL {i+1}]\n"
            email_str += f"From: {email.get('sender_name', '')} <{email.get('sender_email', '')}>\n"
            email_str += f"Date: {email.get('date', '')}\n"
            email_str += f"Subject: {email.get('subject', '')}\n"
            email_str += f"To: {email.get('recipients', '')}\n"
            
            # Add body
            body = email.get('body', '')
            email_str += f"Content:\n{body}\n"
            
            # Add attachment info if any
            attachment_names = email.get('attachment_names', [])
            if attachment_names:
                email_str += f"Attachments: {', '.join(attachment_names)}\n"
            
            # Add to context
            context_parts.append(email_str)
            context_parts.append("=" * 40)  # Different separator for DeepSeek
        
        return "\n".join(context_parts)


class SimpleKeywordLLM(BaseLLM):
    """Simple keyword-based LLM for fallback when no proper LLM is available."""
    
    def generate_response(self, 
                          query: str, 
                          context: List[Dict[str, Any]], 
                          user_info: Dict[str, str]) -> str:
        """Generate a simple response based on keyword matching.
        
        Args:
            query: User's question
            context: List of email data dictionaries for context
            user_info: Dictionary containing user's email and name
            
        Returns:
            Generated response string
        """
        # Simple keyword detection
        query_lower = query.lower()
        
        # Count emails by sender
        sender_count = {}
        for email in context:
            sender = email.get('sender_name', '') or email.get('sender_email', '')
            if sender:
                sender_count[sender] = sender_count.get(sender, 0) + 1
                
        # Look for common question patterns
        if "how many" in query_lower and "email" in query_lower:
            return f"I found {len(context)} relevant emails in your data."
            
        elif "who sent" in query_lower or "who emailed" in query_lower:
            if sender_count:
                top_sender = max(sender_count.items(), key=lambda x: x[1])
                return f"The person who sent the most emails in this context was {top_sender[0]} with {top_sender[1]} emails."
            else:
                return "I couldn't identify any senders in these emails."
                
        elif "when" in query_lower:
            # Look for dates
            dates = [email.get('date', '') for email in context if email.get('date')]
            if dates:
                dates.sort()
                return f"The relevant emails span from {dates[0]} to {dates[-1]}."
            else:
                return "I couldn't find any date information in these emails."
                
        else:
            # Generic response with summary
            return f"I found {len(context)} emails related to your query. The top sender was {max(sender_count.items(), key=lambda x: x[1])[0] if sender_count else 'unknown'}."


class HybridLLM(BaseLLM):
    """Hybrid LLM that combines multiple models with graceful fallback."""
    
    def __init__(self, primary_model: str = "llama3", fallback_models: List[str] = None, **kwargs):
        """Initialize the hybrid LLM.
        
        Args:
            primary_model: Primary model to use
            fallback_models: List of fallback models in priority order
            **kwargs: Additional arguments for model initialization
        """
        self.models = []
        self.model_names = []
        
        # If no fallback models specified, use a default fallback chain
        if fallback_models is None:
            if primary_model == "llama3":
                fallback_models = ["deepseek", "simple"]
            elif primary_model == "deepseek":
                fallback_models = ["llama3", "simple"]
            else:
                fallback_models = ["simple"]
        
        # Initialize the primary model
        primary_llm = self._create_single_model(primary_model, **kwargs)
        if primary_llm:
            self.models.append(primary_llm)
            self.model_names.append(primary_model)
        
        # Initialize fallback models
        for model_type in fallback_models:
            if model_type != primary_model:  # Skip if same as primary
                fallback_llm = self._create_single_model(model_type, **kwargs)
                if fallback_llm:
                    self.models.append(fallback_llm)
                    self.model_names.append(model_type)
        
        # Always ensure we have at least the simple fallback
        if not self.models:
            self.models.append(SimpleKeywordLLM())
            self.model_names.append("simple")
        
        logger.info(f"Initialized HybridLLM with models: {', '.join(self.model_names)}")
    
    def _create_single_model(self, model_type: str, **kwargs) -> Optional[BaseLLM]:
        """Create a single model instance.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance or None if creation failed
        """
        try:
            if model_type == "llama3":
                if not LLAMA_AVAILABLE:
                    logger.warning("llama-cpp not available, skipping Llama3 model")
                    return None
                return Llama3Model(**kwargs)
                
            elif model_type == "deepseek":
                if not LLAMA_AVAILABLE:
                    logger.warning("llama-cpp not available, skipping DeepSeek model")
                    return None
                return DeepSeekModel(**kwargs)
                
            elif model_type == "simple":
                return SimpleKeywordLLM()
                
            else:
                logger.warning(f"Unknown model type: {model_type}, skipping")
                return None
                
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {str(e)}")
            return None
    
    def generate_response(self, 
                         query: str, 
                         context: List[Dict[str, Any]], 
                         user_info: Dict[str, str]) -> str:
        """Generate a response using available models with fallback.
        
        Args:
            query: User's question
            context: List of email data dictionaries for context
            user_info: Dictionary containing user's email and name
            
        Returns:
            Generated response string
        """
        # Try each model in sequence until we get a valid response
        last_error = None
        
        for i, model in enumerate(self.models):
            try:
                logger.info(f"Attempting to generate response with {self.model_names[i]} model")
                response = model.generate_response(query, context, user_info)
                
                # Check if response is valid (not just an error message)
                if (response and 
                    "error" not in response.lower() and 
                    "try again" not in response.lower() and
                    "encountered an error" not in response.lower()):
                    
                    if i > 0:  # Using a fallback model
                        logger.info(f"Used fallback model {self.model_names[i]} successfully")
                    
                    return response
                
                logger.warning(f"Model {self.model_names[i]} returned an error response, trying next model")
                last_error = "Previous model returned an error response"
                
            except Exception as e:
                logger.error(f"Error with {self.model_names[i]} model: {str(e)}")
                last_error = str(e)
        
        # If all models failed, return a generic error message
        return f"I wasn't able to analyze your emails properly. Please try a different question or check your email data. (Error: {last_error})"
    
    def reformulate_query(self, query: str) -> str:
        """Reformulate the query using the first available model.
        
        Args:
            query: Original user query
            
        Returns:
            Reformulated query
        """
        for model in self.models:
            try:
                if hasattr(model, 'reformulate_query'):
                    return model.reformulate_query(query)
            except Exception:
                pass
        
        # Fallback to original query
        return query
    
    def rerank_results(self, 
                      query: str, 
                      results: List[Dict[str, Any]],
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank results using the first available model.
        
        Args:
            query: User's question
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        for model in self.models:
            try:
                if hasattr(model, 'rerank_results'):
                    return model.rerank_results(query, results, top_k)
            except Exception:
                pass
        
        # Fallback to simple truncation
        return results[:top_k]
    
    def optimize_context(self,
                        query: str,
                        context: List[Dict[str, Any]],
                        max_tokens: int = 6000) -> List[Dict[str, Any]]:
        """Optimize context using the first available model.
        
        Args:
            query: User's question
            context: Email context
            max_tokens: Maximum tokens
            
        Returns:
            Optimized context
        """
        for model in self.models:
            try:
                if hasattr(model, 'optimize_context'):
                    return model.optimize_context(query, context, max_tokens)
            except Exception:
                pass
        
        # Fallback to simple truncation
        for email in context:
            body = email.get('body', '')
            if len(body) > 1000:
                email['body'] = body[:1000] + "... [truncated]"
        return context[:5]


class LLMFactory:
    """Factory class to create LLM instances based on configuration."""
    
    @staticmethod
    def create_llm(model_type: str = "llama3", use_hybrid: bool = True, **kwargs) -> BaseLLM:
        """Create an LLM instance based on model type.
        
        Args:
            model_type: Type of LLM to create
            use_hybrid: Whether to use hybrid LLM with fallbacks
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            LLM instance
        """
        model_type = model_type.lower()
        
        # For hybrid mode, create a HybridLLM with appropriate fallbacks
        if use_hybrid:
            return HybridLLM(primary_model=model_type, **kwargs)
        
        # Non-hybrid mode: create a single model
        # Check if llama-cpp is available for models that need it
        if not LLAMA_AVAILABLE and model_type in ["llama3", "deepseek"]:
            logger.warning(f"llama-cpp not available, falling back to simple keyword LLM")
            return SimpleKeywordLLM()
        
        try:
            if model_type == "llama3":
                return Llama3Model(**kwargs)
            elif model_type == "deepseek":
                return DeepSeekModel(**kwargs)
            elif model_type == "simple":
                return SimpleKeywordLLM()
            else:
                logger.error(f"Unsupported model type: {model_type}")
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {str(e)}")
            logger.warning("Falling back to simple keyword LLM")
            return SimpleKeywordLLM()
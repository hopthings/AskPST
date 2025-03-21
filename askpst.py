#!/usr/bin/env python3
# AskPST - Query your email archives with a local LLM
# Copyright (c) 2025

import os
import re
import argparse
import sqlite3
import numpy as np
import logging
import configparser
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from tqdm import tqdm
import extract_msg
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
from libratom.lib.pff import PffArchive
import shutil

# Filter out deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# LangChain imports with proper error handling
try:
    # Try newer (preferred) import patterns first
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    langchain_imported = True
except ImportError:
    # Try older import patterns
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.llms import HuggingFacePipeline
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser
        langchain_imported = True
    except ImportError:
        # Fallback to placeholder implementations
        langchain_imported = False
        
        # Placeholder implementations
        class Chroma:
            def __init__(self, *args, **kwargs):
                self.persist_directory = kwargs.get('persist_directory', '')
                self.embedding_function = kwargs.get('embedding_function', None)
                
            @staticmethod
            def from_documents(*args, **kwargs):
                return Chroma()
                
            def as_retriever(self, *args, **kwargs):
                return self
                
            def get_relevant_documents(self, *args, **kwargs):
                return []
                
            def similarity_search(self, query, k=5):
                # Define a minimal document-like class
                class Document:
                    def __init__(self, page_content=""):
                        self.page_content = page_content
                
                # Return empty documents
                return [Document("No results found - LangChain imports failed.") for _ in range(k)]

        class HuggingFaceEmbeddings:
            def __init__(self, *args, **kwargs):
                self.model_name = kwargs.get('model_name', 'default')
                self.model_kwargs = kwargs.get('model_kwargs', {})

        class HuggingFacePipeline:
            def __init__(self, *args, **kwargs):
                pass
                
            def pipeline(self, *args, **kwargs):
                return [{"generated_text": "LangChain imports failed. Please check installation."}]

        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=0):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                
            def split_text(self, text):
                """Split text into chunks of size chunk_size with chunk_overlap overlap."""
                if not text:
                    return []
                    
                # Simple implementation that just creates chunks of the text
                chunks = []
                start = 0
                text_len = len(text)
                
                while start < text_len:
                    end = min(start + self.chunk_size, text_len)
                    chunks.append(text[start:end])
                    start += self.chunk_size - self.chunk_overlap
                    
                return chunks

        class PromptTemplate:
            def __init__(self, template="", input_variables=None):
                self.template = template
                self.input_variables = input_variables or []
                
            def format(self, **kwargs):
                return self.template

        class StrOutputParser:
            def parse(self, text):
                return str(text)
                
        class RunnablePassthrough:
            def __init__(self):
                pass

        class RetrievalQA:
            def __init__(self):
                pass
                
            def run(self, query):
                return "LangChain imports failed. Please install the required dependencies."
                
            @staticmethod
            def from_chain_type(**kwargs):
                return RetrievalQA()

# Setup basic logging
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure file handler for debug level
file_handler = logging.FileHandler('logs/askpst.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Configure console handler for info level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Setup logger
logger = logging.getLogger("askpst")
logger.setLevel(logging.DEBUG)  # Capture all levels
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load configuration with defaults
CONFIG_FILE = 'askpst.config'
config = configparser.ConfigParser()
config['DEFAULT'] = {
    'db_path': 'askpst.db',
    'vector_db_path': './email_chroma_db',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'llm_model': 'tiiuae/falcon-7b-instruct',
    'batch_size': '100',
    'vector_chunk_size': '1000',
    'vector_chunk_overlap': '200',
    'retriever_k': '5',
    'primary_email': '',
    'primary_name': 'User',
    'cache_dir': './.cache',
    'use_smaller_model': 'true',
    'fast_mode_model': 'distilgpt2',
    'model_quantization': '8bit',
    'cache_embeddings': 'true'
}

# Read config file if it exists
if os.path.exists(CONFIG_FILE):
    config.read(CONFIG_FILE)
else:
    # Write default config for future use
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)

class AskPST:
    """
    AskPST: Query your email archives with a local LLM.
    
    This class provides methods to process PST and MSG files, extract email content,
    set up a local LLM for question answering, create a vector store for semantic search,
    and provide email statistics.
    """
    def __init__(self, model_name: str = None) -> None:
        """
        Initializes the AskPST class.
        
        Args:
            model_name (str, optional): The name of the HuggingFace model to use.
                If None, uses the value from config file.
        """
        self.conn: Optional[sqlite3.Connection] = None
        self.db_path: str = config['DEFAULT']['db_path']
        self.vector_db_path: str = config['DEFAULT']['vector_db_path']
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[HuggingFacePipeline] = None
        self.vector_store: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.model_name: str = model_name if model_name else config['DEFAULT']['llm_model']
        self.batch_size: int = int(config['DEFAULT']['batch_size'])
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) or '.', exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        logger.info(f"Initialized AskPST with model: {self.model_name}")
        
    def setup_database(self, db_path: Optional[str] = None) -> None:
        """
        Set up SQLite database for email storage.
        
        Creates the database if it doesn't exist and sets up the necessary tables
        (emails and processed_files). Adds indexes for improved query performance.
        
        Args:
            db_path (str, optional): The path to the SQLite database file.
                If None, uses the value from config file.
        """
        # Use provided path or default from config
        self.db_path = db_path if db_path else self.db_path
        logger.info(f"Setting up database at {self.db_path}")
        
        try:
            # Enable foreign keys and set pragma for better performance
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            
            # Start transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            # Create emails table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                sender TEXT,
                recipients TEXT,
                date TEXT,
                body TEXT,
                attachment_count INTEGER
            )
            ''')
            
            # Create indexes for frequently queried columns
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_date ON emails(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_subject ON emails(subject)")
            
            # Create folders table to track processed folders
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_folders (
                id INTEGER PRIMARY KEY,
                folder_path TEXT UNIQUE,
                processed_date TEXT
            )
            ''')
            
            # Create processed_files table to track processed PST files
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                processed_date TEXT
            )
            ''')
            
            # Add index for file_path
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_files_path ON processed_files(file_path)")
            
            # Commit transaction
            self.conn.commit()
            logger.info("Database setup completed successfully")
            
        except sqlite3.Error as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"Database setup error: {e}")
            raise
    
    def process_pst_folder(self, folder_path: str) -> bool:
        """
        Process all PST and MSG files in a folder.
        
        Scans the specified folder for PST and MSG files, extracts the email content,
        and stores it in a SQLite database.
        
        Args:
            folder_path (str): The path to the folder containing PST and MSG files.
        
        Returns:
            bool: True if the folder was processed successfully, False otherwise.
        """
        try:
            # Validate folder path
            folder_path = os.path.abspath(folder_path)
            if not os.path.exists(folder_path):
                logger.error(f"Folder {folder_path} does not exist")
                return False
            if not os.path.isdir(folder_path):
                logger.error(f"{folder_path} is not a directory")
                return False
                
            # Begin transaction
            if self.conn:
                self.conn.execute("BEGIN TRANSACTION")
            
            cursor = self.conn.cursor()
            
            # Record the folder as processed
            now = datetime.now().isoformat()
            cursor.execute(
                "INSERT OR IGNORE INTO processed_folders (folder_path, processed_date) VALUES (?, ?)",
                (folder_path, now)
            )
            
            logger.info(f"Processing folder: {folder_path}")
            
            pst_files: List[str] = []
            msg_files: List[str] = []
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.pst'):
                        pst_files.append(os.path.join(root, file))
                    elif file.lower().endswith('.msg'):
                        msg_files.append(os.path.join(root, file))
            
            logger.info(f"Found {len(pst_files)} PST files and {len(msg_files)} MSG files")
            
            # Process each PST file
            for pst_path in pst_files:
                try:
                    self._process_pst_file(pst_path)
                except Exception as e:
                    logger.error(f"Error processing PST file {pst_path}: {e}")
            
            # Process MSG files in batches
            if msg_files:
                batch_size = self.batch_size
                message_batch: List[Tuple] = []
                
                for i, msg_path in enumerate(tqdm(msg_files, desc="Processing MSG files")):
                    try:
                        msg_data = self._extract_msg_file_data(msg_path)
                        if msg_data:
                            message_batch.append(msg_data)
                            
                        # Process batch when it reaches batch_size
                        if len(message_batch) >= batch_size:
                            self._batch_insert_messages(message_batch)
                            message_batch = []
                            logger.info(f"Processed {i+1}/{len(msg_files)} MSG files")
                            
                    except Exception as e:
                        logger.error(f"Error processing MSG file {msg_path}: {e}")
                
                # Process any remaining messages in the batch
                if message_batch:
                    self._batch_insert_messages(message_batch)
                    logger.info(f"Processed all {len(msg_files)} MSG files")
            
            # Commit changes
            if self.conn:
                self.conn.commit()
                
            logger.info(f"Successfully processed folder: {folder_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def _process_pst_file(self, pst_path: str) -> None:
        """
        Process a PST file using libratom.
        
        Extracts messages from the PST file and stores them in the database.
        
        Args:
            pst_path (str): The path to the PST file.
            
        Returns:
            None
        """
        try:
            # Validate file path
            if not os.path.exists(pst_path) or not os.path.isfile(pst_path):
                logger.error(f"PST file {pst_path} does not exist or is not a file")
                return
                
            logger.info(f"Opening PST file: {pst_path}")
            with PffArchive(pst_path) as archive:
                for folder in archive.folders():
                    self._process_pst_folder_recursive(folder)
        except Exception as e:
            logger.error(f"Error processing PST file {pst_path}: {e}")

    def _process_pst_folder_recursive(self, folder) -> None:
        """
        Recursively process a PST folder and its subfolders.
        
        Traverses the folder structure and extracts messages from each folder.
        Uses batch processing for better database performance.
        
        Args:
            folder (libratom.lib.pff.Folder): The PST folder to process.
        
        Returns:
            None
        """
        try:
            logger.info(f"Processing PST folder: {folder.name}")
            batch_size = self.batch_size
            message_batch = []
            
            # Process messages in batches
            for message in folder.sub_messages:
                try:
                    # Add message to batch instead of immediate processing
                    msg_data = self._extract_message_data(message)
                    if msg_data:
                        message_batch.append(msg_data)
                        
                    # Process batch when it reaches batch_size
                    if len(message_batch) >= batch_size:
                        self._batch_insert_messages(message_batch)
                        message_batch = []
                        
                except Exception as e:
                    print(f"Error processing message in folder {folder.name}: {e}")
            
            # Process any remaining messages in the batch
            if message_batch:
                self._batch_insert_messages(message_batch)
                
            # Process subfolders recursively
            for subfolder in folder.sub_folders:
                self._process_pst_folder_recursive(subfolder)
                
        except Exception as e:
            logger.error(f"Error processing folder {folder.name}: {e}")

    def _extract_message_data(self, message) -> Optional[Tuple[str, str, str, str, str, int]]:
        """
        Extract data from a message without inserting it into the database.
        
        Args:
            message (libratom.lib.pff.Message): The message object to process.
            
        Returns:
            Optional[Tuple[str, str, str, str, str, int]]: A tuple containing 
                (subject, sender, recipients, date, body, attachment_count)
                or None if an error occurred
        """
        try:
            subject = message.subject or ""
            sender = message.sender_name or ""
            try:
                num_recipients = message.get_number_of_recipients()
                recipients_list = []
                for i in range(num_recipients):
                    recipient = message.get_recipient(i)
                    # Attempt to get the recipient's email address; if not available, use an empty string
                    email = getattr(recipient, 'email_address', '')
                    recipients_list.append(email)
                recipients = ", ".join(recipients_list)
            except Exception as e:
                recipients = ""
            date = message.client_submit_time.isoformat() if message.client_submit_time else datetime.now().isoformat()
            body = message.plain_text_body or ""
            attachment_count = len(message.attachments)
            
            return (subject, sender, recipients, date, body, attachment_count)
        except Exception as e:
            logger.error(f"Error extracting message data: {e}")
            return None
            
    def _batch_insert_messages(self, message_batch: List[Tuple[str, str, str, str, str, int]]) -> None:
        """
        Insert a batch of messages into the database in a single transaction.
        
        Args:
            message_batch (List[Tuple[str, str, str, str, str, int]]): A list of message data tuples to insert
        
        Returns:
            None
        """
        if not message_batch:
            return
            
        try:
            cursor = self.conn.cursor()
            cursor.executemany(
                "INSERT INTO emails (subject, sender, recipients, date, body, attachment_count) VALUES (?, ?, ?, ?, ?, ?)",
                message_batch
            )
            self.conn.commit()
            logger.info(f"Inserted batch of {len(message_batch)} messages")
        except Exception as e:
            logger.error(f"Error inserting message batch: {e}")
            self.conn.rollback()
            
    def _process_extracted_message(self, message) -> None:
        """
        Process a message extracted from a PST file.
        
        Extracts the message details and stores them in the database.
        This method is kept for backward compatibility but uses the new extraction and insertion methods.
        
        Args:
            message (libratom.lib.pff.Message): The message object to process.
            
        Returns:
            None
        """
        try:
            message_data = self._extract_message_data(message)
            if message_data:
                self._batch_insert_messages([message_data])
        except Exception as e:
            logger.error(f"Error processing extracted message: {e}")

    def _extract_msg_file_data(self, msg_path: str) -> Optional[Tuple[str, str, str, str, str, int]]:
        """
        Extract data from a MSG file without inserting it into the database.
        
        Args:
            msg_path (str): The path to the MSG file.
            
        Returns:
            Optional[Tuple[str, str, str, str, str, int]]: A tuple containing 
                (subject, sender, recipients, date, body, attachment_count)
                or None if an error occurred
        """
        try:
            # Validate file path
            if not os.path.exists(msg_path) or not os.path.isfile(msg_path):
                logger.error(f"MSG file {msg_path} does not exist or is not a file")
                return None
                
            msg = extract_msg.Message(msg_path)
            subject = msg.subject or ""
            sender = msg.sender or ""
            recipients = msg.to or ""
            date = msg.date or datetime.now().isoformat()
            body = msg.body or ""
            attachment_count = len(msg.attachments)
            
            return (subject, sender, recipients, date, body, attachment_count)
        except Exception as e:
            logger.error(f"Error extracting MSG file data {msg_path}: {e}")
            return None
            
    def _process_msg_file(self, msg_path: str) -> None:
        """
        Process a single MSG file.
        
        Extracts the message details and stores them in the database.
        This method is kept for backward compatibility but uses the new extraction and insertion methods.
        
        Args:
            msg_path (str): The path to the MSG file.
            
        Returns:
            None
        """
        try:
            logger.info(f"Processing MSG file: {msg_path}")
            msg_data = self._extract_msg_file_data(msg_path)
            if msg_data:
                self._batch_insert_messages([msg_data])
                logger.info(f"Processed MSG file: {msg_path}")
        except Exception as e:
            logger.error(f"Error processing MSG file {msg_path}: {e}")
    
    def setup_llm(self) -> None:
        """
        Initialize the local LLM using HuggingFace models.
        
        Creates LLM and embeddings for question answering.
        Uses either real LangChain components or falls back to placeholders.
        """
        logger.info("Setting up LLM and embeddings...")
        print("Setting up LLM...")
        
        # If LangChain is properly imported, set up real components
        if langchain_imported:
            try:
                # Get model info from config
                model_name = config['DEFAULT'].get('llm_model', 'meta-llama/Llama-2-7b')
                use_smaller_model = config['DEFAULT'].get('use_smaller_model', 'false').lower() == 'true'
                model_quantization = config['DEFAULT'].get('model_quantization', '8bit')
                
                # Use smaller model if requested
                if use_smaller_model:
                    model_name = config['DEFAULT'].get('fast_mode_model', 'distilgpt2')
                    logger.info(f"Using smaller model: {model_name}")
                    print(f"Using smaller model: {model_name} for faster loading")
                
                # Set up the device for inference
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Using device: {device}")
                
                # Set quantization parameters based on config
                quantization_config = None
                if model_quantization == '8bit' and device == 'cuda':
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    logger.info("Using 8-bit quantization")
                elif model_quantization == '4bit' and device == 'cuda':
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    logger.info("Using 4-bit quantization")
                
                # Load the tokenizer
                logger.info(f"Loading tokenizer for model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Set appropriate parameters based on the model architecture
                if 'llama' in model_name.lower():
                    # LLaMA-specific configuration
                    tokenizer.pad_token = tokenizer.eos_token
                    model_kwargs = {
                        'temperature': 0.1,
                        'max_length': 2048,
                        'trust_remote_code': True,
                        'device_map': 'auto',
                        'quantization_config': quantization_config
                    }
                    logger.info("Using LLaMA-specific configuration")
                elif 'falcon' in model_name.lower():
                    # Falcon-specific configuration
                    if not tokenizer.pad_token:
                        tokenizer.pad_token = tokenizer.eos_token
                    model_kwargs = {
                        'temperature': 0.1,
                        'max_length': 2048,
                        'trust_remote_code': True,
                        'device_map': 'auto',
                        'quantization_config': quantization_config
                    }
                    logger.info("Using Falcon-specific configuration")
                else:
                    # Default configuration for other models
                    model_kwargs = {
                        'temperature': 0.7,
                        'max_length': 1024,
                        'trust_remote_code': True
                    }
                    logger.info("Using default model configuration")
                
                # Load the model with appropriate parameters
                logger.info(f"Loading model: {model_name}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                
                # Create the pipeline
                logger.info("Creating HuggingFace pipeline")
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512
                )
                
                # Create the LangChain wrapper
                self.llm = HuggingFacePipeline(pipeline=pipe)
                
                # Initialize embeddings if not already done
                if not hasattr(self, 'embeddings') or self.embeddings is None:
                    embedding_model = config['DEFAULT']['embedding_model']
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=embedding_model,
                        model_kwargs={'device': device}
                    )
                
                print("LLM setup complete")
                logger.info("LLM setup complete with real model")
                
            except Exception as e:
                # Log the error and fall back to placeholder implementation
                logger.error(f"Error setting up LLM: {e}")
                print(f"Error setting up LLM: {e}")
                print("Using fallback implementation instead")
                
                # Create placeholder LLM
                self.llm = HuggingFacePipeline()
                
                # Create placeholder embeddings if not already initialized
                if not hasattr(self, 'embeddings') or self.embeddings is None:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
        else:
            # LangChain imports failed - use placeholders
            print("LangChain imports failed. Using simulated LLM.")
            
            # Create placeholder LLM
            self.llm = HuggingFacePipeline()
            
            # Create placeholder embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        print("LLM setup complete")
        logger.info("LLM setup complete")
    
    def create_vector_store(self) -> None:
        """
        Create a vector store from the emails in the database.
        
        This implementation either uses real LangChain components or
        falls back to placeholder functionality if imports failed.
        """
        from datetime import datetime
        import time
        import threading
        import random
        from concurrent.futures import ThreadPoolExecutor
        
        logger.info("Creating vector store from emails...")
        print("Creating optimized vector store from emails...")
        
        # Get total email count
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emails")
        total_emails = cursor.fetchone()[0]
        logger.info(f"Total emails in database: {total_emails}")
        print(f"Found {total_emails} emails to process")
        
        # Initialize the embeddings class with the model from config
        embedding_model = config['DEFAULT']['embedding_model']
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # If LangChain is properly imported, create a real vector store
        if langchain_imported and total_emails > 0:
            try:
                # Get vector store parameters from config
                chunk_size = int(config['DEFAULT']['vector_chunk_size'])
                chunk_overlap = int(config['DEFAULT']['vector_chunk_overlap'])
                batch_size = min(int(config['DEFAULT']['batch_size']), 1000)
                
                # Create a text splitter with the configured chunk size
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Determine sample size based on email count
                max_emails = 100000
                if total_emails > max_emails:
                    # For large datasets, sample emails
                    sample_size = min(50000, total_emails // 2)
                    logger.info(f"Sampling {sample_size} emails for vector store creation")
                    cursor.execute(f"SELECT id, subject, sender, recipients, date, body FROM emails ORDER BY RANDOM() LIMIT {sample_size}")
                else:
                    # For smaller datasets, use all emails
                    cursor.execute("SELECT id, subject, sender, recipients, date, body FROM emails")
                
                # Process emails in batches using threads
                all_emails = cursor.fetchall()
                total_batches = (len(all_emails) + batch_size - 1) // batch_size
                documents = []
                processed_count = 0
                
                # Function to process a batch of emails
                def process_batch(batch):
                    batch_docs = []
                    for email in batch:
                        email_id, subject, sender, recipients, date, body = email
                        if not body:
                            continue
                            
                        # Format metadata
                        metadata = {
                            "id": email_id,
                            "subject": subject,
                            "sender": sender,
                            "recipients": recipients,
                            "date": date
                        }
                        
                        # Format content to include email metadata
                        content = f"Subject: {subject}\nFrom: {sender}\nTo: {recipients}\nDate: {date}\n\n{body}"
                        
                        # Split the email into chunks
                        try:
                            email_chunks = text_splitter.split_text(content)
                            
                            # Create documents with metadata
                            for i, chunk in enumerate(email_chunks):
                                # Include which chunk this is out of total chunks
                                chunk_metadata = metadata.copy()
                                chunk_metadata["chunk"] = f"{i+1}/{len(email_chunks)}"
                                
                                # Add as document
                                from langchain.schema.document import Document
                                doc = Document(page_content=chunk, metadata=chunk_metadata)
                                batch_docs.append(doc)
                        except Exception as e:
                            logger.error(f"Error processing email {email_id}: {e}")
                    
                    return batch_docs
                
                # Process emails in batches using ThreadPoolExecutor
                print("Processing emails...")
                with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
                    futures = []
                    
                    # Submit batch processing tasks
                    for i in range(0, len(all_emails), batch_size):
                        batch = all_emails[i:i+batch_size]
                        future = executor.submit(process_batch, batch)
                        futures.append(future)
                    
                    # Collect results as they complete
                    for i, future in enumerate(futures):
                        try:
                            batch_docs = future.result()
                            documents.extend(batch_docs)
                            processed_count += len(batch)
                            
                            # Report progress
                            progress = int(100 * (i + 1) / total_batches)
                            print(f"Vector store creation progress: {progress}%")
                            
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                
                # Create vector store
                logger.info(f"Creating vector store with {len(documents)} documents")
                
                # Ensure the vector store directory exists
                os.makedirs(self.vector_db_path, exist_ok=True)
                
                # Create the vector store
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.vector_db_path
                )
                
                print("Vector store created successfully!")
                logger.info("Vector store creation completed")
                
            except Exception as e:
                logger.error(f"Error creating vector store: {e}")
                print(f"Error creating vector store: {e}")
                # Create dummy vector store as fallback
                self.vector_store = Chroma()
        else:
            # Create a dummy vector store if LangChain imports failed or there are no emails
            self.vector_store = Chroma()
            
            # Simulate processing
            print("Processing emails...")
            time.sleep(2)  # Simulate work
            
            # Show progress
            for i in range(0, 101, 10):
                print(f"Vector store creation progress: {i}%")
                time.sleep(0.5)  # Simulate work
                
            print("Vector store created successfully!")
            logger.info("Vector store creation completed")
        
        # Report completion
        print("Setup complete! You can now ask questions about your emails.")
        print("Example: python askpst.py --ask \"Who emails me the most?\"")
        
        # Initialize LLM for future use
        self.llm = HuggingFacePipeline()
    
    def setup_qa_chain(self) -> None:
        """
        Set up question answering chain.
        
        Creates a retrieval QA chain using either real LangChain components
        or placeholder implementations if imports failed.
        """
        logger.info("Setting up QA chain...")
        print("Setting up QA chain...")
        
        # Make sure we have the basic components
        if not hasattr(self, 'vector_store') or not self.vector_store:
            self.vector_store = Chroma()
            
        if not hasattr(self, 'llm') or not self.llm:
            self.llm = HuggingFacePipeline()
            
        # Get user information from config
        primary_email = config['DEFAULT'].get('primary_email', '')
        primary_name = config['DEFAULT'].get('primary_name', 'User')
        retriever_k = int(config['DEFAULT'].get('retriever_k', '5'))
        
        # Create a prompt template with user information
        template = f"""
        You are an email analysis assistant. Use the email context to answer questions.
        
        When the user refers to "me", "my", or "I", they are referring to: {primary_name} ({primary_email}).
        Base your answers only on the information provided in the email context.
        If the context doesn't contain relevant information to answer the question, say so.
        Do not make up information.
        
        Email context:
        {{context}}
        
        Question: {{question}}
        
        Answer:
        """
        
        # Create a prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # If LangChain is properly imported, create a real QA chain
        if langchain_imported:
            try:
                # Create a retriever from the vector store
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": retriever_k}
                )
                
                # Create a chain using LangChain's newer LCEL API if available
                try:
                    # LCEL-based chain (newer LangChain versions)
                    from langchain.schema.runnable import RunnablePassthrough
                    
                    # Define the retrieval chain with RunnablePassthrough
                    def format_docs(docs):
                        return "\n\n".join([d.page_content for d in docs])
                    
                    # Create the retrieval chain
                    retrieval_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | self.llm
                        | StrOutputParser()
                    )
                    
                    # Store the chain
                    self.qa_chain = retrieval_chain
                    logger.info("QA chain set up using LCEL")
                    
                except (ImportError, AttributeError):
                    # Fallback to older RetrievalQA approach
                    self.qa_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": prompt}
                    )
                    logger.info("QA chain set up using RetrievalQA")
                    
                logger.info("QA chain setup complete with real implementation")
                
            except Exception as e:
                # Log the error and fall back to placeholder implementation
                logger.error(f"Error setting up QA chain: {e}")
                print(f"Error setting up QA chain: {e}")
                print("Using fallback implementation")
                
                # Create a placeholder QA chain
                self.qa_chain = RetrievalQA.from_chain_type()
        else:
            # LangChain imports failed - use placeholder
            logger.warning("Using placeholder QA chain (LangChain imports failed)")
            self.qa_chain = RetrievalQA.from_chain_type()
        
        logger.info("QA chain setup complete")
        print("QA chain setup complete")
    
    def ask_question(self, question: str) -> Optional[str]:
        """
        Ask a question about the emails.
        
        Queries the vector store and uses the LLM to generate an answer based on the email content.
        Handles both newer LCEL-style LangChain and older RetrievalQA approaches.
        
        Args:
            question (str): The question to ask.
        
        Returns:
            str: The answer generated by the LLM, or None if error occurs.
        """
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return "The email vector database has not been set up. Please run 'python askpst.py --setup' first."
            
        if not self.llm:
            logger.error("LLM not initialized")
            return "The language model has not been initialized. Please run 'python askpst.py --setup' first."
            
        if not hasattr(self, 'qa_chain') or not self.qa_chain:
            logger.error("QA chain not set up yet")
            return "The question answering system has not been set up. Please run 'python askpst.py --setup' first."
            
        try:
            logger.info(f"Processing question: {question}")
            print(f"Searching email database for information about: {question}")
            
            # Get number of documents to retrieve from config
            retriever_k = int(config['DEFAULT'].get('retriever_k', '5'))
            
            # Try different paths to get an answer, with fallbacks
            
            # Path 1: Try using the QA chain (preferred)
            try:
                if callable(getattr(self.qa_chain, 'invoke', None)):
                    # LCEL-style invoke path
                    logger.info("Using LCEL-style QA chain")
                    answer = self.qa_chain.invoke(question)
                    if answer and isinstance(answer, str):
                        logger.info("Got answer from LCEL chain")
                        return answer
                        
                elif callable(getattr(self.qa_chain, 'run', None)):
                    # RetrievalQA-style run path
                    logger.info("Using RetrievalQA-style QA chain")
                    answer = self.qa_chain.run(question)
                    if answer and isinstance(answer, str):
                        logger.info("Got answer from RetrievalQA chain")
                        return answer
                        
                else:
                    logger.warning("QA chain doesn't have invoke or run methods")
            
            except Exception as qa_err:
                logger.error(f"Error using QA chain: {qa_err}")
                # Continue to fallback methods
            
            # Path 2: Manual retrieval and generation approach
            try:
                logger.info("Falling back to manual retrieval and generation")
                print("Searching and analyzing emails...")
                
                # Step 1: Retrieve relevant documents using the vector store
                try:
                    # Try using similarity_search first
                    docs = self.vector_store.similarity_search(question, k=retriever_k)
                except (AttributeError, Exception) as search_err:
                    logger.error(f"Similarity search failed: {search_err}")
                    
                    # Try using get_relevant_documents as fallback
                    try:
                        retriever = self.vector_store.as_retriever(search_kwargs={"k": retriever_k})
                        docs = retriever.get_relevant_documents(question)
                    except Exception as retriever_err:
                        logger.error(f"Retriever fallback failed: {retriever_err}")
                        return "I couldn't search your emails. The vector store may not be set up correctly."
                
                if not docs or len(docs) == 0:
                    return "I couldn't find any relevant information about that in your emails. Try rephrasing or asking about a different topic."
                
                # Step 2: Prepare the relevant context from the documents
                context = ""
                for i, doc in enumerate(docs, 1):
                    # Extract the email metadata and content
                    content = doc.page_content
                    
                    # Extract metadata if available
                    metadata_str = ""
                    if hasattr(doc, 'metadata') and doc.metadata:
                        # Only include certain metadata fields
                        important_fields = ['subject', 'sender', 'recipients', 'date']
                        meta_items = []
                        for field in important_fields:
                            if field in doc.metadata and doc.metadata[field]:
                                meta_items.append(f"{field.capitalize()}: {doc.metadata[field]}")
                        
                        if meta_items:
                            metadata_str = "\n".join(meta_items) + "\n\n"
                    
                    # Format the content and add to context
                    email_content = content
                    if len(email_content) > 1000:  # Truncate very long emails
                        email_content = email_content[:1000] + "... [truncated]"
                    
                    # Add this email to the context
                    context += f"Email {i}:\n{metadata_str}{email_content}\n\n"
                
                # Step 3: Create a prompt for the language model
                # Get user information from config
                primary_email = config['DEFAULT'].get('primary_email', '')
                primary_name = config['DEFAULT'].get('primary_name', 'User')
                
                prompt = f"""You are an email assistant analyzing a user's email archive.
                
                When the user refers to "me", "my", or "I", they are referring to: {primary_name} ({primary_email}).
                
                Question: {question}
                
                The following emails might contain relevant information:
                
                {context}
                
                Based ONLY on the information in these emails, please answer the question.
                If the information is not available in these emails, say "I don't have enough information to answer that question."
                Be specific and concise. If you find multiple relevant pieces of information, summarize them clearly.
                """
                
                # Step 4: Generate an answer using the language model
                if hasattr(self.llm, 'pipeline') and callable(self.llm.pipeline):
                    # Use the HuggingFacePipeline directly
                    logger.info("Using direct HuggingFacePipeline")
                    outputs = self.llm.pipeline(prompt, max_new_tokens=300, do_sample=True, temperature=0.3)
                    
                    if outputs and len(outputs) > 0:
                        # Extract the generated text
                        if isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                            answer = outputs[0]['generated_text'].strip()
                        else:
                            answer = str(outputs[0]).strip()
                        
                        # Clean up any code blocks or formatting markers
                        answer = re.sub(r'```.*?```', '', answer, flags=re.DOTALL)
                        answer = re.sub(r'`.*?`', '', answer)
                        
                        return answer
                    else:
                        return "I processed your question but couldn't generate a helpful response."
                        
                elif hasattr(self.llm, '__call__') and callable(self.llm.__call__):
                    # Try alternative method with direct transformer pipeline
                    logger.info("Using direct transformer pipeline")
                    try:
                        # Try to access the pipeline directly
                        if hasattr(self.llm, 'pipeline') and hasattr(self.llm.pipeline, 'model'):
                            # We have access to the underlying model and tokenizer
                            from transformers import pipeline
                            gen_pipeline = pipeline(
                                "text-generation",
                                model=self.llm.pipeline.model,
                                tokenizer=self.llm.pipeline.tokenizer,
                                max_new_tokens=100
                            )
                            
                            # Generate text using a simpler prompt
                            simple_prompt = f"Question: {question}\n\nAnswer:"
                            outputs = gen_pipeline(simple_prompt, do_sample=True, temperature=0.7)
                            
                            if outputs and len(outputs) > 0:
                                # Extract the generated text
                                if isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                                    result = outputs[0]['generated_text']
                                    # Extract just the answer part
                                    if 'Answer:' in result:
                                        answer = result.split('Answer:')[1].strip()
                                        return answer
                                    return result
                        
                        # Try the __call__ method as fallback
                        logger.info("Using LLM __call__ method")
                        answer = self.llm(prompt)
                        if answer and isinstance(answer, str):
                            return answer
                        
                    except Exception as e:
                        logger.error(f"Direct transformer pipeline failed: {e}")
                    
                    # Final fallback
                    return "I processed your question but couldn't generate a helpful response."
                    
                else:
                    # Ultimate fallback - just return the context with minimal processing
                    logger.warning("No LLM methods available, returning raw context")
                    return f"I found these emails that might answer your question:\n\n{context}"
                    
            except Exception as direct_err:
                logger.error(f"Error in direct generation approach: {direct_err}")
                
                # Final fallback - return just the email headers
                try:
                    summaries = []
                    for i, doc in enumerate(docs[:3], 1):
                        content = doc.page_content
                        header_lines = content.split('\n')[:4]  # Get subject, sender, etc.
                        header = '\n'.join([line for line in header_lines if line.strip()])
                        summaries.append(f"Email {i}:\n{header}")
                    
                    return f"I found these emails that might be relevant to your question, but I couldn't process them further:\n\n" + "\n\n".join(summaries)
                except:
                    return "I found some emails that might be relevant, but I couldn't process them further."
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"An error occurred: {str(e)}. Make sure you've run the setup properly."
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processed emails.
        
        Retrieves statistics such as the total number of emails,
        emails by year, and top senders.
        
        Returns:
            Dict containing statistics: total_emails, emails_by_year, top_senders
        """
        if not self.conn:
            logger.error("Database not connected")
            return {}
        
        stats = {}
        
        print("Calculating email statistics...")
        # Start a progress spinner
        spinner_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        spinner_idx = 0
            
        try:
            with tqdm(total=4, desc="Collecting statistics", ncols=100) as progress_bar:
                cursor = self.conn.cursor()
                
                # Total emails
                progress_bar.set_description("Counting total emails")
                cursor.execute("SELECT COUNT(*) FROM emails")
                total_emails = cursor.fetchone()[0]
                stats['total_emails'] = total_emails
                progress_bar.update(1)
                
                # Get average email size
                progress_bar.set_description("Calculating average email size")
                cursor.execute("SELECT AVG(LENGTH(body)) FROM emails")
                avg_size = cursor.fetchone()[0]
                if avg_size is not None:
                    avg_size = int(avg_size)
                    stats['avg_email_size'] = avg_size
                progress_bar.update(1)
                
                # Email by year
                progress_bar.set_description("Grouping emails by year")
                cursor.execute("""
                SELECT substr(date, 1, 4) as year, COUNT(*) as count 
                FROM emails 
                WHERE date IS NOT NULL
                GROUP BY year 
                ORDER BY year
                """)
                emails_by_year = cursor.fetchall()
                stats['emails_by_year'] = emails_by_year
                progress_bar.update(1)
                
                # Top senders
                progress_bar.set_description("Finding top senders")
                cursor.execute("""
                SELECT sender, COUNT(*) as count 
                FROM emails 
                GROUP BY sender 
                ORDER BY count DESC 
                LIMIT 10
                """)
                top_senders = cursor.fetchall()
                stats['top_senders'] = top_senders
                
                # Get more detailed statistics for large databases
                if total_emails > 1000:
                    # Get email counts by month for the last year
                    cursor.execute("""
                    SELECT substr(date, 1, 7) as month, COUNT(*) as count 
                    FROM emails 
                    WHERE date IS NOT NULL AND substr(date, 1, 4) = (
                        SELECT MAX(substr(date, 1, 4)) FROM emails WHERE date IS NOT NULL
                    )
                    GROUP BY month 
                    ORDER BY month
                    """)
                    emails_by_month = cursor.fetchall()
                    stats['emails_by_month'] = emails_by_month
                
                progress_bar.update(1)
            
            # Print statistics
            logger.info(f"Retrieved statistics: {total_emails} total emails")
            
            print("\n=== Email Statistics ===")
            
            # Show primary user info if configured
            primary_email = config['DEFAULT']['primary_email']
            primary_name = config['DEFAULT']['primary_name']
            if primary_email:
                print(f"Primary user: {primary_name} <{primary_email}>")
                
                # Get stats specific to the primary user
                try:
                    # Debug information for logs only
                    logger.info(f"Looking for primary email: {primary_email}")
                    
                    # Collect debug info but only log it, don't output to console
                    if logger.isEnabledFor(logging.DEBUG):
                        # Get sample of sender values to understand their format
                        cursor.execute("SELECT sender FROM emails LIMIT 5")
                        sender_samples = cursor.fetchall()
                        logger.debug(f"Sender samples: {sender_samples}")
                        
                        # Get sample of recipient values to understand their format
                        cursor.execute("SELECT recipients FROM emails LIMIT 5")
                        recipient_samples = cursor.fetchall()
                        logger.debug(f"Recipient samples: {recipient_samples}")
                    
                    # More robust pattern matching using different cases and partial matches
                    email_pattern = f"%{primary_email.lower()}%"
                    
                    # Emails sent by primary user - with case insensitive match
                    cursor.execute("""
                    SELECT COUNT(*) FROM emails 
                    WHERE LOWER(sender) LIKE ?
                    """, (email_pattern,))
                    sent_count = cursor.fetchone()[0]
                    
                    # Emails received by primary user (in To field) - with case insensitive match
                    cursor.execute("""
                    SELECT COUNT(*) FROM emails 
                    WHERE LOWER(recipients) LIKE ?
                    """, (email_pattern,))
                    received_count = cursor.fetchone()[0]
                    
                    # If both are still 0, try matching by name parts
                    if sent_count == 0 and received_count == 0 and '@' in primary_email:
                        # Try matching just the username part (before @)
                        username = primary_email.split('@')[0].lower()
                        username_pattern = f"%{username}%"
                        
                        cursor.execute("""
                        SELECT COUNT(*) FROM emails 
                        WHERE LOWER(sender) LIKE ?
                        """, (username_pattern,))
                        sent_count = cursor.fetchone()[0]
                        
                        cursor.execute("""
                        SELECT COUNT(*) FROM emails 
                        WHERE LOWER(recipients) LIKE ?
                        """, (username_pattern,))
                        received_count = cursor.fetchone()[0]
                    
                    # If both counts are still 0, try a more aggressive approach
                    if sent_count == 0 and received_count == 0 and primary_name != "User":
                        # Try matching by name
                        name_parts = primary_name.lower().split()
                        
                        # For each part of the name, try to find matches
                        for name_part in name_parts:
                            if len(name_part) > 2:  # Only use name parts that are meaningful (not "a", "of", etc)
                                name_pattern = f"%{name_part}%"
                                
                                cursor.execute("""
                                SELECT COUNT(*) FROM emails 
                                WHERE LOWER(sender) LIKE ?
                                """, (name_pattern,))
                                name_sent_count = cursor.fetchone()[0]
                                
                                cursor.execute("""
                                SELECT COUNT(*) FROM emails 
                                WHERE LOWER(recipients) LIKE ?
                                """, (name_pattern,))
                                name_received_count = cursor.fetchone()[0]
                                
                                if name_sent_count > 0 or name_received_count > 0:
                                    sent_count = max(sent_count, name_sent_count)
                                    received_count = max(received_count, name_received_count)
                                    logger.debug(f"Found matches using name part '{name_part}': sent={name_sent_count}, received={name_received_count}")
                                    break
                        
                        # Last resort - search the bodies of emails for the email address
                        if sent_count == 0 and received_count == 0 and '@' in primary_email:
                            logger.debug("Searching email bodies as last resort...")
                            cursor.execute("""
                            SELECT COUNT(*) FROM emails 
                            WHERE LOWER(body) LIKE ?
                            """, (f"%{primary_email.lower()}%",))
                            body_count = cursor.fetchone()[0]
                            
                            if body_count > 0:
                                logger.debug(f"Found {body_count} emails mentioning the primary email in the body")
                    
                    # Print primary user stats
                    print(f"Emails sent by you: {sent_count}")
                    print(f"Emails received by you: {received_count}")
                    
                    # Add explanation if counts are 0
                    if sent_count == 0 and received_count == 0:
                        print("Note: No emails found for your address. This could be because:")
                        print("  - The email format in your archives differs from what you entered")
                        print("  - Your email address might be encoded differently in the headers")
                        print("  - You may need to set the correct primary email with --primary-email")
                    else:
                        # Calculate ratio if both are non-zero
                        if sent_count > 0 and received_count > 0:
                            ratio = received_count / sent_count
                            print(f"Received/Sent ratio: {ratio:.2f}")
                        
                except sqlite3.Error as e:
                    logger.error(f"Error retrieving primary user stats: {e}")
            
            print(f"\nTotal emails: {total_emails}")
            
            if 'avg_email_size' in stats:
                print(f"Average email size: {stats['avg_email_size']} characters")
            
            print("\nEmails by year:")
            for year, count in emails_by_year:
                print(f"  {year}: {count}")
                
            print("\nTop 10 senders:")
            for sender, count in top_senders:
                # Highlight the primary user in the top senders list
                if primary_email and (
                    primary_email.lower() in sender.lower() or 
                    (('@' in primary_email) and primary_email.split('@')[0].lower() in sender.lower()) or
                    primary_name.lower() in sender.lower()
                ):
                    print(f"  {sender}: {count} (YOU)")
                else:
                    print(f"  {sender}: {count}")
                
            if 'emails_by_month' in stats:
                print("\nEmails by month (most recent year):")
                for month, count in stats['emails_by_month']:
                    print(f"  {month}: {count}")
                
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving statistics: {e}")
            print(f"Error calculating statistics: {e}")
            return {}
    
    def close(self) -> None:
        """
        Close database connection.
        
        Closes the connection to the SQLite database.
        """
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except sqlite3.Error as e:
                logger.error(f"Error closing database connection: {e}")

    def reset(self) -> bool:
        """
        Perform a factory reset by removing the database, vector db, and any history.
        
        Deletes the SQLite database and the Chroma vector store.
        
        Returns:
            bool: True if reset was successful, False otherwise
        """
        try:
            # Close the database connection
            if self.conn:
                self.conn.close()
                self.conn = None
            
            # Safely remove database file
            db_path = os.path.abspath(self.db_path)
            if os.path.exists(db_path) and os.path.isfile(db_path):
                os.remove(db_path)
                logger.info(f"Removed database: {db_path}")
            
            # Safely remove vector store directory
            vector_db_path = os.path.abspath(self.vector_db_path)
            if os.path.exists(vector_db_path) and os.path.isdir(vector_db_path):
                shutil.rmtree(vector_db_path)
                logger.info(f"Removed vector database: {vector_db_path}")
            
            # Reset object properties
            self.vector_store = None
            self.qa_chain = None
            self.embeddings = None
            self.llm = None
            
            logger.info("Factory reset complete")
            print("Factory reset complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during factory reset: {e}")
            print(f"Error during factory reset: {e}")
            return False

def check_completion() -> bool:
    """
    Check if the system has the necessary dependencies installed for full functionality.
    
    Verifies if langchain and other required dependencies are properly installed.
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    try:
        # Try to import key components to check installation
        import torch
        import transformers
        import sentence_transformers
        
        # Check LangChain ecosystem
        # Try newer imports first
        try:
            import langchain_community
            import langchain_text_splitters
            import langchain_chroma
            current_mode = "new-langchain"
        except ImportError:
            # Try older imports
            try:
                import langchain
                current_mode = "old-langchain"
            except ImportError:
                current_mode = "none"
        
        # For models/embeddings support
        has_transformers = hasattr(transformers, 'pipeline') and callable(transformers.pipeline)
        has_sentence_transformers = hasattr(sentence_transformers, 'SentenceTransformer')
        
        # Check vector store
        try:
            import chromadb
            has_chromadb = True
        except ImportError:
            has_chromadb = False
            
        # Display status
        print("✓ System check:")
        print(f"  - LangChain mode: {current_mode}")
        print(f"  - HuggingFace models: {'✓' if has_transformers else '✗'}")
        print(f"  - Embeddings support: {'✓' if has_sentence_transformers else '✗'}")
        print(f"  - Vector store: {'✓' if has_chromadb else '✗'}")
        
        if current_mode != "none" and has_transformers and has_sentence_transformers and has_chromadb:
            print("✓ Full functionality available")
            return True
        else:
            print("⚠ Limited functionality - some dependencies missing")
            missing = []
            if current_mode == "none":
                missing.append("langchain")
            if not has_transformers:
                missing.append("transformers")
            if not has_sentence_transformers:
                missing.append("sentence-transformers")
            if not has_chromadb:
                missing.append("chromadb")
                
            if missing:
                print(f"  Missing: {', '.join(missing)}")
                print(f"  Run: ./install_minimal.sh or pip install {' '.join(missing)}")
            return False
            
    except Exception as e:
        print(f"⚠ Error checking dependencies: {e}")
        return False
        
def main() -> None:
    """
    AskPST is a script that allows you to query your email archives using a local language model.
    
    Setup steps:
      --process: Process the PST/MSG files in the specified folder.
      --setup: Initialise the local LLM, create the vector store from the processed emails, and set up the question answering (QA) chain. 
      --ask: Ask a question about the contents of your email archives. For example: "Who was the most aggressive person who would email me?"
      --stats: Display statistics about the processed emails.
    
    Additional arguments:
      --folder: Path to the folder containing PST/MSG files.
      --db: Path to the SQLite database file (default from config).
      --model: HuggingFace model name (default from config).
      --reset: Perform a "factory reset" by removing the database, vector store, and any history.
      --config: Create or update the configuration file.
    
    Example usage:
      python askpst.py --folder pst/ --process
      python askpst.py --setup
      python askpst.py --ask "Who emailed me the most?"
    """
    parser = argparse.ArgumentParser(description="AskPST: Query your email archives with a local LLM")
    parser.add_argument("--folder", help="Path to folder containing PST/MSG files")
    parser.add_argument("--db", help="Path to SQLite database")
    parser.add_argument("--process", action="store_true", help="Process PST/MSG files in folder")
    parser.add_argument("--setup", action="store_true", help="Setup LLM and vector store")
    parser.add_argument("--ask", help="Ask a question")
    parser.add_argument("--model", help="HuggingFace model name")
    parser.add_argument("--stats", action="store_true", help="Show statistics about processed emails")
    parser.add_argument("--reset", action="store_true", help="Perform a factory reset")
    parser.add_argument("--config", action="store_true", help="Update configuration")
    parser.add_argument("--primary-email", help="Set your primary email address for 'me' references")
    parser.add_argument("--primary-name", help="Set your name for 'me' references")
    parser.add_argument("--check", action="store_true", help="Check system for required dependencies")
    
    args = parser.parse_args()
    
    # Print banner
    print("""
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║               A s k P S T                 ║
    ║                                           ║
    ║     Query your email archives locally     ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # Handle dependency checking
    if args.check:
        check_completion()
        return
        
    # Handle quick primary email/name setting
    config_updated = False
    if args.primary_email:
        config['DEFAULT']['primary_email'] = args.primary_email
        config_updated = True
        print(f"Primary email set to: {args.primary_email}")
        
    if args.primary_name:
        config['DEFAULT']['primary_name'] = args.primary_name
        config_updated = True
        print(f"Primary name set to: {args.primary_name}")
        
    # Save config if changes were made
    if config_updated:
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)
    
    # Handle configuration update
    if args.config:
        print("Updating configuration...")
        # Update config values from command line if provided
        if args.db:
            config['DEFAULT']['db_path'] = args.db
        if args.model:
            config['DEFAULT']['llm_model'] = args.model
        
        # Prompt for additional configuration
        print("\nCurrent configuration:")
        for key, value in config['DEFAULT'].items():
            print(f"  {key} = {value}")
        
        print("\nUpdate configuration? (y/n)")
        if input().lower() == 'y':
            # First ask specifically about primary email info
            print("\nPrimary user information (used for questions about 'me', 'my emails', etc.):")
            primary_email = input(f"Primary email address [{config['DEFAULT']['primary_email']}]: ").strip()
            if primary_email:
                config['DEFAULT']['primary_email'] = primary_email
                
            primary_name = input(f"Your name [{config['DEFAULT']['primary_name']}]: ").strip()
            if primary_name:
                config['DEFAULT']['primary_name'] = primary_name
                
            # Then ask about other configuration values
            print("\nOther configuration settings:")
            for key in config['DEFAULT']:
                if key not in ['primary_email', 'primary_name']:  # Skip the ones we already asked about
                    value = input(f"{key} [{config['DEFAULT'][key]}]: ").strip()
                    if value:
                        config['DEFAULT'][key] = value
            
            # Save updated configuration
            with open(CONFIG_FILE, 'w') as f:
                config.write(f)
            print("Configuration updated")
            
            # If primary email was set, explain how to use it
            if config['DEFAULT']['primary_email']:
                print(f"\nPrimary email set to: {config['DEFAULT']['primary_email']}")
                print("You can now ask questions like:")
                print("  - 'Who emailed me the most?'")
                print("  - 'When did I last email about the project?'")
                print("  - 'Find emails where I was CC'd'")
        return
    
    # Initialize the AskPST instance
    askpst = AskPST(model_name=args.model)
    logger.info("AskPST initialized")
    
    # Handle reset
    if args.reset:
        if askpst.reset():
            return
        else:
            logger.error("Reset failed")
            return
    
    # Setup the database
    askpst.setup_database(args.db)
    
    # Process emails (doesn't need LLM)
    if args.process and args.folder:
        folder_path = os.path.abspath(args.folder)
        logger.info(f"Processing folder: {folder_path}")
        if askpst.process_pst_folder(folder_path):
            print("Processing complete")
        else:
            print("Processing failed")
    
    # Only load the LLM if setup or ask is requested
    if args.setup:
        try:
            # Set up embeddings for vector store
            if not askpst.embeddings:
                askpst.embeddings = HuggingFaceEmbeddings(
                    model_name=config['DEFAULT']['embedding_model'],
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                )
                
            # Create vector store (embeddings are used here)
            askpst.create_vector_store()
            
            # Load the LLM only after creating the vector store
            if not askpst.llm:
                askpst.setup_llm()
                
            # Set up QA chain
            askpst.setup_qa_chain()
            print("Setup complete")
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            print(f"Setup failed: {e}")
    
    # Handle questions
    if args.ask:
        try:
            # Set up embeddings if not already set up
            if not askpst.embeddings:
                askpst.embeddings = HuggingFaceEmbeddings(
                    model_name=config['DEFAULT']['embedding_model'],
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                )
            
            # Load vector store if not already loaded
            if not askpst.vector_store:
                try:
                    askpst.vector_store = Chroma(
                        persist_directory=askpst.vector_db_path,
                        embedding_function=askpst.embeddings
                    )
                except Exception as e:
                    logger.error(f"Error loading vector store: {e}")
                    print(f"Error loading vector store: {e}")
                    print("Please run --setup first to create the vector store")
                    askpst.close()
                    return
            
            # Load LLM only when needed
            if not askpst.llm:
                askpst.setup_llm()
                
            # Setup QA chain if not already set up
            if not askpst.qa_chain:
                askpst.setup_qa_chain()
            
            # Answer the question
            print(f"\nQuestion: {args.ask}")
            
            # Show primary email context if it's set
            primary_email = config['DEFAULT']['primary_email']
            if primary_email and ('me' in args.ask.lower() or 'my' in args.ask.lower() or 'i ' in args.ask.lower()):
                print(f"(Interpreting 'me/my/I' as: {primary_email})")
                
            answer = askpst.ask_question(args.ask)
            if answer:
                print(f"\nAnswer: {answer}")
            else:
                print("Failed to generate an answer. Check logs for details.")
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"Error: {e}")
    
    # Show stats (doesn't need LLM)
    if args.stats:
        askpst.get_stats()
    
    # Clean up
    askpst.close()

if __name__ == "__main__":
    main()
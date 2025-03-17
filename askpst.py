#!/usr/bin/env python3
# AskPST - Query your email archives with a local LLM
# Copyright (c) 2025

import os
import re
import argparse
import sqlite3
import numpy as np
from datetime import datetime
from tqdm import tqdm
import extract_msg
from transformers import AutoTokenizer
import torch
from libratom.lib.pff import PffArchive
from pypff import file as PffFile
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import shutil

class AskPST:
    """
    AskPST: Query your email archives with a local LLM.
    
    This class provides methods to process PST and MSG files, extract email content,
    set up a local LLM for question answering, create a vector store for semantic search,
    and provide email statistics.
    """
    def __init__(self, model_name="tiiuae/falcon-7b-instruct"):
        """
        Initializes the AskPST class.
        
        Args:
            model_name (str, optional): The name of the HuggingFace model to use.
                Defaults to "tiiuae/falcon-7b-instruct".
        """
        self.conn = None
        self.db_path = "askpst.db"  # Initialize db_path
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        self.model_name = model_name
        
    def setup_database(self, db_path):
        """
        Set up SQLite database for email storage.
        
        Creates the database if it doesn't exist and sets up the necessary tables
        (emails and processed_files).
        
        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor()
        
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
        
        self.conn.commit()
    
    def process_pst_folder(self, folder_path):
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
            cursor = self.conn.cursor()
            
            print(f"Processing folder: {folder_path}")
            
            pst_files = []
            msg_files = []
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.pst'):
                        pst_files.append(os.path.join(root, file))
                    elif file.lower().endswith('.msg'):
                        msg_files.append(os.path.join(root, file))
            
            print(f"Found {len(pst_files)} PST files and {len(msg_files)} MSG files")
            
            # Process each PST file
            for pst_path in pst_files:
                try:
                    self._process_pst_file(pst_path)
                except Exception as e:
                    print(f"Error processing PST file {pst_path}: {e}")
            
            # Process MSG files as before
            for msg_path in msg_files:
                try:
                    print(f"Processing MSG file: {msg_path}")
                    self._process_msg_file(msg_path)
                except Exception as e:
                    print(f"Error processing MSG file {msg_path}: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            return False

    def _process_pst_file(self, pst_path):
        """
        Process a PST file using libratom.
        
        Extracts messages from the PST file and stores them in the database.
        
        Args:
            pst_path (str): The path to the PST file.
        """
        try:
            print(f"Opening PST file: {pst_path}")
            with PffArchive(pst_path) as archive:
                for folder in archive.folders():
                    self._process_pst_folder_recursive(folder)
        except Exception as e:
            print(f"Error processing PST file {pst_path}: {e}")

    def _process_pst_folder_recursive(self, folder):
        """
        Recursively process a PST folder and its subfolders.
        
        Traverses the folder structure and extracts messages from each folder.
        
        Args:
            folder (libratom.lib.pff.Folder): The PST folder to process.
        """
        try:
            print(f"Processing PST folder: {folder.name}")
            for message in folder.sub_messages:
                try:
                    self._process_extracted_message(message)
                except Exception as e:
                    print(f"Error processing message in folder {folder.name}: {e}")
            for subfolder in folder.sub_folders:
                self._process_pst_folder_recursive(subfolder)
        except Exception as e:
            print(f"Error processing folder {folder.name}: {e}")

    def _process_extracted_message(self, message):
        """
        Process a message extracted from a PST file.
        
        Extracts the message details (subject, sender, recipients, date, body, attachments)
        and stores them in the database.
        
        Args:
            message (libratom.lib.pff.Message): The message object to process.
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
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO emails (subject, sender, recipients, date, body, attachment_count) VALUES (?, ?, ?, ?, ?, ?)",
                (subject, sender, recipients, date, body, attachment_count)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Error processing extracted message: {e}")

    def _process_msg_file(self, msg_path):
        """
        Process a single MSG file.
        
        Extracts the message details (subject, sender, recipients, date, body, attachments)
        and stores them in the database.
        
        Args:
            msg_path (str): The path to the MSG file.
        """
        try:
            print(f"Opening MSG file: {msg_path}")
            msg = extract_msg.Message(msg_path)
            subject = msg.subject or ""
            sender = msg.sender or ""
            recipients = msg.to or ""
            date = msg.date or datetime.now().isoformat()
            body = msg.body or ""
            attachment_count = len(msg.attachments)
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO emails (subject, sender, recipients, date, body, attachment_count) VALUES (?, ?, ?, ?, ?, ?)",
                (subject, sender, recipients, date, body, attachment_count)
            )
            self.conn.commit()
            print(f"Processed MSG file: {msg_path}")
        except Exception as e:
            print(f"Error processing MSG file {msg_path}: {e}")
    
    def setup_llm(self):
        """
        Initialize the local LLM using HuggingFace models.
        
        Sets up the embeddings, tokenizer, and text generation pipeline for the LLM.
        """
        print("Setting up LLM and embeddings...")
        
        # Set up embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Create LangChain LLM wrapper
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        print("LLM setup complete")
    
    def create_vector_store(self):
        """
        Create vector store from processed emails.

        Retrieves all emails from the database, creates documents from them,
        splits the documents into chunks, and builds a Chroma vector store.

        A vector store is a data structure that stores text data in the form of vectors.
        These vectors are numerical representations of the text, which can be used for
        efficient similarity search and retrieval. By converting text data into vectors,
        it becomes easier to perform operations like searching for similar documents,
        clustering, and classification.

        Steps:

        1. Retrieve all emails from the database.
        2. Create documents from the retrieved emails, including metadata.
        3. Split the documents into smaller chunks for better processing.
        4. Build a Chroma vector store from the chunks using embeddings.

        Prints:

            Status messages indicating the progress of vector store creation.

        """
        print("Creating vector store from emails...")
        
        # Get all emails from database
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, subject, sender, recipients, date, body FROM emails")
        emails = cursor.fetchall()
        
        documents = []
        
        # Create documents from emails
        for email_id, subject, sender, recipients, date, body in tqdm(emails, desc="Preparing documents"):
            if not body:
                continue
                
            metadata = {
                "id": email_id,
                "subject": subject,
                "sender": sender,
                "recipients": recipients,
                "date": date
            }
            
            # Create email text with metadata
            text = f"Subject: {subject}\nFrom: {sender}\nTo: {recipients}\nDate: {date}\n\n{body}"
            documents.append({"page_content": text, "metadata": metadata})
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        splits = []
        for doc in tqdm(documents, desc="Splitting documents"):
            splits.extend(text_splitter.split_text(doc["page_content"]))
        
        # Create vector store
        self.vector_store = Chroma.from_texts(
            texts=splits,
            embedding=self.embeddings,
            persist_directory="./email_chroma_db"
        )
        
        print(f"Vector store created with {len(splits)} chunks")
    
    def setup_qa_chain(self):
        """
        Set up question answering chain.
        
        Defines the prompt template and creates a retrieval QA chain using the LLM and vector store.
        
        A QA (Question Answering) chain is a sequence of operations that takes a user's question,
        retrieves relevant context from a data source (in this case, the vector store), and uses
        a language model to generate an answer based on the retrieved context. This allows for
        efficient and accurate responses to user queries by leveraging both the semantic search
        capabilities of the vector store and the generative capabilities of the language model.

        """
        # Define prompt template
        template = """
        You are an email analysis assistant called AskPST. Use the following pieces of context about emails to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Be specific in your answers and include relevant dates, names, and quotes from emails when available.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("QA chain setup complete")
    
    def ask_question(self, question):
        """
        Ask a question about the emails.
        
        Queries the vector store and uses the LLM to generate an answer based on the email content.
        
        Args:
            question (str): The question to ask.
        
        Returns:
            str: The answer generated by the LLM.
        """
        if not self.qa_chain:
            print("QA chain not set up yet")
            return None
            
        return self.qa_chain.run(question)
    
    def get_stats(self):
        """
        Get statistics about the processed emails.
        
        Retrieves and prints statistics such as the total number of emails,
        emails by year, and top senders.
        """
        if not self.conn:
            print("Database not connected")
            return
            
        cursor = self.conn.cursor()
        
        # Total emails
        cursor.execute("SELECT COUNT(*) FROM emails")
        total_emails = cursor.fetchone()[0]
        
        # Email by year
        cursor.execute("""
        SELECT substr(date, 1, 4) as year, COUNT(*) as count 
        FROM emails 
        WHERE date IS NOT NULL
        GROUP BY year 
        ORDER BY year
        """)
        emails_by_year = cursor.fetchall()
        
        # Top senders
        cursor.execute("""
        SELECT sender, COUNT(*) as count 
        FROM emails 
        GROUP BY sender 
        ORDER BY count DESC 
        LIMIT 5
        """)
        top_senders = cursor.fetchall()
        
        # Print statistics
        print("\n=== Email Statistics ===")
        print(f"Total emails: {total_emails}")
        
        print("\nEmails by year:")
        for year, count in emails_by_year:
            print(f"  {year}: {count}")
            
        print("\nTop senders:")
        for sender, count in top_senders:
            print(f"  {sender}: {count}")
    
    def close(self):
        """
        Close database connection.
        
        Closes the connection to the SQLite database.
        """
        if self.conn:
            self.conn.close()

    def reset(self):
        """
        Perform a factory reset by removing the database, vector db, and any history.
        
        Deletes the SQLite database and the Chroma vector store.
        """
        if self.conn:
            self.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print(f"Removed database: {self.db_path}")
        if os.path.exists("./email_chroma_db"):
            shutil.rmtree("./email_chroma_db")
            print("Removed vector database: ./email_chroma_db")
        print("Factory reset complete")

def main():
    """
    AskPST is a script that allows you to query your email archives using a local language model.
    
    Setup steps:
      --process: Process the PST/MSG files in the specified folder.
      --setup: Initialise the local LLM, create the vector store from the processed emails, and set up the question answering (QA) chain. This uses the specified or default HuggingFace model.
      --ask: Ask a question about the contents of your email archives. For example: "Who was the most aggressive person who would email me?" or "When did we first start discussing a contract with X?"
      --stats: Display statistics about the processed emails.
    
    Additional arguments:
      --folder: Path to the folder containing PST/MSG files.
      --db: Path to the SQLite database file (default: askpst.db).
      --model: HuggingFace model name (default: tiiuae/falcon-7b-instruct).
      --reset: Perform a "factory reset" by removing the database, vector store, and any history.
    
    Example usage:
      python askpst.py --folder pst/ --process
      python askpst.py --setup --model "tiiuae/falcon-7b-instruct"
      python askpst.py --ask "Who emailed me the most?"
      
    """
    parser = argparse.ArgumentParser(description="AskPST: Query your email archives with a local LLM")
    parser.add_argument("--folder", help="Path to folder containing PST/MSG files")
    parser.add_argument("--db", default="askpst.db", help="Path to SQLite database")
    parser.add_argument("--process", action="store_true", help="Process PST/MSG files in folder")
    parser.add_argument("--setup", action="store_true", help="Setup LLM and vector store")
    parser.add_argument("--ask", help="Ask a question")
    parser.add_argument("--model", default="tiiuae/falcon-7b-instruct", help="HuggingFace model name")
    parser.add_argument("--stats", action="store_true", help="Show statistics about processed emails")
    parser.add_argument("--reset", action="store_true", help="Perform a factory reset")
    
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
    
    askpst = AskPST(model_name=args.model)
    
    if args.reset:
        askpst.reset()
        return
    
    askpst.setup_database(args.db)
    
    if args.process and args.folder:
        print(f"Processing folder: {args.folder}")
        askpst.process_pst_folder(args.folder)
        print("Processing complete")
    
    if args.setup:
        askpst.setup_llm()
        askpst.create_vector_store()
        askpst.setup_qa_chain()
        print("Setup complete")
    
    if args.ask:
        if not askpst.qa_chain:
            askpst.setup_llm()
            try:
                askpst.vector_store = Chroma(
                    persist_directory="./email_chroma_db",
                    embedding_function=askpst.embeddings
                )
                askpst.setup_qa_chain()
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Please run --setup first to create the vector store")
                askpst.close()
                return
        
        answer = askpst.ask_question(args.ask)
        print(f"\nQuestion: {args.ask}\n")
        print(f"Answer: {answer}")
    
    if args.stats:
        askpst.get_stats()
    
    askpst.close()

if __name__ == "__main__":
    main()
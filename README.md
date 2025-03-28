# AskPST

A tool to process email archives and interact with them using Large Language Models.

## Overview

AskPST allows you to:
- Process email archives (PST, MSG, and MBOX formats)
- Create vector embeddings of email content
- Query your email archive using natural language
- Get semantic answers beyond basic search capabilities

## Requirements

- Python 3.10.8 or higher
- PST files in the designated folder

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/askpst.git
cd askpst

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with available dependencies
pip install -e ".[all]"

# Or install with specific feature sets:
# pip install -e ".[llm]"  # Just LLM support
# pip install -e ".[pst]"  # Just PST processing tools (without libpff)
# pip install -e ".[dev]"  # Development tools

# Optional: Install support for different email formats

# For PST files (Outlook Personal Storage):
# On macOS (including Apple Silicon):
brew install libpst             # Install the libpst command-line tools
pip install libpff-python       # Python binding that works on Apple Silicon

# On Windows/Linux (or as an alternative):
pip install pypff               # Alternative Python binding (may not work on Apple Silicon)

# For MSG files (Outlook Message format):
pip install extract-msg

# MBOX format is supported natively (no additional packages needed)

# Install all email format support:
pip install -e ".[pst]"

# Download the Llama3 model (if you want to use that LLM)
python scripts/download_model.py
```

## Usage

### Setup and Processing PST Files

```bash
# Process PST files in the default location
python -m askpst.simple_cli --pst-dir="pst/" --user-email="your.email@example.com" --user-name="Your Name"
```

This will:
1. Create a SQLite database (`askpst_data.db`)
2. Extract emails from PST files in the specified directory
3. Store the emails in the database

### Asking Questions

There are two ways to search your emails:

#### Simple Keyword Search

```bash
# Search for specific keywords in your emails
python -m askpst.simple_ask "resignation"

# Or search for multiple keywords
python -m askpst.simple_ask "project update deadline"
```

The simple_ask tool will:
1. Extract important keywords from your query
2. Search for emails containing those keywords
3. Display the matching emails sorted by date

#### Semantic Search with LLM

```bash
# Use the LLM-powered search for more complex questions
python -m askpst.llm_ask "who seems the happiest in their emails?"

# Search for emotional cues
python -m askpst.llm_ask "who sent me the angriest emails?"

# Use the simple LLM model for fallback functionality
python -m askpst.llm_ask "how many resignation emails did I get?" --model=simple

# Adjust the number of emails used for context (default is 25)
python -m askpst.llm_ask "important emails from Andy" --top-k 50

# Force a specific model (options: llama3, deepseek, simple)
python -m askpst.llm_ask "who complained about Gina?" --model=deepseek

# For a cleaner experience, use the provided scripts or quiet mode:
python -m askpst.llm_ask "Who asked me about bereavement?" --quiet    # Suppresses most warnings and errors
python scripts/run_askpst_clean.py "Who was angry?"                   # Complete suppression of import errors
./scripts/run_askpst_silent.sh "Who left the company?"                # Guaranteed silent operation - zero warnings/errors
```

The llm_ask tool provides:
1. Vector search for semantic matching (when available)
2. Keyword search with emotion/sentiment detection as fallback
3. LLM-powered analysis of the email content
4. Responses to complex questions beyond simple keyword matching
5. Display of up to 2000 characters from the most relevant email
6. Analysis of 25 emails by default (adjustable with --top-k)

### Using the Library Directly

The `examples` directory contains scripts that demonstrate how to use the library directly:

#### Processing PST Files

```bash
python examples/process_pst.py --pst-dir="./pst" --user-email="your.email@example.com" --user-name="Your Name"
```

#### Querying Emails

```bash
python examples/query_emails.py "How many resignation emails did I receive in 2018?"
```

## Project Structure

The project is organized as follows:

- `askpst/`: Main package directory
  - `__init__.py`: Package initialization
  - `cli.py`: Main CLI interface (requires LLM dependencies)
  - `simple_cli.py`: Simplified CLI for processing PST files
  - `simple_ask.py`: Simplified CLI for keyword-based search
  - `pst_processor.py`: Core logic for processing and storing PST data
  - `models/`: Directory for LLM models
    - `llm_interface.py`: Interface for working with different LLMs
  - `utils/`: Utility functions
    - `email_importers.py`: Functions for importing different email formats
    - `embeddings.py`: Functions for creating vector embeddings

- `tests/`: Test directory
  - `test_pst_processor.py`: Tests for the PST processor
  - `test_embeddings.py`: Tests for the embedding utilities
  - `test_llm_interface.py`: Tests for the LLM interface

- `scripts/`: Utility scripts
  - `download_model.py`: Script for downloading Llama3 model

- `examples/`: Example scripts
  - `process_pst.py`: Example for processing PST files
  - `query_emails.py`: Example for querying emails

## Advanced Configuration

### Building the Vector Index

For semantic search to work properly, you need to build a vector index. Use the provided script:

```bash
# Build the vector index for semantic search
python scripts/build_index.py

# Or specify a custom database path
python scripts/build_index.py --db-path="custom_path.db"
```

This creates a FAISS index file (e.g., `askpst_data.faiss`) in the same directory as your database, which enables semantic search with LLMs.

### LLM Models

Edit the `.env` file to configure which LLM model to use:

```
LLM_MODEL=llama3
# Alternatively: LLM_MODEL=simple for fallback functionality
```

Download a model for local processing:

```bash
python scripts/download_sample_model.py
```

## Troubleshooting

### PST Processing Issues

- **Error accessing PST files**: Make sure you have the correct permissions to read the PST files.
- **PST library errors**: The PST processing libraries may have issues with some PST files. 
  - On macOS, install `libpst` with Homebrew: `brew install libpst`
  - On Apple Silicon Macs, some libraries may have compatibility issues
- **Attachment extraction errors**: Some attachments may not be properly extracted due to limitations in the PST processing libraries. These errors are non-fatal and the rest of the email data will still be processed.

### Dependency Issues

- **NumPy compatibility errors**: Several dependencies (like PyPFF and FAISS) require NumPy < 2.0:
  ```bash
  pip uninstall -y numpy && pip install numpy==1.24.3
  ```
  
  Alternatively, you can use the provided requirements.txt file:
  ```bash
  pip install -r requirements.txt
  ```
  
  If you see warnings like "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.4", follow these steps to resolve the issue. 

  For a quick workaround, you can use our silent scripts that completely suppress these errors:
  ```bash
  python scripts/run_askpst_clean.py "your question"  # Completely suppresses import errors
  # OR
  ./scripts/run_askpst_silent.sh "your question"      # Wrapper around the clean script
  ```

- **Transformers/Torch errors**: The LLM features require compatible versions of transformers and torch:
  ```bash
  pip uninstall -y transformers && pip install transformers==4.30.2
  pip uninstall -y sentence-transformers && pip install sentence-transformers==2.2.2
  pip uninstall -y torch && pip install torch==2.0.1
  ```

### Semantic Search Issues

- **"No FAISS index available"**: If you get this error, you need to build the vector index:
  ```bash
  python scripts/build_index.py
  ```

- **LLM compatibility issues**: If the LLM fails to load, try using the simple model:
  ```bash
  python -m askpst.llm_ask "your question" --model=simple
  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

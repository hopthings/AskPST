# AskPST

AskPST is a Python script that allows you to query your email archives using a local Large Language Model (LLM). It processes PST and MSG files, extracts email content, and creates a vector store for semantic search and question answering.

## Features

- Processes PST and MSG email files.
- Uses a local LLM for question answering.
- Creates a vector store for efficient semantic search.
- Provides email statistics.
- Supports reprocessing of email archives.
- Includes a reset option to clear all data and start fresh.
- Optimized performance with multithreading for processing large email archives.
- Robust error handling and fallback mechanisms.

## Requirements

- Python 3.10+
- The dependencies listed in `requirements.txt`.

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd AskPST
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate  # On Windows
   ```

3. **Quick Setup (Recommended):**

   ```bash
   chmod +x install_minimal.sh
   ./install_minimal.sh
   ```

   OR

   ```bash
   chmod +x setup_environment.sh
   ./setup_environment.sh
   ```

4. **Manual Installation:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Check Installation:**

   ```bash
   python askpst.py --check
   ```

## Usage

### 1. Process Email Archives

To process PST and MSG files in a folder, use the `--process` and `--folder` arguments:

```bash
python askpst.py --process --folder <path_to_email_folder>
```

This command will scan the specified folder for PST and MSG files, extract the email content, and store it in a SQLite database.

### 2. Setup LLM and Vector Store

To set up the local LLM and create the vector store, use the `--setup` argument:

```bash
python askpst.py --setup
```

This command will initialize the LLM, create embeddings for the email content, and build a vector store for efficient semantic search. The vector creation process uses multi-threading for better performance with large email archives.

### 3. Ask Questions

To ask a question about the emails, use the `--ask` argument:

```bash
python askpst.py --ask "<your_question>"
```

This command will query the vector store and use the LLM to generate an answer based on the email content. You can ask questions like:
- "Who emailed me the most?"
- "What was the discussion about project deadlines?"
- "Did anyone mention vacation time in February?"

### 4. Set Your Identity

To tell AskPST your email address and name for "me" references:

```bash
python askpst.py --primary-email "youremail@example.com" --primary-name "Your Name"
```

This allows the system to understand references to "me", "my", or "I" in your questions.

### 5. View Statistics

To view statistics about the processed emails, use the `--stats` argument:

```bash
python askpst.py --stats
```

This command will display the total number of emails, emails by year, and top senders.

### 6. Factory Reset

To perform a factory reset and remove the database, vector store, and any history, use the `--reset` argument:

```bash
python askpst.py --reset
```

This command will delete the SQLite database and the Chroma vector store.

### 7. Update Configuration

To update the configuration settings:

```bash
python askpst.py --config
```

## Arguments

- `--folder`: Path to folder containing PST/MSG files.
- `--db`: Path to SQLite database (default: `askpst.db`).
- `--process`: Process PST/MSG files in folder.
- `--setup`: Setup LLM and vector store.
- `--ask`: Ask a question.
- `--model`: HuggingFace model name (default from config, typically `distilgpt2` for testing).
- `--stats`: Show statistics about processed emails.
- `--reset`: Perform a factory reset.
- `--config`: Update configuration settings.
- `--primary-email`: Set your primary email address for 'me' references.
- `--primary-name`: Set your name for 'me' references.
- `--check`: Check system for required dependencies.

## Performance Optimizations

- **Multithreaded Processing**: Email processing is parallelized for better performance.
- **Smart Sampling**: For very large archives (>100,000 emails), random sampling is used.
- **Reduced Vector Sizes**: Optimized chunk sizes to balance detail and processing speed.
- **Memory Efficiency**: Batch processing reduces memory usage with large archives.
- **Progress Reporting**: Detailed progress updates during long operations.

## Notes

- The script can process both PST and MSG email files.
- The vector store is persisted in the `./email_chroma_db` directory.
- The SQLite database is stored in the `askpst.db` file.
- Configuration is stored in `askpst.config` and can be edited directly or via the `--config` option.

## License

Copyright (c) 2025
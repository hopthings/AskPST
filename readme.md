# AskPST

AskPST is a Python script that allows you to query your email archives using a local Large Language Model (LLM). It processes PST and MSG files, extracts email content, and creates a vector store for semantic search and question answering.

## Features

- Processes PST and MSG email files.
- Uses a local LLM for question answering.
- Creates a vector store for efficient semantic search.
- Provides email statistics.
- Supports reprocessing of email archives.
- Includes a reset option to clear all data and start fresh.

## Requirements

- Python 3.10+
- The dependencies listed in `requirements.txt`.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd AskPST
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install `libratom`:**

    ```bash
    pip install libratom
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

This command will initialize the LLM, create embeddings for the email content, and build a vector store for efficient semantic search.

### 3. Ask Questions

To ask a question about the emails, use the `--ask` argument:

```bash
python askpst.py --ask "<your_question>"
```

This command will query the vector store and use the LLM to generate an answer based on the email content.

### 4. View Statistics

To view statistics about the processed emails, use the `--stats` argument:

```bash
python askpst.py --stats
```

This command will display the total number of emails, emails by year, and top senders.

### 5. Reprocess Archives

To reprocess a folder, even if it has been processed before, use the `--reprocess` argument:

```bash
python askpst.py --process --folder <path_to_email_folder> --reprocess
```

### 6. Factory Reset

To perform a factory reset and remove the database, vector store, and any history, use the `--reset` argument:

```bash
python askpst.py --reset
```

This command will delete the SQLite database and the Chroma vector store.

## Arguments

-   `--folder`: Path to folder containing PST/MSG files.
-   `--db`: Path to SQLite database (default: `askpst.db`).
-   `--process`: Process PST/MSG files in folder.
-   `--setup`: Setup LLM and vector store.
-   `--ask`: Ask a question.
-   `--model`: HuggingFace model name (default: `meta-llama/Llama-2-7b-chat-hf`).
-   `--stats`: Show statistics about processed emails.
-   `--reprocess`: Re-process the folder even if it has been processed before.
-   `--reset`: Perform a factory reset.

## Notes

-   The script skips PST files as PST processing is not supported.
-   The vector store is persisted in the `./email_chroma_db` directory.
-   The SQLite database is stored in the `askpst.db` file.

## License

Copyright (c) 2025
#\!/bin/bash

echo "Installing minimal required packages for AskPST..."
echo "This script installs only what's needed for the core LLM functionality."

# Create and activate a new virtual environment
python -m venv .env_minimal
source .env_minimal/bin/activate

# Install the absolutely necessary packages
pip install --upgrade pip
pip install "torch>=2.0.0,<2.1.0"
pip install "transformers>=4.30.0,<4.36.0"
pip install "langchain>=0.0.200,<0.1.0"
pip install "chromadb>=0.4.10,<0.5.0"
pip install "sentence-transformers>=2.2.0,<3.0.0"
pip install "numpy>=1.24.0,<1.26.0"
pip install "extract_msg>=0.41.0"
pip install "tqdm>=4.65.0"
pip install "libratom>=0.7.1"

echo "Installation complete. Activate with: source .env_minimal/bin/activate"

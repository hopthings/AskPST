#!/usr/bin/env python3
"""Script to download a small sample model for testing."""

import os
import sys
import argparse
import urllib.request
import shutil
from tqdm import tqdm

# A small test model (llama2 7B Q4_0, only 3.8GB) to keep downloads reasonable
DEFAULT_MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf"
DEFAULT_MODEL_DIR = "models"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    """Download a small model for testing."""
    parser = argparse.ArgumentParser(description="Download sample LLM model for testing")
    parser.add_argument("--url", default=DEFAULT_MODEL_URL, help="URL of the model to download")
    parser.add_argument("--dir", default=DEFAULT_MODEL_DIR, help="Directory to save the model")
    args = parser.parse_args()
    
    # Create directory if it doesn't exist
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    
    # Get model filename from URL
    model_filename = os.path.basename(args.url)
    model_path = os.path.join(args.dir, model_filename)
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        sys.exit(0)
    
    print(f"Downloading model from {args.url}...")
    print(f"This will download approximately 3.8GB. Please be patient.")
    
    try:
        download_url(args.url, model_path)
        print(f"Model downloaded successfully to {model_path}")
        
        # Update .env file
        env_file = ".env"
        env_content = f"LLM_MODEL=llama3\nLLAMA_MODEL_PATH={model_path}\n"
        
        with open(env_file, "w") as f:
            f.write(env_content)
        
        print(f"Created {env_file} with model path")
    
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
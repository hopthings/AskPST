#!/usr/bin/env python3
"""Script to download LLM models for AskPST."""

import os
import argparse
import subprocess
import sys

# Default model URLs
LLAMA3_8B_URL = "https://huggingface.co/TheBloke/Llama-3-8B-Chat-GGUF/resolve/main/llama-3-8b-chat.Q4_K_M.gguf"
LLAMA3_70B_URL = "https://huggingface.co/TheBloke/Llama-3-70B-Chat-GGUF/resolve/main/llama-3-70b-chat.Q4_K_M.gguf"
DEEPSEEK_7B_URL = "https://huggingface.co/TheBloke/deepseek-coder-7B-instruct-GGUF/resolve/main/deepseek-coder-7b-instruct.Q4_K_M.gguf"
DEEPSEEK_33B_URL = "https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF/resolve/main/deepseek-coder-33b-instruct.Q4_K_M.gguf"

DEFAULT_MODEL_DIR = "models"

def download_model(url: str, dir_path: str) -> str:
    """Download a model from a URL.
    
    Args:
        url: URL of the model to download
        dir_path: Directory to save the model
        
    Returns:
        Path to the downloaded model
    """
    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Get model filename from URL
    model_filename = os.path.basename(url)
    model_path = os.path.join(dir_path, model_filename)
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path
    
    print(f"Downloading model from {url}...")
    print(f"This may take a while depending on your internet speed.")
    
    try:
        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            # Use curl on macOS and Linux
            subprocess.run(["curl", "-L", url, "-o", model_path], check=True)
        else:
            # Use python's urllib on Windows
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
        
        print(f"Model downloaded successfully to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        sys.exit(1)

def update_env_file(model_path: str, model_type: str):
    """Update the .env file with model paths.
    
    Args:
        model_path: Path to the downloaded model
        model_type: Type of the model (llama3 or deepseek)
    """
    env_file = ".env"
    env_vars = {}
    
    # Read existing .env file if it exists
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
    
    # Update model path variables
    if model_type == "llama3":
        env_vars["LLM_MODEL"] = "llama3"
        env_vars["LLAMA_MODEL_PATH"] = model_path
    elif model_type == "deepseek":
        env_vars["DEEPSEEK_MODEL_PATH"] = model_path
    
    # Write the updated .env file
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"Updated {env_file} with model path")

def main():
    """Download the selected model."""
    parser = argparse.ArgumentParser(description="Download LLM models for AskPST")
    parser.add_argument("--model", default="llama3-8b", 
                        choices=["llama3-8b", "llama3-70b", "deepseek-7b", "deepseek-33b"],
                        help="Which model to download")
    parser.add_argument("--dir", default=DEFAULT_MODEL_DIR, help="Directory to save the model")
    args = parser.parse_args()
    
    # Select URL based on model type
    if args.model == "llama3-8b":
        url = LLAMA3_8B_URL
        model_type = "llama3"
    elif args.model == "llama3-70b":
        url = LLAMA3_70B_URL
        model_type = "llama3"
    elif args.model == "deepseek-7b":
        url = DEEPSEEK_7B_URL
        model_type = "deepseek"
    elif args.model == "deepseek-33b":
        url = DEEPSEEK_33B_URL
        model_type = "deepseek"
    
    # Download the model
    model_path = download_model(url, args.dir)
    
    # Update .env file
    update_env_file(model_path, model_type)
    
    print("Model setup complete!")
    print(f"You can now use the model with: python -m askpst.llm_ask \"your question\" --model={model_type}")

if __name__ == "__main__":
    main()
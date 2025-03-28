from setuptools import setup, find_packages

setup(
    name="askpst",
    version="0.1.0",
    description="A tool to process PST files and interact with them using LLMs",
    author="Matt Hopkins",
    packages=find_packages(),
    python_requires=">=3.10.8",
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "typer",
        "python-dotenv",
        "rich",
    ],
    extras_require={
        "llm": [
            "torch",
            "transformers",
            "sentencepiece",
            "faiss-cpu",
            "llama-cpp-python",
        ],
        "pst": [
            # PST file processing libraries
            "libpff-python",  # Works on Apple Silicon
            "tika",
            "extract-msg",
            # Optional dependencies that may not work on Apple Silicon:
            # "pypff",
            # Note: On macOS, also run: brew install libpst
        ],
        "dev": [
            "pytest",
            "flake8",
            "black",
            "isort",
            "mypy",
        ],
        "all": [
            "torch",
            "transformers",
            "sentencepiece",
            "faiss-cpu",
            "llama-cpp-python",
            "tika",
            "extract-msg",
            "pypff",  # Using pypff instead of libpff-python
        ]
    },
    entry_points={
        "console_scripts": [
            "askpst=askpst.cli:app",
        ],
    },
)
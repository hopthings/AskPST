"""AskPST package for processing PST files and querying them with LLMs."""

# Set up global warning filters
import os
import warnings

# Suppress NumPy 1.x / 2.x compatibility warnings 
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x cannot be run in NumPy 2")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

__version__ = "0.1.0"

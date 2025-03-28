"""Utility functions for the AskPST package."""

import warnings

# Suppress NumPy 1.x / 2.x compatibility warnings
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x cannot be run in NumPy 2")

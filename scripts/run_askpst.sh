#!/bin/bash
# Simple wrapper script to run AskPST with suppressed warnings

# Set environment variable to suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning"

# Redirect stderr to /dev/null to suppress the NumPy warnings
python -m askpst.llm_ask "$@" 2>/dev/null

# Exit with the same status code
exit $?
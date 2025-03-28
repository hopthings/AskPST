#!/usr/bin/env python3
"""
Clean wrapper script for askpst.llm_ask that completely suppresses import errors.
This script redirects stderr before any imports happen.
"""

import os
import sys

# Redirect stderr to /dev/null before any imports
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    # Only import after stderr is redirected
    import argparse
    from askpst.llm_ask import main as llm_ask_main
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Clean AskPST interface (no warnings or errors)")
    parser.add_argument("question", nargs="?", help="Question to ask about your emails")
    parser.add_argument("--db-path", default="askpst_data.db", help="Path to the database")
    parser.add_argument("--top-k", type=int, default=25, help="Maximum number of results to use for context")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable vector search and use keywords only")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search (vector + keyword)")
    parser.add_argument("--model", help="LLM model to use (llama3, deepseek, simple)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fallback to other models")
    parser.add_argument("--quiet", action="store_true", help="Suppress all non-essential output")
    
    # Parse args and ensure quiet flag is set
    args, unknown = parser.parse_known_args()
    
    # Only add --quiet if it's not already present
    if '--quiet' not in sys.argv:
        sys.argv = [sys.argv[0]] + ['--quiet'] + sys.argv[1:]
    
    # Run the main function and exit with its return code
    sys.exit(llm_ask_main())
    
except Exception as e:
    # Restore stderr for error reporting
    sys.stderr = stderr_backup
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
    
finally:
    # Restore stderr in case it's needed
    sys.stderr = stderr_backup
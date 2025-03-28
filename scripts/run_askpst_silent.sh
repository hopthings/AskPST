#!/bin/bash
# Silent wrapper script for the clean Python implementation
# This guarantees zero errors or warnings will be shown

# Run the clean Python script which handles stderr redirection internally
python "$(dirname "$0")/run_askpst_clean.py" "$@"

# Exit with the same status code
exit $?
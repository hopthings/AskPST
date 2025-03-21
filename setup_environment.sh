#\!/bin/bash

# AskPST Environment Setup Script
# This script creates a fresh virtual environment and installs all required dependencies
# in the correct order to avoid conflicts.

# Color constants for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored status messages
print_status() {
  echo -e "${BLUE}[SETUP]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Determine OS for environment activation command
if [[ "$OSTYPE" == "darwin"* ]]; then
  ACTIVATE_CMD="source venv/bin/activate"
  print_status "Detected macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  ACTIVATE_CMD="source venv/bin/activate"
  print_status "Detected Linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  ACTIVATE_CMD="venv\\Scripts\\activate"
  print_status "Detected Windows"
else
  ACTIVATE_CMD="source venv/bin/activate"
  print_warning "Unknown OS type: $OSTYPE, using default activation command"
fi

# Create virtual environment folder name
VENV_NAME="venv"

# Start setup
print_status "Starting AskPST environment setup..."

# Check if Python is installed
if \! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing virtual environment..."
        rm -rf $VENV_NAME
    else
        print_warning "Setup cancelled. Using existing environment."
        print_status "To activate the environment, run: $ACTIVATE_CMD"
        exit 0
    fi
fi

# Create virtual environment
print_status "Creating virtual environment in '$VENV_NAME'..."
python3 -m venv $VENV_NAME
if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment. Please check your Python installation."
    exit 1
fi
print_success "Virtual environment created successfully\!"

# Activate virtual environment
print_status "Activating virtual environment..."
eval "$ACTIVATE_CMD"
if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip
print_status "Upgrading pip to latest version..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    print_warning "Failed to upgrade pip, continuing with existing version."
fi

# Install packages in the correct order to avoid dependency conflicts
print_status "Installing core dependencies..."

# First, install PyTorch and transformers
print_status "Installing PyTorch and transformers..."
pip install torch==2.0.1
pip install transformers==4.35.0
if [ $? -ne 0 ]; then
    print_error "Failed to install PyTorch or transformers. Setup cannot continue."
    exit 1
fi

# Install sentence-transformers (for embeddings)
print_status "Installing sentence-transformers..."
pip install sentence-transformers==2.2.2
if [ $? -ne 0 ]; then
    print_error "Failed to install sentence-transformers. Setup cannot continue."
    exit 1
fi

# Install langchain ecosystem packages
print_status "Installing LangChain ecosystem packages..."
pip install pydantic==1.10.8
pip install langchain==0.0.267
pip install chromadb==0.4.18
if [ $? -ne 0 ]; then
    print_error "Failed to install LangChain packages. Setup cannot continue."
    exit 1
fi

# Install email processing packages
print_status "Installing email processing packages..."
pip install extract_msg==0.41.0 tqdm==4.66.1 libratom==0.7.1
if [ $? -ne 0 ]; then
    print_warning "Some email processing packages failed to install. Basic functionality may be limited."
fi

# Install utility packages
print_status "Installing utility packages..."
pip install numpy==1.24.4 psutil==5.9.5
if [ $? -ne 0 ]; then
    print_warning "Failed to install some utility packages. Performance optimizations may be limited."
fi

# Final check - import critical packages to verify they work
print_status "Verifying installation..."
python -c "import torch; import transformers; import langchain; import chromadb; import sentence_transformers; print('All critical packages imported successfully\!')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Some packages could not be imported. The environment may not be properly set up."
else
    print_success "All critical packages verified successfully\!"
fi

# Create a simple activation script
cat > activate_env.sh << 'END'
#\!/bin/bash
source venv/bin/activate
END

chmod +x activate_env.sh

# Display instructions
print_success "\nAskPST environment setup complete\!"
echo -e "\nTo activate this environment:"
echo -e "  ${BLUE}$ACTIVATE_CMD${NC}"
echo -e "  or run ${BLUE}./activate_env.sh${NC}"
echo -e "\nTo test the environment with AskPST:"
echo -e "  ${BLUE}python askpst.py --setup${NC}"
echo -e "  ${BLUE}python askpst.py --ask \"Who emailed me the most?\"${NC}"
echo -e "\nHappy querying\!"

# Exit the script but leave the environment activated for the user
exec bash

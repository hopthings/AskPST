#!/usr/bin/env python3
"""Simple PST file viewer to test pypff functionality."""

import os
import sys
import argparse

try:
    import pypff
    print("pypff imported successfully")
except ImportError:
    print("Error: pypff not found. Please install with: pip install pypff")
    sys.exit(1)


def process_folder(folder, indent=0):
    """Process a folder and its subfolders."""
    # Print folder information
    name = folder.get_name() or "Unnamed folder"
    print(f"{' ' * indent}Folder: {name} ({folder.get_number_of_sub_folders()} subfolders, {folder.get_number_of_sub_messages()} messages)")
    
    # Print some information about messages
    if folder.get_number_of_sub_messages() > 0:
        print(f"{' ' * indent}Messages:")
        
        # Print the first message as an example
        try:
            message = folder.get_sub_message(0)
            print(f"{' ' * (indent+2)}Example message:")
            print(f"{' ' * (indent+4)}Subject: {message.get_subject() or 'N/A'}")
            print(f"{' ' * (indent+4)}From: {message.get_sender_name() or 'N/A'}")
            
            # List available attributes and methods
            attributes = [attr for attr in dir(message) if not attr.startswith('_')]
            print(f"{' ' * (indent+4)}Available attributes/methods: {', '.join(attributes[:10])}...")
            
        except Exception as e:
            print(f"{' ' * (indent+2)}Error accessing message: {str(e)}")
    
    # Process subfolders
    for i in range(folder.get_number_of_sub_folders()):
        subfolder = folder.get_sub_folder(i)
        process_folder(subfolder, indent + 2)


def main():
    """Main function to view PST file structure."""
    parser = argparse.ArgumentParser(description="Simple PST file viewer")
    parser.add_argument("pst_file", help="Path to the PST file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pst_file):
        print(f"Error: File {args.pst_file} does not exist")
        return 1
    
    try:
        # Open the PST file
        pst = pypff.file()
        pst.open(args.pst_file)
        
        print(f"PST file opened: {args.pst_file}")
        
        # List all available attributes and methods
        attributes = [attr for attr in dir(pst) if not attr.startswith('_')]
        print(f"Available PST file attributes/methods: {', '.join(attributes)}")
        
        # Get the root folder
        root_folder = pst.get_root_folder()
        
        # Process the root folder
        process_folder(root_folder)
        
        # Close the PST file
        pst.close()
        
    except Exception as e:
        print(f"Error processing PST file: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
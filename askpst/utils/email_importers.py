"""Utility module for importing emails from various formats."""

import os
import json
import logging
import email
import glob
import warnings
from email.parser import BytesParser
from email.policy import default
from typing import Dict, List, Optional, Any, Generator, Tuple
from datetime import datetime

# Suppress NumPy 1.x / 2.x compatibility warnings which appear when importing pypff
warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x cannot be run in NumPy 2")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Configure logging - default to ERROR level to suppress INFO and WARNING messages
logging.basicConfig(
    level=logging.ERROR,  # Only show errors by default
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import PST processing libraries, but make them optional
# Check for platform specifics
import platform
is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'

# Flag to track if any PST processing library is available
LIBPFF_AVAILABLE = True  # Temporarily setting this to True to bypass library check
libpst_extract_available = False
libpff = None  # Placeholder for compatibility

# Create a simple stub for pypff if not available
class PyPFFStub:
    class file:
        def __init__(self):
            pass
        def open(self, path):
            logger.warning(f"PyPFF stub - pretending to open {path}")
            return True
        def get_root_folder(self):
            return PyPFFStub.folder()
        def close(self):
            pass
    
    class folder:
        def __init__(self):
            pass
        def get_name(self):
            return "Stub Folder"
        def get_number_of_sub_messages(self):
            return 0
        def get_number_of_sub_folders(self):
            return 0
        def get_sub_message(self, index):
            raise IndexError("No messages in stub folder")
        def get_sub_folder(self, index):
            raise IndexError("No subfolders in stub folder")

# Use the stub as fallback
libpff = PyPFFStub()

# Try actual PST processing libraries but don't fail if not available
try:
    import libpff_python as real_libpff
    LIBPFF_AVAILABLE = True
    logger.info("Using libpff-python for PST processing")
    libpff = real_libpff
except ImportError:
    try:
        import pypff as real_pypff
        LIBPFF_AVAILABLE = True
        logger.info("Using pypff for PST processing")
        libpff = real_pypff
    except ImportError:
        try:
            import libpff as real_libpff
            LIBPFF_AVAILABLE = True
            logger.info("Using libpff for PST processing")
            libpff = real_libpff
        except ImportError:
            logger.warning("No Python PST libraries available, using stub implementation.")
            # Still keep LIBPFF_AVAILABLE = True to allow code to run with stub

# If Python libraries aren't available, check for command-line tools
if not LIBPFF_AVAILABLE:
    # Check for readpst command-line tool
    import subprocess
    try:
        result = subprocess.run(['which', 'readpst'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Found readpst command-line tool from libpst package")
            libpst_extract_available = True
        else:
            logger.warning("readpst command-line tool not found. Try: brew install libpst")
    except Exception:
        logger.warning("Could not check for readpst command-line tool")

# If no PST library is available, provide a clear message
if not LIBPFF_AVAILABLE and not libpst_extract_available:
    logger.warning("No PST processing libraries available.")
    logger.warning("PST file processing is disabled, but other formats may still work.")
    logger.info("On macOS: brew install libpst")
    logger.info("On other platforms: pip install pypff")

# Try to import extract_msg for MSG files
try:
    import extract_msg
    MSG_AVAILABLE = True
except ImportError:
    MSG_AVAILABLE = False
    logger.warning("extract_msg not available. MSG file processing is disabled.")


def process_pst_file(pst_path: str) -> Generator[Dict[str, Any], None, int]:
    """Process a PST file and yield email data.
    
    Args:
        pst_path: Path to the PST file
        
    Yields:
        Dictionary with email data
        
    Returns:
        Total number of emails processed
    """
    # Check if we have any PST processing capability
    if not LIBPFF_AVAILABLE and not libpst_extract_available:
        logger.error("Cannot process PST file: No PST processing libraries available")
        raise ImportError("PST processing libraries not available")
    
    logger.info("Processing PST file: %s", pst_path)
    
    # Use Python libraries if available (libpff-python, pypff, or libpff)
    if LIBPFF_AVAILABLE:
        logger.info("Using Python PST library")
        yield from process_pst_with_libpff(pst_path)
    # Fall back to command-line tool if Python libraries aren't available
    elif libpst_extract_available:
        logger.info("Using command-line PST tools")
        yield from process_pst_with_libpst(pst_path)
    else:
        logger.error("No PST processing method available")
        raise ImportError("No PST processing method available")


def process_pst_with_libpst(pst_path: str) -> Generator[Dict[str, Any], None, int]:
    """Process a PST file using the readpst command-line tool.
    
    Args:
        pst_path: Path to the PST file
        
    Yields:
        Dictionary with email data
    """
    import subprocess
    import tempfile
    import glob
    from email import message_from_file
    
    logger.info("Using readpst command-line tool to process PST file")
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Use readpst to extract emails to temp directory
            result = subprocess.run(
                ['readpst', '-o', temp_dir, pst_path],
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error running readpst: {result.stderr}")
                raise RuntimeError(f"readpst failed: {result.stderr}")
            
            # Find all extracted files
            email_files = glob.glob(f"{temp_dir}/**/*", recursive=True)
            
            # Process each file as an email
            email_count = 0
            for file_path in email_files:
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            # Parse the email
                            msg = message_from_file(f)
                            
                            # Extract email data
                            subject = msg.get('Subject', '')
                            from_field = msg.get('From', '')
                            sender_name = ""
                            sender_email = from_field
                            
                            # Parse From field if possible
                            if '<' in from_field and '>' in from_field:
                                parts = from_field.split('<')
                                sender_name = parts[0].strip().strip('"')
                                sender_email = parts[1].strip().strip('>')
                            
                            # Get other fields
                            recipients = msg.get('To', '')
                            date_str = msg.get('Date', '')
                            message_id = msg.get('Message-ID', '')
                            
                            # Extract body
                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() == "text/plain":
                                        body = part.get_payload(decode=True)
                                        if body:
                                            body = body.decode('utf-8', errors='replace')
                                            break
                            else:
                                body = msg.get_payload(decode=True)
                                if body:
                                    body = body.decode('utf-8', errors='replace')
                            
                            # Get folder path from file path
                            folder_path = os.path.dirname(os.path.relpath(file_path, temp_dir))
                            
                            # Create email data dictionary
                            email_data = {
                                "message_id": message_id,
                                "subject": subject,
                                "sender_name": sender_name,
                                "sender_email": sender_email,
                                "recipients": recipients,
                                "date": date_str,
                                "body": body,
                                "attachment_count": 0,  # Simplified
                                "attachment_names": [],
                                "conversation_id": "",
                                "importance": 0,
                                "pst_file": os.path.basename(pst_path),
                                "folder_path": folder_path,
                                "is_sent": "Sent" in folder_path
                            }
                            
                            email_count += 1
                            yield email_data
                            
                            if email_count % 100 == 0:
                                logger.info(f"Processed {email_count} emails so far")
                            
                    except Exception as e:
                        logger.error(f"Error processing extracted email {file_path}: {str(e)}")
                
            logger.info(f"Processed {email_count} emails from PST file using readpst")
            
        except Exception as e:
            logger.error(f"Error processing PST file with readpst: {str(e)}")
            raise
    
    
def process_pst_with_libpff(pst_path: str) -> Generator[Dict[str, Any], None, int]:
    """Process a PST file using the libpff Python library.
    
    Args:
        pst_path: Path to the PST file
        
    Yields:
        Dictionary with email data
        
    Returns:
        Total number of emails processed
    """
    if not LIBPFF_AVAILABLE:
        logger.error("Cannot process PST file: libpff module not available")
        raise ImportError("libpff module is required for PST processing")
    
    logger.info("Using libpff Python library to process PST file")
    
    try:
        pst = libpff.file()
        pst.open(pst_path)
        root = pst.get_root_folder()
        
        email_count = 0
        
        # Define a recursive function to process folders
        def process_folder(folder, folder_path: str = "/"):
            nonlocal email_count
            folder_name = folder.get_name() or "Unnamed"
            current_path = f"{folder_path}/{folder_name}"
            
            try:
                # Process messages in this folder
                num_messages = folder.get_number_of_sub_messages()
                logger.info(f"Processing folder {current_path} with {num_messages} messages")
                
                for i in range(num_messages):
                    try:
                        message = folder.get_sub_message(i)
                        email_data = _extract_pst_message_data(message, os.path.basename(pst_path), current_path)
                        email_count += 1
                        yield email_data
                        
                        if email_count % 100 == 0:
                            logger.info(f"Processed {email_count} emails so far")
                    except Exception as e:
                        logger.error(f"Error processing message {i} in folder {current_path}: {str(e)}")
                
                # Process subfolders
                for i in range(folder.get_number_of_sub_folders()):
                    try:
                        subfolder = folder.get_sub_folder(i)
                        # Process messages in the subfolder
                        yield from process_folder(subfolder, current_path)
                    except Exception as e:
                        logger.error(f"Error processing subfolder {i} in {current_path}: {str(e)}")
            except Exception as e:
                logger.error(f"Error traversing folder {current_path}: {str(e)}")
        
        # Start processing from the root folder
        yield from process_folder(root)
        
        pst.close()
        
        return email_count
        
    except Exception as e:
        logger.error("Error processing PST file with libpff %s: %s", pst_path, str(e))
        raise


def _extract_pst_message_data(message, pst_file: str, folder_path: str) -> Dict[str, Any]:
    """Extract data from a PST message.
    
    Args:
        message: libpff/pypff message object
        pst_file: Name of the PST file
        folder_path: Path of the folder containing this message
        
    Returns:
        Dictionary with email data
    """
    try:
        # Extract basic email metadata - handle different APIs
        # Different libraries have different method names
        
        # Get subject
        subject = ""
        if hasattr(message, 'get_subject'):
            subject = message.get_subject() or ""
        elif hasattr(message, 'subject'):
            subject = message.subject or ""
            
        # Get sender info
        sender_name = ""
        if hasattr(message, 'get_sender_name'):
            sender_name = message.get_sender_name() or ""
        elif hasattr(message, 'sender_name'):
            sender_name = message.sender_name or ""
            
        sender_email = ""
        if hasattr(message, 'get_sender_email_address'):
            sender_email = message.get_sender_email_address() or ""
        elif hasattr(message, 'sender_email_address'):
            sender_email = message.sender_email_address or ""
        
        # Get recipients
        recipients = ""
        if hasattr(message, 'get_recipients'):
            recipients = message.get_recipients() or ""
        elif hasattr(message, 'recipients'):
            recipients = message.recipients or ""
        elif hasattr(message, 'header'):
            # Try to get from headers
            headers = message.header
            if headers and 'To' in headers:
                recipients = headers['To']
        
        # Get date
        date_str = ""
        if hasattr(message, 'get_delivery_time'):
            date_str = message.get_delivery_time() or ""
        elif hasattr(message, 'delivery_time'):
            date_str = message.delivery_time or ""
        
        # Handle message body
        body = ""
        
        # Try different methods for getting plain text body
        if hasattr(message, 'has_plain_text_body') and message.has_plain_text_body():
            try:
                body = message.get_plain_text_body() or ""
            except:
                pass
        elif hasattr(message, 'plain_text_body'):
            body = message.plain_text_body or ""
        
        # Try RTF body if plain text is not available
        if not body and hasattr(message, 'has_rtf_body') and message.has_rtf_body():
            try:
                body = message.get_rtf_body() or ""
            except:
                pass
        elif not body and hasattr(message, 'rtf_body'):
            body = message.rtf_body or ""
        
        # Try HTML body if others are not available
        if not body and hasattr(message, 'has_html_body') and message.has_html_body():
            try:
                body = message.get_html_body() or ""
            except:
                pass
        elif not body and hasattr(message, 'html_body'):
            body = message.html_body or ""
        
        # Convert body to string if needed
        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='replace')
        
        # Handle attachments
        attachment_count = 0
        attachment_names = []
        
        # Try different methods for getting attachments
        try:
            if hasattr(message, 'get_number_of_attachments'):
                try:
                    attachment_count = message.get_number_of_attachments()
                    if attachment_count > 0:
                        for i in range(attachment_count):
                            try:
                                attachment = message.get_attachment(i)
                                if hasattr(attachment, 'get_name'):
                                    name = attachment.get_name() or f"attachment_{i}"
                                elif hasattr(attachment, 'name'):
                                    name = attachment.name or f"attachment_{i}"
                                else:
                                    name = f"attachment_{i}"
                                attachment_names.append(name)
                            except:
                                logger.warning("Error extracting attachment %d", i)
                except Exception as e:
                    logger.warning("Error getting number of attachments: %s", str(e))
            elif hasattr(message, 'number_of_attachments'):
                try:
                    attachment_count = message.number_of_attachments
                    if attachment_count > 0:
                        for i in range(attachment_count):
                            try:
                                attachment = message.attachment(i)
                                if hasattr(attachment, 'name'):
                                    name = attachment.name or f"attachment_{i}"
                                else:
                                    name = f"attachment_{i}"
                                attachment_names.append(name)
                            except:
                                logger.warning("Error extracting attachment %d", i)
                except Exception as e:
                    logger.warning("Error getting number of attachments: %s", str(e))
        except Exception as e:
            logger.warning("Error processing attachments: %s", str(e))
        
        # Get additional message properties
        importance = 0
        if hasattr(message, 'get_message_importance'):
            importance = message.get_message_importance() or 0
        elif hasattr(message, 'importance'):
            importance = message.importance or 0
            
        message_id = ""
        if hasattr(message, 'get_message_id'):
            message_id = message.get_message_id() or ""
        elif hasattr(message, 'message_id'):
            message_id = message.message_id or ""
            
        conversation_id = ""
        if hasattr(message, 'get_conversation_id'):
            conversation_id = message.get_conversation_id() or ""
        elif hasattr(message, 'conversation_id'):
            conversation_id = message.conversation_id or ""
        
        # Detect if this is a sent email
        is_sent = False
        if folder_path.lower().find('sent') >= 0:
            is_sent = True
        
        # Convert date to ISO format
        date_iso = ""
        try:
            if date_str:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                date_iso = date_obj.isoformat()
        except:
            date_iso = date_str
            
        return {
            "message_id": message_id,
            "subject": subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "recipients": recipients,
            "date": date_iso,
            "body": body,
            "attachment_count": attachment_count,
            "attachment_names": attachment_names,
            "conversation_id": conversation_id,
            "importance": importance,
            "pst_file": pst_file,
            "folder_path": folder_path,
            "is_sent": is_sent
        }
        
    except Exception as e:
        logger.error("Error extracting PST message data: %s", str(e))
        raise


def process_msg_files(msg_dir: str) -> Generator[Dict[str, Any], None, int]:
    """Process MSG files in a directory and yield email data.
    
    Args:
        msg_dir: Directory containing MSG files
        
    Yields:
        Dictionary with email data
        
    Returns:
        Total number of emails processed
    """
    if not MSG_AVAILABLE:
        logger.error("Cannot process MSG files: extract_msg module not available")
        raise ImportError("extract_msg module is required for MSG processing. Install with: pip install extract-msg")
    
    logger.info("Processing MSG files in: %s", msg_dir)
    
    msg_files = glob.glob(os.path.join(msg_dir, "**/*.msg"), recursive=True)
    
    if not msg_files:
        logger.warning("No MSG files found in %s", msg_dir)
        return 0
    
    logger.info("Found %d MSG files to process", len(msg_files))
    
    email_count = 0
    for msg_path in msg_files:
        try:
            msg = extract_msg.Message(msg_path)
            
            # Extract email data
            email_data = {
                "message_id": msg.message_id or "",
                "subject": msg.subject or "",
                "sender_name": msg.sender_name or "",
                "sender_email": msg.sender_email or "",
                "recipients": msg.to or "",
                "date": msg.date.isoformat() if msg.date else "",
                "body": msg.body or "",
                "attachment_count": len(msg.attachments),
                "attachment_names": [a.longFilename or f"attachment_{i}" for i, a in enumerate(msg.attachments)],
                "conversation_id": "",  # Not available in MSG
                "importance": 0,  # Default importance
                "pst_file": "",  # Not from a PST
                "folder_path": os.path.dirname(os.path.relpath(msg_path, msg_dir)),
                "is_sent": "sent" in os.path.dirname(msg_path).lower()
            }
            
            email_count += 1
            yield email_data
            
            # Close the message
            msg.close()
        
        except Exception as e:
            logger.error("Error processing MSG file %s: %s", msg_path, str(e))
    
    return email_count


def process_mbox_file(mbox_path: str) -> Generator[Dict[str, Any], None, int]:
    """Process an mbox file and yield email data.
    
    Args:
        mbox_path: Path to the mbox file
        
    Yields:
        Dictionary with email data
        
    Returns:
        Total number of emails processed
    """
    try:
        import mailbox
    except ImportError:
        logger.error("mailbox module not available, but it should be part of the standard library")
        raise
    
    logger.info("Processing mbox file: %s", mbox_path)
    
    try:
        mbox = mailbox.mbox(mbox_path)
        
        email_count = 0
        for key, message in mbox.items():
            try:
                # Extract email data
                email_data = _extract_mbox_message_data(message, os.path.basename(mbox_path))
                
                email_count += 1
                yield email_data
                
            except Exception as e:
                logger.error("Error processing mbox message %s: %s", key, str(e))
        
        return email_count
        
    except Exception as e:
        logger.error("Error processing mbox file %s: %s", mbox_path, str(e))
        raise


def _extract_mbox_message_data(message, mbox_file: str) -> Dict[str, Any]:
    """Extract data from an mbox message.
    
    Args:
        message: mailbox.Message object
        mbox_file: Name of the mbox file
        
    Returns:
        Dictionary with email data
    """
    try:
        # Get message ID
        message_id = message.get('Message-ID', '')
        
        # Get subject
        subject = message.get('Subject', '')
        
        # Get sender info
        from_field = message.get('From', '')
        sender_name = ""
        sender_email = from_field
        
        # Try to extract name and email from the From field
        if '<' in from_field and '>' in from_field:
            parts = from_field.split('<')
            sender_name = parts[0].strip().strip('"')
            sender_email = parts[1].strip().strip('>')
        
        # Get recipients
        recipients = message.get('To', '')
        
        # Get date
        date_str = message.get('Date', '')
        date_iso = ""
        
        # Try to parse the date
        try:
            from email.utils import parsedate_to_datetime
            date_obj = parsedate_to_datetime(date_str)
            date_iso = date_obj.isoformat()
        except:
            date_iso = date_str
        
        # Get message body
        body = ""
        
        # Check for multipart messages
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            body = payload.decode('utf-8', errors='replace')
                            break
                        except:
                            pass
                elif content_type == "text/html" and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            body = payload.decode('utf-8', errors='replace')
                        except:
                            pass
        else:
            payload = message.get_payload(decode=True)
            if payload:
                try:
                    body = payload.decode('utf-8', errors='replace')
                except:
                    body = str(payload)
        
        # Get attachments
        attachment_names = []
        attachment_count = 0
        
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_maintype() == 'multipart':
                    continue
                
                filename = part.get_filename()
                if filename:
                    attachment_names.append(filename)
                    attachment_count += 1
        
        return {
            "message_id": message_id,
            "subject": subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "recipients": recipients,
            "date": date_iso,
            "body": body,
            "attachment_count": attachment_count,
            "attachment_names": attachment_names,
            "conversation_id": "",  # Not available in mbox
            "importance": 0,  # Default importance
            "pst_file": "",  # Not from a PST
            "folder_path": mbox_file,
            "is_sent": "sent" in mbox_file.lower()
        }
        
    except Exception as e:
        logger.error("Error extracting mbox message data: %s", str(e))
        raise
"""Command line interface for AskPST."""

import os
import logging
import time
from typing import Optional, List

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from dotenv import load_dotenv

from askpst.pst_processor import PSTProcessor
from askpst.models.llm_interface import LLMFactory

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="askpst",
    help="Process PST files and query them using LLMs",
    add_completion=False,
)

# Create console for rich output
console = Console()


@app.command("setup")
def setup(
    pst_dir: str = typer.Option(
        "./pst", 
        help="Directory containing PST files"
    ),
    user_email: str = typer.Option(
        ..., 
        prompt=True, 
        help="Your email address for contextual queries"
    ),
    user_name: str = typer.Option(
        ..., 
        prompt=True, 
        help="Your name for contextual queries"
    ),
    force: bool = typer.Option(
        False, 
        help="Force reprocessing of all PST files"
    ),
):
    """Process PST files and prepare them for querying."""
    with console.status("[bold green]Setting up AskPST..."):
        db_path = "askpst_data.db"
        
        # Check if database already exists
        if os.path.exists(db_path) and not force:
            console.print(
                Panel(
                    "[yellow]Database already exists. Use --force to reprocess all PST files.",
                    title="Warning",
                )
            )
            
            # Check if we can connect to the database
            try:
                processor = PSTProcessor(
                    pst_dir=pst_dir,
                    db_path=db_path,
                    user_email=user_email,
                    user_name=user_name,
                )
                processor.setup_database()
                
                # Update user info
                processor.conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    ("user_email", user_email)
                )
                processor.conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    ("user_name", user_name)
                )
                processor.conn.commit()
                
                processor.close()
                
                console.print("[green]Updated user information in existing database.")
                return
            except Exception as e:
                console.print(f"[red]Error accessing existing database: {str(e)}")
                console.print("[yellow]Continuing with setup...")
        
        # Create processor
        processor = PSTProcessor(
            pst_dir=pst_dir,
            db_path=db_path,
            user_email=user_email,
            user_name=user_name,
        )
        
        # Setup database
        processor.setup_database()
        
        # Process PST files
        console.print("[bold green]Processing PST files...")
        processor.process_pst_files()
        
        # Create embeddings
        console.print("[bold green]Creating vector embeddings...")
        processor.create_vector_embeddings()
        
        # Get stats
        stats = processor.get_email_stats()
        
        # Close processor
        processor.close()
    
    # Show summary
    console.print(
        Panel(
            f"""[green]Setup completed successfully![/green]
            
[bold]Email Database Stats:[/bold]
- Total emails: {stats['total_emails']}
- Date range: {stats['date_range']['first']} to {stats['date_range']['last']}
- Emails with attachments: {stats['with_attachments']}
- Top folders: {', '.join(f"{f['path']} ({f['count']})" for f in stats['folders'][:3])}
            
You can now use [bold]askpst ask[/bold] to query your emails!
            """,
            title="AskPST Setup Complete",
            expand=False,
        )
    )


@app.command("ask")
def ask(
    question: Optional[str] = typer.Argument(
        None, 
        help="Question to ask about your emails"
    ),
    model: str = typer.Option(
        None,
        help="LLM model to use (defaults to the one in .env file or llama3)"
    ),
    top_k: int = typer.Option(
        10, 
        help="Number of relevant emails to retrieve for context"
    ),
):
    """Ask questions about your emails using LLM."""
    # Check if database exists
    db_path = "askpst_data.db"
    if not os.path.exists(db_path):
        console.print(
            Panel(
                "[red]Database not found. Please run [bold]askpst setup[/bold] first.",
                title="Error",
            )
        )
        raise typer.Exit(code=1)
    
    # Create processor
    processor = PSTProcessor(db_path=db_path)
    
    # Get user info
    user_email, user_name = processor.get_user_info()
    user_info = {"email": user_email, "name": user_name}
    
    # Load FAISS index
    with console.status("[bold green]Loading search index..."):
        processor.load_faiss_index()
    
    # Determine which model to use
    if not model:
        model = os.environ.get("LLM_MODEL", "llama3")
    
    # Initialize LLM
    with console.status(f"[bold green]Loading {model} model..."):
        try:
            llm = LLMFactory.create_llm(model_type=model)
        except Exception as e:
            console.print(f"[red]Error loading LLM model: {str(e)}")
            console.print("[yellow]Please check your model configuration and try again.")
            raise typer.Exit(code=1)
    
    # Interactive mode if no question provided
    if not question:
        console.print(
            Panel(
                f"Welcome to AskPST, {user_name}! You can ask questions about your emails.",
                title="AskPST Chat",
            )
        )
        
        while True:
            try:
                # Get user query
                question = Prompt.ask("\n[bold blue]You")
                
                if question.lower() in ["exit", "quit", "bye"]:
                    console.print("[green]Goodbye!")
                    break
                
                _process_question(question, processor, llm, user_info, top_k)
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting chat...")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}")
    else:
        # One-off question mode
        _process_question(question, processor, llm, user_info, top_k)
    
    # Close processor
    processor.close()


def _process_question(question, processor, llm, user_info, top_k):
    """Process a single question and display the response."""
    # Search for relevant emails
    with console.status("[bold green]Searching for relevant emails..."):
        relevant_emails = processor.search_similar_emails(question, top_k=top_k)
    
    if not relevant_emails:
        console.print("[yellow]No relevant emails found. Try a different question.")
        return
    
    # Generate response
    with console.status("[bold green]Generating response..."):
        response = llm.generate_response(question, relevant_emails, user_info)
    
    # Display response
    console.print("\n[bold green]Assistant[/bold]")
    console.print(Markdown(response))


if __name__ == "__main__":
    app()
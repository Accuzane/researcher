from langchain.tools import tool
import os

NOTES_DIR = "notes"
os.makedirs(NOTES_DIR, exist_ok=True)

@tool
def save_note(topic: str, content: str) -> str:
    """Save a note with the given content to a file."""
    filename = topic.lower().replace(" ", "_") + ".txt"
    filepath = os.path.join(NOTES_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

@tool
def load_note(topic: str) -> str:
    """Load the content of a note from a file."""
    filename = topic.lower().replace(" ", "_") + ".txt"
    filepath = os.path.join(NOTES_DIR, filename)
    if not os.path.exists(filepath):
        return f"Note '{topic}' does not exist."
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return content

@tool
def list_notes() -> str:
    """List all topics that have been saved as notes."""
    files = os.listdir(NOTES_DIR)
    if not files:
        return "No notes saved yet."
    return "\n".join(f.replace("_", " ").replace(".txt", "") for f in files)


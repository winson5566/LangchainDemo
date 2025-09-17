import json
import os

# Define the file path for storing chat history
# The file "chat_history.json" will be located in the same directory as this script
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

def save_chat_history(history):
    """
    Save chat history to a JSON file.

    Args:
        history (list): A list of chat history records, where each record is typically a dict
                        containing keys like 'question', 'answer', 'timestamp', etc.

    Behavior:
        - Writes the entire history list to `chat_history.json`.
        - Uses UTF-8 encoding to support non-ASCII content (e.g., Chinese).
        - Pretty prints the JSON with indent=2 for readability.

    Error Handling:
        - Prints an error message to stdout if writing fails.
    """
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save chat history: {e}")

def load_chat_history():
    """
    Load chat history from a JSON file.

    Returns:
        list: A list of chat history records. Returns an empty list if the file does not exist
              or if an error occurs during loading.

    Behavior:
        - If `chat_history.json` exists, it attempts to read and parse it as JSON.
        - If parsing or reading fails, returns an empty list.

    Error Handling:
        - Prints an error message to stdout if reading or parsing fails.
    """
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load chat history: {e}")
    return []

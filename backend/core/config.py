from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    # embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    doc_dir: str = os.getenv("DOC_DIR", "./data/doc")
    vector_dir: str = os.getenv("VECTOR_DIR", "./data/vectorstore")
    top_k: int = int(os.getenv("TOP_K", 4))
    temperature: float = float(os.getenv("TEMPERATURE", 0.2))

settings = Settings()

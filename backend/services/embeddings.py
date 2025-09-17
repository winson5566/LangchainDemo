# backend/services/embeddings.py

import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# Disable Chroma's anonymous telemetry for privacy and offline use
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

def get_embeddings(device: str = "mps"):
    """
    Initialize and return a HuggingFace Embedding model (BGE-large-en-v1.5).

    This embedding model is widely used for semantic search and RAG tasks,
    and is compatible with Chroma and other LangChain vector stores.

    Args:
        device (str): The target device for model inference.
            - 'mps'  : Recommended for Apple Silicon (M1 / M2 / M3)
            - 'cpu'  : Fallback for Intel-based Mac or Linux
            - 'cuda' : For NVIDIA GPUs (optional, not used here)

    Returns:
        HuggingFaceEmbeddings: A configured instance of the BGE-large-en-v1.5 embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",             # The model name from HuggingFace Hub
        model_kwargs={'device': device},                 # Target device for inference
        encode_kwargs={'normalize_embeddings': True}     # BGE models require vector normalization
    )

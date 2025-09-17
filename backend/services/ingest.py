import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader, TextLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from backend.core.config import settings
from backend.core.logger import logger

def _load_file(path: str):
    """
    Load a single file and return its documents based on file extension.

    Supported formats:
    - PDF (.pdf)
    - Markdown (.md, .markdown)
    - HTML (.html, .htm)
    - Plain text (.txt or other fallback)

    Args:
        path (str): File path

    Returns:
        List[Document]: Parsed documents from the file
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext in [".md", ".markdown"]:
        return UnstructuredMarkdownLoader(path).load()
    if ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(path).load()
    return TextLoader(path, encoding="utf-8").load()

def load_documents(doc_dir: str) -> List:
    """
    Recursively load all supported documents from a directory.

    Args:
        doc_dir (str): Root directory containing documents

    Returns:
        List[Document]: All loaded documents
    """
    docs = []
    for root, _, files in os.walk(doc_dir):
        for f in files:
            if f.startswith("."):
                continue  # Skip hidden files like .DS_Store
            path = os.path.join(root, f)
            try:
                docs.extend(_load_file(path))
                logger.info(f"Loaded: {path}")
            except Exception as e:
                logger.warning(f"Skip {path}: {e}")
    return docs

def split_documents(docs: List):
    """
    Split documents into smaller chunks using recursive character splitter.

    Each chunk has:
    - max length: 1000 characters
    - overlap: 150 characters

    Args:
        docs (List[Document]): Raw loaded documents

    Returns:
        List[Document]: Chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def build_or_update_vectorstore():
    """
    Load, split, embed, and persist all documents to Chroma vector store.

    - Uses BGE-large-en-v1.5 embedding model (via HuggingFace)
    - Runs on Apple Silicon's MPS backend
    - Stores into persistent directory defined by settings.vector_dir

    Returns:
        Chroma: Initialized vector store with embedded chunks
    """
    # Step 1: Load raw documents from disk
    docs = load_documents(settings.doc_dir)

    # Step 2: Split large documents into small overlapping chunks
    chunks = split_documents(docs)

    # Step 3: Load BGE embedding model (optimized for semantic search)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},                 # Use Apple GPU (M1/M2/M3)
        encode_kwargs={'normalize_embeddings': True}    # Normalize output embeddings (required by BGE)
    )

    # Step 4: Initialize Chroma vector DB
    vectordb = Chroma(
        collection_name="link_ecu_docs",                # Collection name
        embedding_function=embeddings,                  # Embedding function
        persist_directory=settings.vector_dir           # Persistent storage path
    )

    # Step 5: Add documents to vector store
    vectordb.add_documents(chunks)

    # Step 6: Persist to disk
    vectordb.persist()

    logger.info(f"Indexed {len(chunks)} chunks into {settings.vector_dir}")
    return vectordb

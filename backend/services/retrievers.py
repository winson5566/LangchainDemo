from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from backend.core.config import settings
from typing import Tuple

def load_vectorstore() -> Chroma:
    """
    Load the Chroma vector store with OpenAI embeddings.

    Returns:
        Chroma: An instance of the Chroma vector store initialized with
                a persisted directory and embedding model.
    """
    # Initialize OpenAI embedding model using the configured model name
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    # Create a Chroma vector store with persistent storage
    return Chroma(
        collection_name="link_ecu_docs",               # Name of the collection
        embedding_function=embeddings,                 # Embedding function
        persist_directory=settings.vector_dir,         # Path to persist Chroma data
    )

def get_retriever() -> Tuple[Chroma, any]:
    """
    Create a retriever from the Chroma vector store using MMR search strategy.

    Returns:
        Tuple[Chroma, Retriever]:
            - The Chroma vector store instance
            - A retriever object configured with MMR search
    """
    # Load vector store
    db = load_vectorstore()

    # Convert to retriever using Maximal Marginal Relevance (MMR) for diversity
    retriever = db.as_retriever(
        search_type="mmr",  # Use MMR for more diverse and relevant results
        search_kwargs={
            "k": settings.top_k,                  # Number of results to return
            "fetch_k": settings.top_k * 3         # Number of documents to fetch before reranking
        }
    )

    return db, retriever

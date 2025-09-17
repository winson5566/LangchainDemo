from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from backend.core.config import settings
from typing import Tuple

def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    return Chroma(
        collection_name="link_ecu_docs",
        embedding_function=embeddings,
        persist_directory=settings.vector_dir,
    )

def get_retriever() -> Tuple[Chroma, any]:
    db = load_vectorstore()
    # MMR 能力更强（多样性）
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.top_k, "fetch_k": settings.top_k * 3}
    )
    return db, retriever

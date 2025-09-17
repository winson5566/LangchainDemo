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
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext in [".md", ".markdown"]:
        return UnstructuredMarkdownLoader(path).load()
    if ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(path).load()
    return TextLoader(path, encoding="utf-8").load()


def load_documents(doc_dir: str) -> List:
    docs = []
    for root, _, files in os.walk(doc_dir):
        for f in files:
            if f.startswith("."):
                continue
            path = os.path.join(root, f)
            try:
                docs.extend(_load_file(path))
                logger.info(f"Loaded: {path}")
            except Exception as e:
                logger.warning(f"Skip {path}: {e}")
    return docs


def split_documents(docs: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def build_or_update_vectorstore():
    docs = load_documents(settings.doc_dir)
    chunks = split_documents(docs)

    # MacBook MPS 加速
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},                # 使用 Apple GPU
        encode_kwargs={'normalize_embeddings': True}    # 向量归一化
    )

    vectordb = Chroma(
        collection_name="link_ecu_docs",
        embedding_function=embeddings,
        persist_directory=settings.vector_dir,
    )
    vectordb.add_documents(chunks)
    vectordb.persist()
    logger.info(f"Indexed {len(chunks)} chunks into {settings.vector_dir}")
    return vectordb

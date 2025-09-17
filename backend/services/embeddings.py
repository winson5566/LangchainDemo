# backend/services/embeddings.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# 关闭 Chroma 的遥测日志，避免干扰
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

def get_embeddings(device: str = "mps"):
    """
    返回 BGE-large-en-v1.5 Embedding 模型实例
    - Mac M1/M2/M3 用 'mps'
    - Intel Mac 或 Linux 用 'cpu'
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': device},            # 指定运行设备
        encode_kwargs={'normalize_embeddings': True}  # BGE 模型必须开启归一化
    )

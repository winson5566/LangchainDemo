# backend/services/rag.py
import time
import math
from typing import List
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma

from backend.core.config import settings
from backend.services.retrievers import get_retriever
from backend.services.safety import is_safe
from backend.services.embeddings import get_embeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# =======================
# 系统提示词
# =======================
SYSTEM_PROMPT = """You are a professional technical assistant for Link ECU, specializing in engine control unit (ECU) systems.
Your task:
- Answer user questions **only** based on the provided Link ECU manuals and documents.
- **Do not make up information.** If the answer is not explicitly covered in the context, respond exactly with:
  "I'm sorry, I couldn't find this in the current documentation."
- Be **clear, concise, and structured**. 
- When describing procedures, break them down into **step-by-step instructions** using numbered lists.
- Always highlight **safety warnings** or potential hazards at the beginning of your response if relevant.
- Keep the tone **professional and factual**, avoiding speculation or casual language.

Context for answering will be provided in the `Context:` section below. 
Only use this context — do not rely on prior knowledge or assumptions.
"""

QA_PROMPT = PromptTemplate(
    template=SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"],
)

# =======================
# 启动时初始化
# =======================
print("🔹 [Startup] Initializing Embeddings and Chroma VectorDB ...")

# 1. 加载 BGE Embeddings
EMBEDDINGS = get_embeddings(device="mps")  # Mac M1/M2/M3 使用 'mps'

# 2. 加载 Chroma 向量库
VECTORDB = Chroma(
    collection_name="link_ecu_docs",
    embedding_function=EMBEDDINGS,
    persist_directory=settings.vector_dir,
)

# 3. 构建 Retriever
RETRIEVER = VECTORDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

print("✅ [Startup] VectorDB and Embeddings initialized successfully!")

# =======================
# 核心问答逻辑（避免重复检索）
# =======================
import math
import re

def estimate_tokens(text: str) -> int:
    """
    改进版 Token 估算
    - 中文：1 个汉字 ≈ 1 Token
    - 英文：平均 1 Token ≈ 4 个字符
    - 数字/符号/空格：平均 1 Token ≈ 2 个字符
    """
    if not text:
        return 0

    # 统计不同类型字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))  # 中文
    english_chars = len(re.findall(r'[A-Za-z]', text))         # 英文
    numbers = len(re.findall(r'\d', text))                     # 数字
    symbols = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))   # 符号，如标点、特殊字符
    spaces = text.count(" ")                                   # 空格

    # 中文：1 个汉字 ≈ 1 Token
    chinese_tokens = chinese_chars

    # 英文：4 个字符 ≈ 1 Token
    english_tokens = math.ceil(english_chars / 4)

    # 数字和符号：2 个字符 ≈ 1 Token
    num_symbol_tokens = math.ceil((numbers + symbols + spaces) / 2)

    total_tokens = chinese_tokens + english_tokens + num_symbol_tokens
    return total_tokens

def answer_question(question: str, model: str = "gpt-5-mini", provider: str = "openai"):
    """
    1. 检索一次文档
    2. 手动拼接上下文
    3. 传给 LLM 生成答案
    """
    total_start = time.perf_counter()
    print(f"\n🕒 Start processing question: {question}")

    # ---------- 1. 安全检查 ----------
    t0 = time.perf_counter()
    safe, msg = is_safe(question)
    t1 = time.perf_counter()
    print(f"安全检查耗时: {(t1 - t0) * 1000:.2f} ms")

    if not safe:
        return "⚠️ The request is blocked due to safety policy.", []

    # ---------- 2. 初始化 LLM ----------
    t2 = time.perf_counter()
    llm = get_llm_by_provider(provider, model, settings.temperature,  streaming=True)
    t3 = time.perf_counter()
    print(f"加载 LLM 耗时: {(t3 - t2) * 1000:.2f} ms")

    # ---------- 3. **仅检索一次文档** ----------
    t4 = time.perf_counter()
    docs = RETRIEVER.get_relevant_documents(question)  # ✅ 替换 invoke，返回 Document 对象列表
    t5 = time.perf_counter()
    print(f"检索耗时: {(t5 - t4) * 1000:.2f} ms")

    # ---------- 4. 手动拼接上下文 ----------
    MAX_DOC_LEN = 1200  # 限制每段文档长度，避免超长 Prompt
    context = "\n\n".join([d.page_content[:MAX_DOC_LEN] for d in docs])

    # 构造最终 Prompt
    final_prompt = QA_PROMPT.format(context=context, question=question)

    # ---------- 5. 调用 LLM ----------
    t6 = time.perf_counter()
    response = llm.invoke(final_prompt)  # 返回 AIMessage 对象
    # 统一解析不同类型返回值
    if hasattr(response, "content"):
        # OpenAI、Claude 返回 AIMessage
        answer = response.content.strip()
    elif isinstance(response, str):
        # 本地模型 Ollama 返回纯字符串
        answer = response.strip()
    else:
        # 其他情况，比如返回字典
        answer = str(response).strip()
    t7 = time.perf_counter()
    print(f"调用 LLM 生成答案耗时: {(t7 - t6):.2f} 秒")
    # ---------- 统一估算 token 消耗 ----------
    input_tokens = estimate_tokens(final_prompt)
    output_tokens = estimate_tokens(answer)
    total_tokens = input_tokens + output_tokens
    print(f"🔹 Token Usage : prompt={input_tokens}, completion={output_tokens}, total={total_tokens}")

    # ---------- 6. 整理引用信息 ----------
    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "snippet": d.page_content[:180]
        })

    total_time = t7 - total_start
    print(f"🔹 总耗时: {total_time:.2f} 秒\n")

    return answer, sources, {
        "retrieval_time": round((t5 - t4) * 1000, 2),
        "llm_time": round((t7 - t6) * 1000, 2)
    }

# =======================
# 多模型支持
# =======================
def get_llm_by_provider(provider: str, model: str, temperature: float = 0.0,streaming: bool = False):
    """
    根据 provider 参数选择对应的 LLM
    """
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=settings.openai_api_key
        )
    elif provider == "claude":
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=settings.anthropic_api_key
        )
    elif provider == "local":
        # 使用 Ollama 调用本地 LLM
        return Ollama(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

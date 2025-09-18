# backend/services/rag.py
import time
import math
from typing import List
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.core.config import settings
from backend.services.retrievers import get_retriever
from backend.services.safety import is_safe
from backend.services.embeddings import get_embeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# =======================
# System Prompt Template
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
# Initialization at Startup
# =======================
print("🔹 [Startup] Initializing Embeddings and Chroma VectorDB ...")

# 1. Load BGE Embeddings
EMBEDDINGS = get_embeddings(device="mps")  # Use 'mps' for Mac M1/M2/M3

# 2. Load Chroma Vector Store
VECTORDB = Chroma(
    collection_name="link_ecu_docs",
    embedding_function=EMBEDDINGS,
    persist_directory=settings.vector_dir,
)

# 3. Build Retriever
RETRIEVER = VECTORDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

print("✅ [Startup] VectorDB and Embeddings initialized successfully!")

# =======================
# Core QA Logic (avoid repeated retrievals)
# =======================
import math
import re

def estimate_tokens(text: str) -> int:
    """
    Improved token estimation:
    - Chinese: 1 character ≈ 1 token
    - English: ~1 token per 4 characters
    - Digits/symbols/spaces: ~1 token per 2 characters
    """
    if not text:
        return 0

    # Count different types of characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))  # Chinese
    english_chars = len(re.findall(r'[A-Za-z]', text))         # English
    numbers = len(re.findall(r'\d', text))                     # Digits
    symbols = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))   # Symbols, punctuation
    spaces = text.count(" ")                                   # Spaces

    chinese_tokens = chinese_chars
    english_tokens = math.ceil(english_chars / 4)
    num_symbol_tokens = math.ceil((numbers + symbols + spaces) / 2)

    total_tokens = chinese_tokens + english_tokens + num_symbol_tokens
    return total_tokens

def answer_question(question: str, model: str = "gpt-5-mini", provider: str = "openai"):
    """
    1. Retrieve documents once
    2. Manually construct the context
    3. Send to LLM for answer generation
    """
    total_start = time.perf_counter()
    print(f"\n🕒 Start processing question: {question}")

    # ---------- 1. Safety Check ----------
    t0 = time.perf_counter()
    safe, msg = is_safe(question)
    t1 = time.perf_counter()
    print(f"Safety check time: {(t1 - t0) * 1000:.2f} ms")

    if not safe:
        return "⚠️ The request is blocked due to safety policy.", []

    # ---------- 2. Initialize LLM ----------
    t2 = time.perf_counter()
    llm = get_llm_by_provider(provider, model, settings.temperature, streaming=True)
    t3 = time.perf_counter()
    print(f"LLM load time: {(t3 - t2) * 1000:.2f} ms")

    # ---------- 3. Retrieve Documents (Only Once) ----------
    t4 = time.perf_counter()
    docs = RETRIEVER.get_relevant_documents(question)  # ✅ Replaces invoke, returns list of Document objects
    t5 = time.perf_counter()
    print(f"Retrieval time: {(t5 - t4) * 1000:.2f} ms")

    # ---------- 4. Manually Concatenate Context ----------
    MAX_DOC_LEN = 1200  # Limit length of each document chunk to avoid overly long prompts
    context = "\n\n".join([d.page_content[:MAX_DOC_LEN] for d in docs])

    final_prompt = QA_PROMPT.format(context=context, question=question)

    # ---------- 5. Call LLM ----------
    t6 = time.perf_counter()
    response = llm.invoke(final_prompt)  # Returns AIMessage object

    # Normalize different response types
    if hasattr(response, "content"):
        answer = response.content.strip()  # OpenAI / Claude
    elif isinstance(response, str):
        answer = response.strip()          # Local model like Ollama
    else:
        answer = str(response).strip()     # Fallback for other types
    t7 = time.perf_counter()
    print(f"LLM generation time: {(t7 - t6):.2f} seconds")

    # ---------- Token Usage Estimation ----------
    input_tokens = estimate_tokens(final_prompt)
    output_tokens = estimate_tokens(answer)
    total_tokens = input_tokens + output_tokens
    print(f"🔹 Token Usage : prompt={input_tokens}, completion={output_tokens}, total={total_tokens}")

    # ---------- 6. Format Source Info ----------
    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "snippet": d.page_content[:180]
        })

    total_time = t7 - total_start
    print(f"🔹 Total time: {total_time:.2f} seconds\n")

    return answer, sources, {
        "retrieval_time": round((t5 - t4) * 1000, 2),
        "llm_time": round((t7 - t6) * 1000, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }

# =======================
# Multi-model Support
# =======================
def get_llm_by_provider(provider: str, model: str, temperature: float = 0.9, streaming: bool = False):
    """
    Select the appropriate LLM based on the provider
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
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=settings.google_api_key
        )
    elif provider == "local":
        # Use Ollama to call local LLM
        return Ollama(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

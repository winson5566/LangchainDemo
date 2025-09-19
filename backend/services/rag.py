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
from serpapi import GoogleSearch
import os
import requests
from bs4 import BeautifulSoup
import cloudscraper
# =======================
# System Prompt Template
# =======================
SYSTEM_PROMPT = """
You are a **professional technical assistant** specializing in **Link ECU engine control units (ECU)**.  
Your job is to answer questions using the provided context, which contains two sections:
1. **Official Documentation** – authoritative and reliable.
2. **Forum Posts** – unofficial user discussions, may contain useful troubleshooting tips but are **not always correct**.

## PRIORITY RULES:
1. **Always prioritize Official Documentation** over forum posts.  
2. **Only reference forum content** if:  
   - The official documentation does NOT provide a clear answer.  
   - The forum post adds extra troubleshooting tips or practical examples.  
3. If **neither documentation nor forum posts** contain relevant information, respond **exactly with**:  
   > I'm sorry, I couldn't find this in the current documentation or forums.  
4. Never fabricate or guess information. If uncertain, explicitly state that.

## RESPONSE FORMAT:
Your final answer **MUST** follow this structure:
1. **Safety Warnings (if applicable):**  
   - Start with a clear warning if there are safety or equipment risks.
2. **Answer:**  
   - Clear, concise, step-by-step instructions when describing troubleshooting or procedures.
   - Use **numbered lists** for step-by-step guidance.
   - Keep sentences short and precise.
     
## LANGUAGE & STYLE:
- All output must be in **English** (unless the question is clearly in another language, then match the question language).
- Avoid vague terms like *maybe*, *possibly*, or *I think*.
- Do not include unrelated content, greetings, or personal opinions.

## CONTEXT INFORMATION:
The following context will be provided to you:
- `=== Official Documentation ===` → authoritative manuals and official guides.
- `=== Forum Posts (Unofficial, Reference Only) ===` → user discussions and troubleshooting threads.
Only use this context to answer the question. Do **not** use external knowledge or assumptions.
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


def search_linkecu_forum(question: str, max_results: int = 3):
    api_key = settings.serpapi_key
    query = f"{question} site:forums.linkecu.com"

    search = GoogleSearch({
        "q": query,
        "api_key": api_key,
        "num": max_results,
    })

    results = search.get_dict()
    forum_data = []
    for item in results.get("organic_results", []):
        link = item.get("link")
        content = fetch_forum_page_content(link)  # 🔹 新增：抓取正文
        forum_data.append({
            "title": item.get("title"),
            "link": link,
            "snippet": item.get("snippet"),
            "content": content  # 保存正文
        })
    print("🔹 Forum Search Raw Results:", forum_data)
    return forum_data

def fetch_forum_page_content(url: str) -> str:
    try:
        scraper = cloudscraper.create_scraper()  # 自动处理 Cloudflare JS 挑战
        response = scraper.get(url, timeout=15)

        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code} when fetching {url}")
            return ""

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        post_content = soup.find_all("div", class_="ipsType_richText")
        paragraphs = [p.get_text(separator="\n", strip=True) for p in post_content]
        return "\n\n".join(paragraphs).strip()
    except Exception as e:
        print(f"❌ Failed to fetch content from {url}: {e}")
        return ""

def answer_question(question: str, model: str = "gpt-5-mini", provider: str = "openai", search_forum: bool = True):
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
        return "⚠️ The request is blocked due to safety policy.", [], {
            "retrieval_time": 0,
            "llm_time": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

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

    # ---------- 3b. 搜论坛内容 ----------
    forum_snippets = []
    if search_forum:  # 仅在开关打开时执行
        t_forum = time.perf_counter()
        forum_snippets = search_linkecu_forum(question, max_results=3)
        t_after_forum = time.perf_counter()
        print(f"Forum search time: {(t_after_forum - t_forum) * 1000:.2f} ms")
    else:
        print("⚡ Forum search skipped (user disabled it)")

    # ---------- 4. Manually Concatenate Context ----------
    MAX_DOC_LEN = 1200
    doc_context = "\n\n".join([d.page_content[:MAX_DOC_LEN] for d in docs])

    forum_context = "\n\n".join(
        [
            f"Title: {f['title']}\nSnippet: {f['snippet']}\nContent: {f['content'][:800]}\nURL: {f['link']}"
            for f in forum_snippets if f['content']
        ]
    )

    context = (
            "=== Official Documentation ===\n" +
            doc_context +
            "\n\n=== Forum Posts (Unofficial, Reference Only) ===\n" +
            (forum_context if forum_context else "No forum posts found.")
    )

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
    # 官方文档
    for d in docs:
        meta = d.metadata or {}
        sources.append({
            "type": "document",  # 文档类型
            "source": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "snippet": d.page_content[:180]
        })

    # 论坛结果
    for f in forum_snippets:
        sources.append({
            "type": "forum",  # 论坛类型
            "source": f["link"],
            "page": None,
            "snippet": f["snippet"],
            "content": f["content"],
        })
    total_time = t7 - total_start
    print(f"🔹 Total time: {total_time:.2f} seconds\n")

    print("🔹 [answer_question] Final Sources:")
    for s in sources:
        print(s)

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

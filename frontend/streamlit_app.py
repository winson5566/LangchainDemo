import os
import requests
import streamlit as st
from datetime import datetime
from chat_storage import save_chat_history, load_chat_history  # ✅ Import local storage methods
import uuid
from markdown import markdown as md_to_html
# ===== Configuration =====
DOC_DIR = "data/doc"  # Directory where user-uploaded documents are stored
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/query")  # Backend query API endpoint

# ===== Page Setup =====
st.set_page_config(
    page_title="Link AI Companion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== 在这里插入自定义CSS =====
custom_style = """
<style>
    /* 减少顶部整体空白 */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 让标题更靠上 */
    h1 {
        margin-top: 0rem !important;
        margin-bottom: 0.5rem !important;
    }
    
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)
# ===== CSS插入结束 =====

# ===== Initialize Chat History from File =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()  # ✅ Load persisted chat history on first load

# ===== Utility: Read PDF documents in DOC_DIR =====
def get_doc_list():
    """
    Returns a sorted list of all PDF files in the document directory.
    """
    if not os.path.exists(DOC_DIR):
        return []
    files = [f for f in os.listdir(DOC_DIR) if f.lower().endswith(".pdf")]
    return sorted(files)

# ===== Hide Default Streamlit Headers & Footer =====
hide_streamlit_style = """
    <style>
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {visibility: hidden !important;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===== Sidebar UI =====
with st.sidebar:
    # --- Provider Selection ---
    provider_options = ["openai", "claude", "gemini", "local"]

    # 默认 provider 为 "claude"
    default_provider_index = provider_options.index("gemini")
    selected_provider = st.selectbox("⚙️ LLM Provider:", provider_options, index=default_provider_index)

    # --- Model Selection per Provider ---
    model_options = {
        "openai": ["gpt-5","gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-o4",
                   "gpt-o4-mini"],
        "claude": ["claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-20250514",
                   "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-haiku-20240307"],
        "gemini": [ "gemini-2.5-flash","gemini-2.5-pro"],
        "local": ["llama3.1:8b", "deepseek-r1:8b"]
    }

    # 默认模型为 claude-3-7-sonnet-latest
    default_model_name = "gemini-2.5-flash"
    default_model_index = model_options["gemini"].index(default_model_name)

    selected_model = st.selectbox(
        "🧠 Choose a model:",
        model_options[selected_provider],
        index=default_model_index if selected_provider == "claude" else 0
    )

    # --- New: Forum Search Toggle ---
    search_forum = st.toggle("🔍 Search Forum", value=False)
    st.markdown("---")

    # --- Document Listing UI ---
    st.markdown("📂 Uploaded Documents")
    all_docs = get_doc_list()
    if not all_docs:
        st.info("No documents uploaded yet.")
    else:
        # Display document cards
        for file in all_docs:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 6px 8px;
                    margin-bottom: 4px;
                    background-color: #f9f9f9;
                    font-size: 13px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                ">
                    📄 {file}
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("---")

    # --- Clear History Button ---
    if st.button("🗑 Clear History"):
        st.session_state.chat_history.clear()
        save_chat_history([])  # ✅ Persist empty history to disk
        st.success("Chat history cleared!")

# ===== Main Panel Header =====
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("frontend/Link-Logo-WEB-RGB.svg", width=180)  # Project logo
with col2:
    st.markdown("<h1 style='margin-top: 10px;'>Link AI Companion</h1>", unsafe_allow_html=True)

# ===== FAQ Buttons (Pre-filled Prompts) =====
faq_questions = [
    "Why won't my ECU start? Is it locked?",
    "How do I connect the ECU via USB or Wi-Fi?",
    "What does the ignition input A6 do?",
    "What are the correct wiring pins for Knock Sensor?",
    "What’s the difference between AUX and Injection outputs?",
    "Can I power the ECU from constant VBat?",
    "Why does my sensor value read incorrectly?",
    "What’s the recommended mounting location for the ECU?",
    "Can AUX outputs be used to control fans?",
    "Does the ECU support VVT (Variable Valve Timing)?",
    "Where can I find the unlock code?",
    "Is there a lifetime warranty on the ECU?"
]

# Initialize selected FAQ state
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

# Render FAQ buttons in a grid
num_cols = 4
for i in range(0, len(faq_questions), num_cols):
    cols = st.columns(num_cols)
    for j, q in enumerate(faq_questions[i:i+num_cols]):
        if cols[j].button(q, use_container_width=True):
            st.session_state.selected_question = q

# ===== User Input Section =====
col1, col2 = st.columns([6, 1])
with col1:
    user_question = st.text_input(
        "",  # Hide label
        placeholder="Type your question here...",
        value=st.session_state.selected_question  # Auto-filled by FAQ
    )
with col2:
    st.write("")  # Padding
    ask_clicked = st.markdown(
        """
        <style>
        .arrow-button {
            background-color: black;
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            font-size: 20px;
            cursor: pointer;
        }
        </style>
        <form action="#" method="get">
            <button class="arrow-button" type="submit">↑</button>
        </form>
        """,
        unsafe_allow_html=True
    )

# ===== Submit Query to Backend API =====
if ask_clicked and user_question.strip():
    with st.spinner("Retrieving answer..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "question": user_question,
                    "model": selected_model,
                    "provider": selected_provider,
                    "search_forum": search_forum
                },
                timeout=60
            )
            if response.ok:
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])
                retrieval_time = data.get("retrieval_time", None)
                llm_time = data.get("llm_time", None)

                # Save to session and persist to local file
                st.session_state.chat_history.append({
                    "question": user_question,
                    "provider": selected_provider,
                    "model": selected_model,
                    "answer": answer,
                    "sources": sources,
                    "retrieval_time": retrieval_time,
                    "llm_time": llm_time,
                    "input_tokens": data.get("input_tokens", 0),
                    "output_tokens": data.get("output_tokens", 0),
                    "total_tokens": data.get("total_tokens", 0),
                    "time": datetime.now().strftime("%H:%M:%S"),
                })
                save_chat_history(st.session_state.chat_history)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# ===== Render Chat History Below Input =====
if st.session_state.chat_history:
    for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
        # 问题标题
        st.markdown(f"### 👤 {chat['question']}")

        # 元信息行
        st.markdown(
            f"""
            <div style='display:flex; gap:20px; font-size:14px; color:#555; margin:5px 0 10px 0;'>
                <span>📄 <b style='color:#4CAF50;'>Retrieval:</b> {chat.get('retrieval_time', '?')} ms</span>
                <span>🧠 <b style='color:#2196F3;'>LLM:</b> {round(chat.get('llm_time', 0) / 1000, 2)} s</span>
                <span>🤖 <b style='color:#9C27B0;'>Model:</b> {chat.get('provider', 'unknown')} / {chat.get('model', 'unknown')}</span>
                <span>🔢 <b style='color:#FF9800;'>Tokens:</b> {chat.get('total_tokens', 0)} (in: {chat.get('input_tokens', 0)}, out: {chat.get('output_tokens', 0)})</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 答案卡片
        answer_html = md_to_html(chat["answer"], extensions=["extra", "sane_lists"])

        # ===== 渲染答案卡片（整块内容都在同一个容器里，不会“飘出框外”）=====
        final_html = f"""
        <div style="
            background-color:#f9f9ff;
            border:1px solid #ddd;
            border-radius:8px;
            padding:12px 15px;
            margin:10px 0;
            box-shadow:0 1px 3px rgba(0,0,0,0.05);
        ">
          <div style="font-weight:bold; margin-bottom:6px;">🤖 Answer:</div>
          <div style="font-size:16px; line-height:1.6; color:#222;">
            {answer_html}
          </div>
        </div>
        """

        st.markdown(final_html, unsafe_allow_html=True)

        if chat['sources']:
            with st.expander("📚 Sources", expanded=False):
                for i, src in enumerate(chat['sources'], 1):
                    src_type = src.get("type", "document")
                    snippet = src.get("snippet", "")
                    content = src.get("content", "")
                    unique_id = str(uuid.uuid4()).replace("-", "")  # 保证每条唯一

                    if src_type == "document":
                        st.markdown(f"**{i}. 📄 Document:** `{src['source']}` - Page {src.get('page', '-')}")
                        st.caption(snippet)

                    elif src_type == "forum":
                        st.markdown(f"**{i}. 🌍 Forum:** [{src['source']}]({src['source']})")
                        st.caption(snippet)

                        if content:
                            st.markdown(
                                f"""
                                <details style='margin-top:6px;'>
                                  <summary style='cursor: pointer; color:#4A90E2;'>Show Content</summary>
                                  <div style='margin-top:8px; padding:8px; background-color:#f5f5f5;
                                              border:1px solid #ddd; border-radius:6px;'>
                                    {content}
                                  </div>
                                </details>
                                """,
                                unsafe_allow_html=True
                            )
        st.markdown("---")


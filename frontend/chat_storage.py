# frontend/chat_storage.py
import json
import os

# 存储聊天记录的文件路径（放在项目根目录下）
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

def save_chat_history(history):
    """保存聊天记录到 JSON 文件"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] 保存聊天记录失败: {e}")

def load_chat_history():
    """从 JSON 文件加载聊天记录"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ERROR] 加载聊天记录失败: {e}")
    return []

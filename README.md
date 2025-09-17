# 准备
pip install -r requirements.txt

# 索引文档
python -m scripts.ingest

# 启动后端
python -m backend.app

# 启动前端
streamlit run frontend/streamlit_app.py

# 访问
前端：http://localhost:8501

# Demo 讲解要点（面试时可用）
RAG 流程：文档加载 → 切分 → 向量化（Chroma 持久化） → 检索（MMR） → LLM 生成 → 引用来源
安全：safety.py 做关键词阻断（可扩展为分类器/策略）

# 可扩展性：
新数据直接丢 data/doc/，运行 scripts/ingest.py 增量更新
需要实时论坛/FAQ？添加 services/realtime.py 并在 rag.py 路由器里组合 retriever + API
如需重排器或 BM25 混合检索，可在 retrievers.py 里扩展
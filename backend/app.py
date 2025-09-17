from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.logger import logger
from backend.routers.query import router as query_router
from backend.services import rag  # 导入 rag 模块，触发全局初始化

# ================================
# FastAPI 应用初始化
# ================================
app = FastAPI(title="Link's Companion App - RAG API")

# 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（开发环境）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 启动事件：项目启动时加载 Embeddings & 向量库
# ================================
@app.on_event("startup")
async def startup_event():
    """
    项目启动时执行，确保 Embeddings 和向量库提前加载，避免冷启动延迟。
    """
    logger.info("🚀 FastAPI Startup: Initializing Embeddings & VectorDB ...")
    if rag.EMBEDDINGS and rag.RETRIEVER:
        logger.info("✅ Embeddings & VectorDB are ready to use!")
    else:
        logger.error("❌ Failed to initialize Embeddings or VectorDB!")

# ================================
# 健康检查接口
# ================================
@app.get("/health")
def health():
    """
    用于检测后端是否存活
    """
    return {"status": "ok"}

# ================================
# 注册路由
# ================================
app.include_router(query_router, prefix="/api")

# ================================
# 主入口
# ================================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server...")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

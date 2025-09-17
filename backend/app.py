from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.logger import logger
from backend.routers.query import router as query_router
from backend.services import rag  # Import the rag module to trigger global initialization

# ================================
# FastAPI App Initialization
# ================================
app = FastAPI(title="Link's Companion App - RAG API")

# Allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# Startup Event: Load Embeddings & Vector Store on Startup
# ================================
@app.on_event("startup")
async def startup_event():
    """
    Executed when the app starts.
    Ensures that embeddings and the vector store are preloaded to avoid cold start delays.
    """
    logger.info("🚀 FastAPI Startup: Initializing Embeddings & VectorDB ...")
    if rag.EMBEDDINGS and rag.RETRIEVER:
        logger.info("✅ Embeddings & VectorDB are ready to use!")
    else:
        logger.error("❌ Failed to initialize Embeddings or VectorDB!")

# ================================
# Health Check Endpoint
# ================================
@app.get("/health")
def health():
    """
    Used to verify whether the backend is running
    """
    return {"status": "ok"}

# ================================
# Register Routers
# ================================
app.include_router(query_router, prefix="/api")

# ================================
# Entry Point
# ================================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server...")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

from backend.services.ingest import build_or_update_vectorstore

if __name__ == "__main__":
    build_or_update_vectorstore()
    print("✅ Ingestion finished.")

from chromadb import PersistentClient

# Initialize a Chroma client with persistent storage
# This will create or connect to a vector store located at ./data/vectorstore
client = PersistentClient(path="./data/vectorstore")

# Access the specific collection named "link_ecu_docs"
# This collection should have been previously created via Chroma API or LangChain
collection = client.get_collection("link_ecu_docs")

# Print basic information about the collection (e.g. name, size, metadata)
print(collection)

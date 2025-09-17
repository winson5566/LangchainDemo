from chromadb import PersistentClient

client = PersistentClient(path="./data/vectorstore")
collection = client.get_collection("link_ecu_docs")
print(collection)

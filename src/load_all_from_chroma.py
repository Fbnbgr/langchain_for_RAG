import os
import chromadb

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    collection = client.get_collection("langchain")  # oder dein Collection-Name
    data = collection.get(include=["documents", "metadatas"])
    
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    
    print(f"Gesamt: {len(ids)} Dokumente")
    for idx, doc_id in enumerate(ids, start=1):
        print(f"[{idx}] ID: {doc_id}")
        print(f"    Metadata: {metadatas[idx-1]}")
        print(f"    Text: {documents[idx-1][:200]!r}")
        print()
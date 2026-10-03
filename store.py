# store.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")  # persists to disk

collection = client.get_or_create_collection(
    name="invoices",
    metadata={"hnsw:space": "cosine"}  # explicitly set cosine distance for the HNSW index
)

def store_chunks(chunks: list[dict]):
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

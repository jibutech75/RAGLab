# query.py
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("invoices")
print('jibu prints---')
print(collection)
def retrieve(query: str, k: int = 5) -> list[str]:
    # bge models recommend prefixing queries (not needed for the doc side)
    query_prefixed = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = model.encode([query_prefixed], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    return results["documents"][0]  # top-k chunk texts

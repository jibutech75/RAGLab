# embed.py
from sentence_transformers import SentenceTransformer

# bge-small: 130MB, 384-dim, strong for its size, runs fast on CPU (Apple Silicon)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [c["content"] for c in chunks]
    # bge models recommend a prefix for queries (not documents) — see step 7
    embeddings = model.encode(texts, normalize_embeddings=True)  # normalized → cosine == dot product
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks

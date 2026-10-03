# Local RAG Pipeline for Invoice PDFs — Developer Guide
### Target: macOS, 30GB RAM, fully local, open-source stack

---

## 1. Architecture Overview

```
PDF invoice
   │
   ▼
[1] Parse & Extract (layout-aware text/tables)
   │
   ▼
[2] Chunk (structure-aware, not naive fixed windows)
   │
   ▼
[3] Embed chunks (local sentence-transformer model)
   │
   ▼
[4] Store in vector DB (ChromaDB, persisted on disk)
   │
   ▼
User query ──► Embed query ──► ANN search (cosine similarity, HNSW index)
   │                                   │
   │                                   ▼
   │                          Top-k relevant chunks
   │                                   │
   └──────────────► Prompt template (query + retrieved context)
                                       │
                                       ▼
                          Local LLM via Ollama (generation)
                                       │
                                       ▼
                                 Final answer
```

---

## 2. Component Choices & Why

| Stage | Tool | Size / Footprint | Why this one |
|---|---|---|---|
| PDF parsing | `PyMuPDF` (fitz) + `pdfplumber` for tables | ~50MB | Fast, preserves layout, good table extraction for invoices |
| OCR (scanned PDFs) | `pytesseract` + Tesseract binary | ~100MB | Only needed if invoice is an image-based scan |
| Chunking | Custom rule-based + `langchain-text-splitters` | trivial | Invoices need structure-aware chunking, not blind 500-char windows |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) | ~130MB | Small, CPU-fast, strong retrieval quality for its size |
| Vector DB | `chromadb` (embedded/local mode) | trivial (SQLite+Parquet) | No server to run, persists to disk, built-in HNSW ANN index |
| LLM (generation) | `Ollama` running `qwen2.5:7b-instruct-q4_K_M` or `llama3.1:8b-instruct-q4_K_M` | ~4.5–5GB on disk | Quantized, runs comfortably in 30GB RAM, Metal-accelerated on Mac |
| Orchestration | Plain Python (recommended for learning) or LangChain | — | Plain Python teaches you what's actually happening |

**Total resident memory during a query:** embedding model (~1GB loaded) + LLM (~6-8GB loaded) + Chroma (~negligible) → comfortably under 30GB, leaving room for your OS and other apps.

---

## 3. Install Everything

```bash
# Create an isolated environment
python3 -m venv rag_env
source rag_env/bin/activate

# Core libraries
pip install pymupdf pdfplumber pytesseract pillow
pip install sentence-transformers
pip install chromadb
pip install ollama                 # Python client for Ollama
pip install numpy

# Tesseract binary (only if you'll handle scanned PDFs)
brew install tesseract

# Ollama itself (the local LLM runtime)
brew install ollama
ollama serve &                     # starts the local server (localhost:11434)
ollama pull qwen2.5:7b-instruct-q4_K_M
```

---

## 4. Step 1 — Parse the PDF

Invoices are structured documents (line items, totals, tax tables) so treat them differently from a "read the wall of text" PDF. Extract text AND tables separately, and keep page/line structure — this dramatically improves chunk quality later.

```python
# parse_pdf.py
import fitz  # PyMuPDF
import pdfplumber

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """Extract text per page with basic layout info."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages

def extract_tables(pdf_path: str) -> list[dict]:
    """Extract tables (line items, totals) using pdfplumber."""
    tables_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                tables_out.append({
                    "page": page_num + 1,
                    "table_index": t_idx,
                    "rows": table
                })
    return tables_out

def is_scanned_pdf(pdf_path: str) -> bool:
    """Heuristic: if extracted text is near-empty, it's likely a scanned image."""
    pages = extract_text_pymupdf(pdf_path)
    total_chars = sum(len(p["text"].strip()) for p in pages)
    return total_chars < 20  # tune threshold as needed

def ocr_pdf(pdf_path: str) -> list[dict]:
    """Fallback OCR path for scanned invoices."""
    import pytesseract
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages

if __name__ == "__main__":
    path = "invoice.pdf"
    if is_scanned_pdf(path):
        print("Detected scanned PDF — running OCR")
        pages = ocr_pdf(path)
    else:
        pages = extract_text_pymupdf(path)
    tables = extract_tables(path)
    print(pages)
    print(tables)
```

---

## 5. Step 2 — Chunk (structure-aware)

For invoices, don't blindly split every 500 characters — you'll cut a line item in half. Instead:
- Keep each **table row** as its own chunk (or the whole table as one chunk, if small).
- Keep **header info** (invoice #, date, vendor, bill-to) as one chunk.
- Keep **totals/tax section** as one chunk.
- For free-text narrative sections (terms, notes), use recursive character splitting with overlap.

```python
# chunk.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_invoice(pages: list[dict], tables: list[dict]) -> list[dict]:
    chunks = []

    # 1. Free text — split with overlap so context isn't lost at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": f"text_p{page['page']}_c{i}",
                "content": piece,
                "metadata": {"type": "text", "page": page["page"]}
            })

    # 2. Tables — keep structurally intact, rendered as readable text
    for t in tables:
        rows_text = "\n".join([" | ".join(str(c) for c in row if c) for row in t["rows"]])
        chunks.append({
            "id": f"table_p{t['page']}_t{t['table_index']}",
            "content": rows_text,
            "metadata": {"type": "table", "page": t["page"]}
        })

    return chunks
```

This is the part worth iterating on most — retrieval quality lives or dies on chunk quality, far more than on which embedding model you pick.

---

## 6. Step 3 — Embed

```python
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
```

**Note on `normalize_embeddings=True`:** once vectors are L2-normalized, cosine similarity and dot product give identical rankings — this is why most vector DBs (Chroma included) let you pick either metric interchangeably once you normalize at embedding time.

---

## 7. Step 4 — Store in ChromaDB

```python
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
```

---

## 8. Step 5 — Query: Retrieval

This is where your cosine-similarity / nearest-neighbour question lands.

**What's actually happening:** Chroma builds an HNSW graph index over your vectors. HNSW is an *approximate* nearest-neighbour algorithm — but at your scale (hundreds to low-thousands of chunks per invoice batch), it behaves essentially like exact search. The distance metric it uses to rank candidates is cosine similarity (set via `hnsw:space: cosine` above).

```python
# query.py
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("invoices")

def retrieve(query: str, k: int = 5) -> list[str]:
    # bge models recommend prefixing queries (not needed for the doc side)
    query_prefixed = f"Represent this sentence for searching relevant passages: {query}"
    query_embedding = model.encode([query_prefixed], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    return results["documents"][0]  # top-k chunk texts
```

**Should you use cosine, dot product, or Euclidean?**
- Cosine: best default for text embeddings — robust to differences in vector magnitude/length of chunk.
- Dot product: same as cosine once vectors are normalized (you're already doing this).
- Euclidean (L2): rarely better for text; typically used for image or other non-normalized embeddings.

→ **Stick with cosine.** It's the right call here, not a compromise.

---

## 9. Step 6 — Augment + Generate (RAG completion)

```python
# generate.py
import ollama

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n---\n".join(context_chunks)
    return f"""You are an assistant answering questions about an invoice document.
Use ONLY the context below to answer. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

def generate_answer(query: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(query, context_chunks)
    response = ollama.chat(
        model="qwen2.5:7b-instruct-q4_K_M",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
```

---

## 10. Tying it all together

```python
# main.py
from parse_pdf import extract_text_pymupdf, extract_tables, is_scanned_pdf, ocr_pdf
from chunk import chunk_invoice
from embed import embed_chunks
from store import store_chunks
from query import retrieve
from generate import generate_answer

def ingest(pdf_path: str):
    pages = ocr_pdf(pdf_path) if is_scanned_pdf(pdf_path) else extract_text_pymupdf(pdf_path)
    tables = extract_tables(pdf_path)
    chunks = chunk_invoice(pages, tables)
    chunks = embed_chunks(chunks)
    store_chunks(chunks)
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")

def ask(question: str):
    context = retrieve(question, k=5)
    answer = generate_answer(question, context)
    print(answer)

if __name__ == "__main__":
    ingest("invoice.pdf")
    ask("What is the total amount due and the due date?")
    ask("List all line items with their quantities and prices.")
```

---

## 11. Practical tips / gotchas for invoices specifically

1. **Test with a real invoice PDF first, not a text-heavy sample.** Invoices have irregular layouts (multi-column headers, tables) — `pdfplumber.extract_tables()` sometimes misses tables with no visible borders; if that happens, try `camelot-py` as a fallback.
2. **Add metadata filters.** Store `invoice_number`, `vendor`, `date` as Chroma metadata so you can filter (`where={"vendor": "Acme"}`) before doing similarity search — useful once you have multiple invoices in one collection.
3. **k=3-5 is usually enough** for a single invoice; going higher just adds noise to the LLM's context.
4. **Quantized model choice:** `q4_K_M` is the sweet spot of quality vs. size for 8B models on 30GB RAM. If you want faster responses and don't mind slightly lower quality, try `q4_0`; for better quality with more RAM headroom, `q5_K_M` or `q6_K`.
5. **Evaluate retrieval separately from generation.** If answers seem wrong, first check what chunks were actually retrieved (print them) before assuming the LLM is at fault — bad retrieval is the #1 cause of bad RAG answers.
6. **Re-ranking (optional, next step once basics work):** add a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB) after initial retrieval to re-score the top ~20 candidates down to your final top-5. This measurably improves precision for structured documents like invoices, at very low extra compute cost.

---

## 12. Suggested learning progression

1. Get ingestion + retrieval working with one clean, text-based invoice PDF.
2. Print retrieved chunks for a few test queries and sanity-check relevance manually.
3. Add the LLM generation step.
4. Add a scanned/OCR invoice to test that path.
5. Add multiple invoices to the same collection + metadata filtering.
6. (Optional) Add a cross-encoder reranker.
7. (Optional) Swap ChromaDB for `FAISS` or `LanceDB` to compare — good exercise in seeing that the vector DB layer is fairly interchangeable once you understand the embed → index → search flow.

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

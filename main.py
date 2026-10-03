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
    #ingest("invoice.pdf")
    ask("What is the total amount due and the due date?")
   # ask("List all line items with their quantities and prices.")
    ask("who is the prime miniser of india?")

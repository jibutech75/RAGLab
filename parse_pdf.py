# parse_pdf.py
import pymupdf as fitz  # PyMuPDF
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

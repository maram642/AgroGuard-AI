"""
text_extractor.py
Extracts text per page using PyMuPDF. Reports character counts so the
pipeline can decide, per page, whether OCR is needed.
"""

import fitz


def extract_text_per_page(pdf_path):
    """
    Returns a list of dicts, one per page:
    {"page": int, "chars": int, "text": str}
    """
    results = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text")
        results.append({
            "page": i,
            "chars": len(text),
            "text": text
        })
    doc.close()
    return results

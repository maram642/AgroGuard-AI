"""
ocr_extractor.py
OCR fallback for pages where PyMuPDF found little/no text (scanned images).
Language is configurable per file: "eng" for your disease PDFs,
"fra+ara" for the pesticide catalog (only needed if some pages
in that specific file ever turn out to be scanned - unlikely,
but supported).

REQUIRES: Tesseract OCR installed and on PATH, plus Poppler for pdf2image.
If Tesseract isn't on PATH, set pytesseract.pytesseract.tesseract_cmd below.
"""

import pytesseract
from pdf2image import convert_from_path

# Tesseract: reads text out of an image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Poppler: converts a PDF page into an image in the first place
POPPLER_PATH = r"C:\Users\asus\Desktop\Plant Internship\Release-26.02.0-0\poppler-26.02.0\Library\bin"


def ocr_page(pdf_path, page_num, lang="eng"):
    """
    Rasterizes a single PDF page and runs OCR on it.
    page_num is 0-indexed to match PyMuPDF's convention;
    pdf2image's first_page/last_page are 1-indexed, so we convert.
    """
    images = convert_from_path(
        pdf_path,
        first_page=page_num + 1,
        last_page=page_num + 1,
        poppler_path=POPPLER_PATH
    )
    if not images:
        return ""
    text = pytesseract.image_to_string(images[0], lang=lang)
    return text


def ocr_document(pdf_path, page_numbers, lang="eng"):
    """
    OCRs a specific list of page numbers (only the ones that need it,
    as decided by the pipeline based on text_extractor's char counts).
    Returns {page_num: ocr_text}.
    """
    results = {}
    for page_num in page_numbers:
        try:
            results[page_num] = ocr_page(pdf_path, page_num, lang=lang)
        except Exception as e:
            print(f"  OCR failed on page {page_num} of {pdf_path}: {e}")
            results[page_num] = ""
    return results

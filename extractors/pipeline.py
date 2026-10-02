"""
pipeline.py
Final orchestration: for every PDF in metadata.xlsx, runs text extraction,
OCR fallback (per page, only where needed), and table extraction —
then merges everything into one JSON per file in extracted/.

Language for OCR is chosen automatically:
  - "pesticide" document_type -> fra+ara
  - everything else           -> eng
(Based on your corpus: only the pesticide catalog is French/Arabic,
everything else is English.)
"""

import os
import json
import pandas as pd

from text_extractor import extract_text_per_page
from ocr_extractor import ocr_document
from table_extractor import extract_tables

METADATA_FILE = "metadata.xlsx"
OUTPUT_DIR = "extracted"
MIN_CHARS_BEFORE_OCR = 100  # below this, a page is considered "needs OCR"

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_excel(METADATA_FILE)


def get_ocr_lang(doc_type):
    return "fra+ara" if doc_type == "pesticide" else "eng"


def process_pdf(file_path, doc_type):
    lang = get_ocr_lang(doc_type)

    # Step 1: text extraction, per page
    text_pages = extract_text_per_page(file_path)

    # Step 2: figure out which pages need OCR
    pages_needing_ocr = [p["page"] for p in text_pages if p["chars"] < MIN_CHARS_BEFORE_OCR]

    ocr_results = {}
    if pages_needing_ocr:
        print(f"  {len(pages_needing_ocr)} page(s) need OCR (lang={lang})")
        ocr_results = ocr_document(file_path, pages_needing_ocr, lang=lang)

    # Step 3: table extraction (independent of text/OCR)
    table_results = extract_tables(file_path)

    # Step 4: merge everything per page
    merged_pages = []
    for p in text_pages:
        page_num = p["page"]
        final_text = p["text"]
        used_ocr = False

        if page_num in ocr_results:
            final_text = ocr_results[page_num]
            used_ocr = True

        page_entry = {
            "page": page_num,
            "text": final_text,
            "used_ocr": used_ocr,
            "tables": None
        }

        if page_num in table_results:
            page_entry["tables"] = table_results[page_num]

        merged_pages.append(page_entry)

    return merged_pages


for _, row in df.iterrows():
    file_path = row["file_path"]
    doc_type = row["document_type"]

    if not os.path.exists(file_path):
        print(f"MISSING FILE: {file_path}")
        continue

    print(f"Processing: {file_path}")

    try:
        pages = process_pdf(file_path, doc_type)

        output = {
            "metadata": row.to_dict(),
            "pages": pages
        }

        out_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  OK -> {out_path}")

    except Exception as e:
        print(f"  FAILED: {file_path} -> {e}")

print("\nPipeline complete. Check the 'extracted/' folder.")
print("Spot-check a few JSON files - especially any page with used_ocr=true")
print("or tables != null - to confirm quality before moving to chunking.")

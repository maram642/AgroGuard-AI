"""
audit_ocr_failures.py
Scans every extracted/*.json file and reports pages where OCR was
attempted (used_ocr=true) but produced empty or near-empty text -
a sign of silent OCR failure (e.g. Poppler misconfigured).

Run this to find exactly which files need to be re-processed,
instead of guessing or re-running the whole 36-file batch.
"""

import os
import json

EXTRACTED_DIR = "extracted"
MIN_OCR_CHARS = 20  # OCR output shorter than this is suspicious

problem_files = []

for fname in sorted(os.listdir(EXTRACTED_DIR)):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(EXTRACTED_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for page in data.get("pages", []):
        if page.get("used_ocr") and len(page.get("text", "")) < MIN_OCR_CHARS:
            problem_files.append({
                "file": fname,
                "page": page["page"],
                "chars": len(page.get("text", ""))
            })

if not problem_files:
    print("No silent OCR failures found. All OCR'd pages have text.")
else:
    print(f"Found {len(problem_files)} page(s) with likely silent OCR failure:\n")
    seen_files = set()
    for p in problem_files:
        print(f"  {p['file']} - page {p['page']} ({p['chars']} chars)")
        seen_files.add(p["file"])
    print(f"\n{len(seen_files)} distinct file(s) affected:")
    for f in sorted(seen_files):
        print(f"  - {f}")
    print("\nFix Poppler path in ocr_extractor.py, then re-run extraction")
    print("for just these specific files (see re-run snippet below).")

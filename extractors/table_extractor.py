"""
table_extractor.py
Extracts tables using pdfplumber (always available) and Camelot (if
Ghostscript is installed). Filters out fragment "tables" that are just
wrapped/merged cell pieces (e.g. a single word split off from its row).

Decision rule:
  - If Camelot found a table on this page with accuracy >= CAMELOT_MIN_ACCURACY,
    use Camelot's version (it's usually cleaner on bordered docs).
  - Otherwise, fall back to pdfplumber's version, filtered for junk fragments.
"""

import pdfplumber
import pandas as pd

try:
    import camelot
    HAS_CAMELOT = True
except Exception as e:
    HAS_CAMELOT = False
    print(f"NOTE: camelot unavailable ({e}). Using pdfplumber only for tables.")

CAMELOT_MIN_ACCURACY = 80
MIN_ROWS = 3   # discard tables with fewer rows than this - likely a fragment
MIN_COLS = 2   # discard tables with fewer columns than this - likely a fragment


def is_real_table(table):
    """Filters out fragment 'tables' (wrapped words split off from a real row)."""
    if not table or len(table) < MIN_ROWS:
        return False
    if len(table[0]) < MIN_COLS:
        return False
    return True


def forward_fill_table(table):
    """Merged cells show blank on rows below the first; carry the value down."""
    if not table or len(table) < 2:
        return table
    tdf = pd.DataFrame(table[1:], columns=table[0])
    tdf = tdf.replace("", None).ffill()
    return [list(tdf.columns)] + tdf.values.tolist()


def extract_tables_pdfplumber(pdf_path):
    """Returns {page_num: [table, table, ...]} - only tables that pass the filter."""
    results = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_tables = page.extract_tables()
            good_tables = [forward_fill_table(t) for t in raw_tables if is_real_table(t)]
            if good_tables:
                results[page_num] = good_tables
    return results


def extract_tables_camelot(pdf_path):
    """Returns {page_num: [(rows, accuracy), ...]} for tables Camelot could parse."""
    results = {}
    if not HAS_CAMELOT:
        return results
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        for t in tables:
            page_num = int(t.page) - 1
            accuracy = t.parsing_report.get("accuracy", 0)
            rows = t.df.values.tolist()
            results.setdefault(page_num, []).append((rows, accuracy))
    except Exception as e:
        print(f"  Camelot failed entirely on {pdf_path}: {e}")
    return results


def extract_tables(pdf_path):
    """
    Main entry point. Returns {page_num: {"source": "camelot"/"pdfplumber",
                                           "tables": [table, ...]}}
    """
    plumber_results = extract_tables_pdfplumber(pdf_path)
    camelot_results = extract_tables_camelot(pdf_path)

    final = {}
    all_pages = set(plumber_results.keys()) | set(camelot_results.keys())

    for page_num in all_pages:
        camelot_candidates = camelot_results.get(page_num, [])
        good_camelot = [rows for rows, acc in camelot_candidates if acc >= CAMELOT_MIN_ACCURACY]

        if good_camelot:
            final[page_num] = {"source": "camelot", "tables": good_camelot}
        elif page_num in plumber_results:
            final[page_num] = {"source": "pdfplumber", "tables": plumber_results[page_num]}

    return final

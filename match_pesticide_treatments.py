"""
match_pesticide_treatments.py

Takes your already-extracted pesticide catalog table (from the pipeline's
table_extractor output) and, for every crop+disease in your metadata.xlsx,
searches the catalog for a matching row and pulls out the real product
name, substance, and dosage - ONLY if it's actually found in the catalog.

If no match is found, it's marked clearly as NOT FOUND - never guessed.

Two-stage approach:
  1. Fast keyword matching (French crop + disease terms) to find candidate
     rows - cheap, no LLM needed.
  2. For matched candidates, a local LLM (Ollama) extracts just the
     dosage/product from that specific row's USAGE text - low hallucination
     risk since it's extracting from a short, already-matched snippet,
     not summarizing/interpreting a whole document.
"""

import os
import json
import re
import pandas as pd
import ollama

METADATA_FILE = "metadata.xlsx"
EXTRACTED_DIR = "extracted"
PESTICIDE_FILE_KEYWORD = "liste-des-produits-pesticides"  # matches the catalog's filename
OUTPUT_FILE = "pesticide_treatment_matches.xlsx"
OLLAMA_MODEL = "mistral"

# ============================================================
# Crop name translation: your metadata's crop value -> French
# term(s) as they appear in the catalog. Add more if your crop
# list has entries not covered here.
# ============================================================
CROP_FR = {
    "olive": ["olivier"],
    "citrus": ["agrumes"],
    "tomate": ["tomate"],
    "tomato": ["tomate"],
    "potato": ["pomme de terre", "p.de terre", "p. de terre"],
    "pepper": ["piment", "poivron"],
    "figuier": ["figuier"],
    "fig": ["figuier"],
    "grapevine": ["vigne"],
    "wheat": ["blé", "ble", "céréales", "cereales"],
}

# ============================================================
# Disease name translation: your metadata's disease value ->
# French keyword(s) that would appear in the catalog's USAGE
# column for that target. Add more as needed - if a disease
# isn't listed here, it will be skipped with a note to add it.
# ============================================================
DISEASE_FR = {
    "leaf spot": ["taches foliaires", "septoriose"],
    "peacock spot": ["œil de paon", "oeil de paon", "tavelure"],
    "gummosis": ["gommose"],
    "downy mildew": ["mildiou"],
    "powdery mildew": ["oïdium", "oidium"],
    "leaf rust": ["rouille"],
    "stem rust": ["rouille"],
    "stripe rust": ["rouille"],
    "rust": ["rouille"],
    "late blight": ["mildiou"],
    "early blight": ["alternariose", "alternaria"],
    "bacterial leaf spot": ["bactériose", "bacteriose"],
    "fruit and leaf spot": ["bactériose"],
    "septoria leaf blotch": ["septoriose"],
    "fusarium head blight": ["fusariose"],
    "black spot": ["black spot", "taches noires"],  # citrus (Phyllosticta citricarpa)
    "black rot": ["black-rot", "black rot"],  # grapevine (Guignardia bidwellii) - was incorrectly
                                                # sharing a key with "black spot" before; these are
                                                # different diseases on different crops
    "anthracnose": ["anthracnose"],
    "cercospora leaf spot": ["cercosporiose"],
    "phytophthora blight": ["mildiou"],
    "phomopsis leaf spot": ["excoriose"],  # Tunisian catalog's official name for this target
    "septoria leaf spot": ["septoriose"],
}


def normalize(text):
    """Lowercases AND treats underscores/hyphens as spaces, so 'early_blight',
    'Late-Blight', and 'late blight' all normalize to the same thing before
    dictionary lookup - fixes the biggest cause of missed matches."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+$", "", text).strip()  # strip trailing digits (e.g. "stem rust1" -> "stem rust")
    return text


def lookup_terms(name, mapping_dict):
    """Exact match first, then falls back to substring match in either
    direction (e.g. 'head blight' matches the 'fusarium head blight' key)."""
    norm = normalize(name)
    if norm in mapping_dict:
        return mapping_dict[norm]
    for key, terms in mapping_dict.items():
        if key in norm or norm in key:
            return terms
    return []


def load_catalog_rows():
    """Loads the pesticide catalog's extracted table rows (all pages)."""
    catalog_json_path = None
    for fname in os.listdir(EXTRACTED_DIR):
        if PESTICIDE_FILE_KEYWORD in fname.lower():
            catalog_json_path = os.path.join(EXTRACTED_DIR, fname)
            break

    if not catalog_json_path:
        raise FileNotFoundError(
            f"Could not find the pesticide catalog's extracted JSON in {EXTRACTED_DIR}/ "
            f"(looking for a filename containing '{PESTICIDE_FILE_KEYWORD}')"
        )

    with open(catalog_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for page in data.get("pages", []):
        tables_block = page.get("tables")
        if not tables_block:
            continue
        for table in tables_block.get("tables", []):
            if len(table) < 2:
                continue
            header = table[0]
            if len(header) < 6:
                continue  # skip fragment/debris tables (real catalog rows have 8+ columns)
            for row in table[1:]:
                row_text = " | ".join(str(c) for c in row if c)
                if len(row_text) > 20:
                    rows.append({"page": page["page"], "text": row_text})
    return rows


def find_candidate_rows(catalog_rows, crop, disease):
    crop_terms = lookup_terms(crop, CROP_FR)
    disease_terms = lookup_terms(disease, DISEASE_FR)

    if not crop_terms or not disease_terms:
        return None  # unknown mapping - skip matching, flag separately

    candidates = []
    for row in catalog_rows:
        text_lower = row["text"].lower()
        crop_positions = [text_lower.find(t) for t in crop_terms if t in text_lower]
        disease_positions = [text_lower.find(t) for t in disease_terms if t in text_lower]

        if not crop_positions or not disease_positions:
            continue

        # Proximity score: how close is the crop mention to the disease
        # mention? A row where "Agrumes" and "Black spot" sit right next
        # to each other is far more likely to be the real match than a
        # giant row where they just happen to both appear somewhere.
        best_distance = min(abs(c - d) for c in crop_positions for d in disease_positions)
        candidates.append({**row, "proximity": best_distance})

    if not candidates:
        return []

    candidates.sort(key=lambda r: r["proximity"])
    return candidates


EXTRACT_PROMPT = """From this pesticide catalog row (French, Tunisia official registry), \
extract ONLY what is explicitly written. Do not infer or guess.

ROW TEXT:
{row_text}

Return ONLY valid JSON:
{{
  "product_name": "commercial product name if present, else empty string",
  "substance_active": "active substance if present, else empty string",
  "dosage": "exact dosage/rate as written, else empty string",
  "homologation_number": "N°.H. registration number if present, else empty string"
}}
"""


PLACEHOLDER_PHRASES = [
    "not explicitly stated", "not stated", "none explicitly stated",
    "not mentioned", "empty string", "n/a", "none",
    "not applicable", "not found", "no information", "not provided",
]


def is_placeholder(value):
    if not value or not isinstance(value, str):
        return not value
    return any(p in value.strip().lower() for p in PLACEHOLDER_PHRASES)


def clean_extracted(extracted):
    """Blanks out any field where the model leaked a placeholder phrase
    instead of an actual empty string - same issue distill.py had."""
    if not extracted:
        return extracted
    return {k: ("" if is_placeholder(v) else v) for k, v in extracted.items()}


def extract_from_row(row_text):
    prompt = EXTRACT_PROMPT.format(row_text=row_text[:1500])
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


# ============================================================
# Main
# ============================================================

df = pd.read_excel(METADATA_FILE)
disease_rows = df[df["document_type"] == "disease"]
unique_pairs = disease_rows[["crop", "disease"]].drop_duplicates()

print("Loading pesticide catalog table rows...")
catalog_rows = load_catalog_rows()
print(f"Loaded {len(catalog_rows)} catalog rows.\n")

results = []

for _, pair in unique_pairs.iterrows():
    crop = str(pair["crop"]).strip()
    disease = str(pair["disease"]).strip()

    print(f"Matching: {crop} - {disease}")

    candidates = find_candidate_rows(catalog_rows, crop, disease)

    if candidates is None:
        results.append({
            "crop": crop, "disease": disease, "status": "NO_MAPPING",
            "note": "Crop or disease not in CROP_FR/DISEASE_FR mapping - add it and re-run",
            "product_name": "", "substance_active": "", "dosage": "", "homologation_number": ""
        })
        print("  SKIPPED - no crop/disease keyword mapping defined")
        continue

    if not candidates:
        results.append({
            "crop": crop, "disease": disease, "status": "NOT_FOUND_IN_CATALOG",
            "note": "No matching row found - verify manually in the full catalog before assuming no treatment exists",
            "product_name": "", "substance_active": "", "dosage": "", "homologation_number": ""
        })
        print("  NOT FOUND in catalog")
        continue

    # Take the best candidate (first match); extract structured fields
    best = candidates[0]
    extracted = clean_extracted(extract_from_row(best["text"]))

    if extracted:
        product_name = extracted.get("product_name", "")
        status = "FOUND" if product_name else "FOUND_LOW_CONFIDENCE"
        note = f"Matched on catalog page {best['page']} ({len(candidates)} candidate(s) found)"
        if not product_name:
            note += " - WARNING: no product name extracted, this row may be a table fragment, verify manually"

        results.append({
            "crop": crop, "disease": disease, "status": status,
            "note": note,
            "product_name": product_name,
            "substance_active": extracted.get("substance_active", ""),
            "dosage": extracted.get("dosage", ""),
            "homologation_number": extracted.get("homologation_number", "")
        })
        print(f"  {status} -> {product_name or '?'} @ {extracted.get('dosage', '?')}")
    else:
        results.append({
            "crop": crop, "disease": disease, "status": "FOUND_BUT_EXTRACTION_FAILED",
            "note": f"Matched row on page {best['page']} but LLM could not parse it - check manually: {best['text'][:200]}",
            "product_name": "", "substance_active": "", "dosage": "", "homologation_number": ""
        })
        print("  MATCHED but extraction failed - needs manual check")

results_df = pd.DataFrame(results)
results_df.to_excel(OUTPUT_FILE, index=False)

print(f"\nDone. Results saved to {OUTPUT_FILE}")
print(results_df["status"].value_counts())

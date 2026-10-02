"""
distill.py
Reads extracted/*.json (raw text from disease PDFs), splits long documents
into manageable chunks, runs each chunk through a LOCAL LLM via Ollama,
then MERGES the per-chunk results into one final structured record per
disease document.

Why chunk before the LLM call (rather than just raising a char limit):
long documents fed in one shot risk the model under-attending to content
buried deep in the context (e.g. management advice on page 10 of 12).
Chunking + merging avoids this instead of just hoping a bigger limit helps.

SCOPE NOTE: Dosage and chemical treatment recommendations are
intentionally NOT extracted here - those come only from the official
Tunisian pesticide catalog, handled separately.

IMPORTANT: This only processes document_type == "disease" files.

Requires: Ollama installed and running (https://ollama.com/download)
Requires: model pulled first, e.g.  ollama pull mistral
Requires: pip install ollama
"""

import os
import json
import re
import pandas as pd
import ollama

OLLAMA_MODEL = "mistral"  # change this to test a different model

METADATA_FILE = "metadata.xlsx"
EXTRACTED_DIR = "extracted"

# Output folder and review file are named after the model, so results from
# different models (mistral, llama3.1:8b, qwen2.5:14b-instruct, ...) land in
# separate places and can be compared side by side instead of overwriting
# each other.
MODEL_TAG = OLLAMA_MODEL.replace(":", "_").replace("/", "_")
OUTPUT_DIR = f"distilled_{MODEL_TAG}"
REVIEW_XLSX = f"distilled_for_review_{MODEL_TAG}.xlsx"

CHUNK_SIZE_CHARS = 4000  # per-chunk size sent to the LLM

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(METADATA_FILE)
disease_rows = df[df["document_type"] == "disease"]


def load_extracted_text(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    json_path = os.path.join(EXTRACTED_DIR, base_name + ".json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return "\n".join(p["text"] for p in data.get("pages", []) if p.get("text"))


PLACEHOLDER_PHRASES = [
    "not explicitly stated", "not stated", "none explicitly stated",
    "not mentioned", "not affected", "empty string", "n/a", "none",
    "not applicable", "not found", "no information", "not provided",
]


def is_placeholder(value):
    """Detects Mistral's various ways of saying 'nothing here' so they
    don't get saved as if they were real content."""
    if not value or not isinstance(value, str):
        return not value
    lowered = value.strip().lower()
    return any(phrase in lowered for phrase in PLACEHOLDER_PHRASES)


def clean_json_value(v):
    """Converts NaN/None to a safe empty value so json.dump never writes
    an invalid NaN token."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return v


def split_into_chunks(text, chunk_size=CHUNK_SIZE_CHARS):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


DISTILL_PROMPT = """You are helping build an agricultural knowledge base. Below is a \
PORTION of a document about a plant disease (this may be one part of a longer document).

Crop: {crop}
Disease (as labeled in our corpus): {disease}

Extract ONLY information explicitly stated in this text portion. Do NOT add, infer, \
or guess anything not present. If a field has no information in this portion, use an \
empty string or empty list.

Do NOT extract chemical product names, brand names, or dosages/application rates \
(examples of what to EXCLUDE: "Score", "Tilt", "Actinovate", "Bravo", any named \
commercial product) - those come from a separate official Tunisian source in this \
project, not from this document. Only extract general/cultural management practices \
(crop rotation, sanitation, resistant varieties, pruning, irrigation timing, \
monitoring) - not chemical treatments, not brand names, not specific products.

If a field has no information in this text portion, you MUST use exactly an empty \
string "" or empty list [] - do NOT write phrases like "not stated", "not mentioned", \
"none", or similar. Those phrases are NOT valid values for any field.

Return ONLY valid JSON, no other text, in exactly this shape:
{{
  "pathogen": "scientific name if stated in this portion, else empty string",
  "symptoms": ["distinct symptoms/effects found in this portion, up to 6, no padding or repeats"],
  "affected_parts": ["plant parts mentioned as affected, e.g. leaves, fruit, stem, roots"],
  "spread": "how the disease spreads, if stated in this portion, else empty string",
  "disease_cycle": "disease cycle/survival info if stated in this portion, else empty string",
  "management": ["up to 4 GENERAL (non-chemical) management practices found in this portion"],
  "preventive_advice": "preventive advice if stated in this portion, else empty string"
}}

TEXT PORTION:
{text}
"""


def extract_json_from_text(raw):
    """Local models often wrap JSON in extra prose or code fences despite
    instructions. Try direct parse first, then fall back to finding the
    first {...} block in the response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def call_llm(crop, disease, chunk_text, retry=True):
    prompt = DISTILL_PROMPT.format(crop=crop, disease=disease, text=chunk_text)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}  # low temperature for more consistent JSON
    )
    raw = response["message"]["content"]

    result = extract_json_from_text(raw)

    if result is None and retry:
        # One retry with a stricter reminder - local models sometimes need
        # a second nudge to actually follow the "JSON only" instruction.
        strict_prompt = prompt + "\n\nREMINDER: Output ONLY the JSON object. No explanation, no markdown, no extra text before or after."
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": strict_prompt}],
            options={"temperature": 0.1}
        )
        raw = response["message"]["content"]
        result = extract_json_from_text(raw)

    return result


MAX_SYMPTOMS = 6
MAX_AFFECTED_PARTS = 6
MAX_MANAGEMENT = 6  # raised from 4 now that blob items get split into atomic facts, not discarded
MAX_ITEM_LENGTH = 150  # items longer than this are likely blob-paragraphs, not atomic facts


def normalize_for_dedup(item):
    return re.sub(r"\s+", " ", item.strip().lower())


def split_blob_item(item, max_len=MAX_ITEM_LENGTH):
    """If an item is too long to be one atomic fact, split it on natural
    delimiters (semicolons, then commas) into separate items instead of
    discarding the whole thing - avoids throwing away real content just
    because the model crammed several facts into one list entry."""
    if len(item) <= max_len:
        return [item]

    parts = re.split(r";\s*", item)
    if len(parts) == 1:
        parts = re.split(r",\s*(?=[a-z])", item)  # split on commas before lowercase words (new clause), not mid-phrase

    cleaned = [p.strip().rstrip(".") for p in parts if p.strip() and len(p.strip()) > 3]
    return cleaned if cleaned else [item[:max_len]]  # fallback: truncate rather than lose it entirely


def add_unique_item(target_list, item, seen_set, cap, max_len=MAX_ITEM_LENGTH):
    """Adds item(s) only if real content (not a placeholder) and not a
    duplicate (case/whitespace-insensitive). Oversized items are split
    into atomic facts rather than dropped. Stops once `cap` is reached,
    checked per-piece since one blob item can expand into several."""
    if is_placeholder(item):
        return

    for piece in split_blob_item(item, max_len):
        if len(target_list) >= cap:
            return
        if is_placeholder(piece):
            continue
        key = normalize_for_dedup(piece)
        if key in seen_set:
            continue
        seen_set.add(key)
        target_list.append(piece)


def merge_chunk_results(chunk_results):
    """Combines per-chunk extractions into one record, filtering out
    placeholder leakage, enforcing list caps, and deduplicating."""
    merged = {
        "pathogen": "",
        "symptoms": [],
        "affected_parts": [],
        "spread": "",
        "disease_cycle": "",
        "management": [],
        "preventive_advice": "",
    }

    seen = {
        "symptoms": set(),
        "affected_parts": set(),
        "management": set(),
    }

    for r in chunk_results:
        if not r:
            continue

        for field in ("pathogen", "spread", "disease_cycle", "preventive_advice"):
            value = r.get(field)
            if value and not is_placeholder(value) and not merged[field]:
                merged[field] = value

        for item in r.get("symptoms", []):
            if item:
                add_unique_item(merged["symptoms"], item, seen["symptoms"], MAX_SYMPTOMS)
        for item in r.get("affected_parts", []):
            if item:
                add_unique_item(merged["affected_parts"], item, seen["affected_parts"], MAX_AFFECTED_PARTS)
        for item in r.get("management", []):
            if item:
                add_unique_item(merged["management"], item, seen["management"], MAX_MANAGEMENT)

    return merged


CHEMICAL_LEAK_KEYWORDS = [
    "fungicide", "pesticide", "insecticide", "spray", "spraying",
    "chemical control", "apply copper", "apply sulfur", "apply sulphur",
    "systemic product", "contact fungicide", "dose", "dosage", "ppm",
    "ml/l", "g/l", "kg/ha", "l/ha", "active ingredient"
]


def contains_chemical_leak(text_list_or_str):
    """Flags management/preventive_advice content that strays into chemical
    treatment territory - out of scope for disease-paper distillation per
    the project's Tunisia-only dosage sourcing rule. Formatting filters
    (placeholder/blob detection) can't catch this since it's legitimate-
    looking prose, not junk text - needs an explicit keyword check."""
    if isinstance(text_list_or_str, list):
        combined = " ".join(text_list_or_str).lower()
    else:
        combined = str(text_list_or_str).lower()
    return any(kw in combined for kw in CHEMICAL_LEAK_KEYWORDS)


def compute_missing_info(merged):
    missing = []
    if not merged["pathogen"]:
        missing.append("No pathogen name found")
    elif "," in merged["pathogen"]:
        missing.append("Pathogen field has multiple comma-separated names - verify which is correct")
    if not merged["symptoms"]:
        missing.append("No symptoms found")
    if not merged["management"]:
        missing.append("No management practices found")
    if not merged["spread"]:
        missing.append("No spread/transmission info found")
    if not merged["disease_cycle"]:
        missing.append("No disease cycle info found")
    if not merged["preventive_advice"]:
        missing.append("No preventive advice found")
    if contains_chemical_leak(merged["management"]):
        missing.append("CHEMICAL SCOPE LEAK in management - review and remove chemical/dosage content")
    if contains_chemical_leak(merged["preventive_advice"]):
        missing.append("CHEMICAL SCOPE LEAK in preventive_advice - review and remove chemical/dosage content")
    return missing


review_rows = []

for _, row in disease_rows.iterrows():
    file_path = row["file_path"]
    crop = row["crop"]
    disease = row["disease"]

    out_name = os.path.splitext(os.path.basename(file_path))[0] + "_distilled.json"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if os.path.exists(out_path):
        print(f"SKIPPING (already exists, not touching your corrections): {out_path}")
        continue

    print(f"Distilling: {file_path}")

    text = load_extracted_text(file_path)
    if not text or len(text.strip()) < 50:
        print(f"  SKIPPED: no usable extracted text (check extraction/OCR output first)")
        continue

    chunks = split_into_chunks(text)
    print(f"  {len(chunks)} chunk(s) to process")

    chunk_results = []
    for i, chunk in enumerate(chunks):
        result = call_llm(crop, disease, chunk)
        if result is None:
            print(f"    chunk {i}: LLM output could not be parsed, skipped")
        chunk_results.append(result)

    merged = merge_chunk_results(chunk_results)
    merged["crop"] = crop
    merged["disease"] = disease
    merged["source_file"] = file_path
    merged["source"] = {
        "country": clean_json_value(row.get("country", "")),
        "source_type": clean_json_value(row.get("source", "")),
        "year": clean_json_value(row.get("year", ""))
    }
    merged["missing_information"] = compute_missing_info(merged)
    merged["extraction_quality"] = "needs_review" if merged["missing_information"] else "complete"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2, allow_nan=False)

    review_rows.append({
        "crop": crop,
        "disease": disease,
        "pathogen": merged["pathogen"],
        "symptoms": " | ".join(merged["symptoms"]),
        "affected_parts": " | ".join(merged["affected_parts"]),
        "spread": merged["spread"],
        "disease_cycle": merged["disease_cycle"],
        "management": " | ".join(merged["management"]),
        "preventive_advice": merged["preventive_advice"],
        "missing_information": " | ".join(merged["missing_information"]),
        "extraction_quality": merged["extraction_quality"],
        "source_file": file_path,
        "needs_review": "YES"
    })

    print(f"  OK -> {out_path} ({merged['extraction_quality']})")

review_df = pd.DataFrame(review_rows)
review_df.to_excel(REVIEW_XLSX, index=False)

print(f"\nDone. {len(review_rows)} documents distilled.")
print(f"Review spreadsheet saved: {REVIEW_XLSX}")
print("Sort by 'extraction_quality' to prioritize rows marked 'needs_review',")
print("and check 'missing_information' for what specifically to verify against")
print("the original PDFs before using this content in your RAG.")


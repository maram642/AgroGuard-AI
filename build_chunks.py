"""
build_chunks.py

Builds ONE combined JSON record per (crop, disease) by merging:
    - distilled disease information from ALL source files for that disease
    - all matched pesticide treatments

Output:
    chunks.jsonl

Important:
    - Crop/disease matching is normalized.
    - Disease aliases are supported.
    - Multiple pesticide products for the same crop+disease are preserved.
    - Multiple SOURCE FILES for the same crop+disease are merged into a
      single chunk instead of producing duplicate chunks.
    - Each chunk's metadata now includes:
        * "products": structured list of {produit, substance_active,
          dosage, homologation, utilisation, confiance} -- built directly
          from the pesticide Excel rows, never left for an LLM to
          reconstruct from free text.
        * "management_raw" / "preventive_advice_raw": the RAW, unmerged
          list of items from the source fact sheets (before being joined
          into a single text blob). These are consumed by
          precompute_translations.py to generate controlled,
          human-reviewable FR/EN summaries -- never by an LLM reading
          the full free-text chunk.
"""

import os
import json
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DISTILLED_DIR = "distilled_mistral"
PESTICIDE_MATCHES_FILE = "pesticide_treatment_matches_final.xlsx"
OUTPUT_FILE = "chunks.jsonl"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for matching.

    Examples:
        Tomato          -> tomato
        TOMATO          -> tomato
        powdery_mildew  -> powdery mildew
        powdery-mildew  -> powdery mildew
        Late_Blight     -> late blight
    """

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Replace underscores and hyphens with spaces
    value = value.replace("_", " ")
    value = value.replace("-", " ")

    # Remove parentheses
    value = value.replace("(", " ")
    value = value.replace(")", " ")

    # Remove extra spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ============================================================
# DISEASE ALIASES
# ============================================================

DISEASE_ALIASES = {

    # POWDERY MILDEW
    "powdery mildew": "powdery mildew",
    "powdery mildew oidium": "powdery mildew",
    "powdery mildew oidium spp": "powdery mildew",

    # LATE BLIGHT
    "late blight": "late blight",

    # SEPTORIA
    "septoria leaf blotch": "septoria leaf blotch",
    "septoria leaf spot": "septoria leaf blotch",
    "septoriose": "septoria leaf blotch",

    # FUSARIUM
    "fusarium head blight": "fusarium head blight",
    "fusarium head blight fusariose": "fusarium head blight",

    # RUST
    "rust": "rust",
    "rouille": "rust",
    "leaf rust": "leaf rust",
    "brown rust": "leaf rust",
    "rouille brune": "leaf rust",
    "stem rust": "stem rust",
    "rouille noire": "stem rust",
    "stripe rust": "stripe rust",
    "yellow rust": "stripe rust",
    "rouille jaune": "stripe rust",

    # EARLY BLIGHT
    "early blight": "early blight",
    "alternariose": "early blight",

    # DOWNY MILDEW
    "downy mildew": "downy mildew",
    "mildiou": "downy mildew",

    # GUMMOSIS
    "gummosis": "gummosis",
    "gommose": "gummosis",

    # LEAF SPOT FAMILY (kept distinct on purpose -- do not merge)
    "leaf spot": "leaf spot",
    "fruit and leaf spot": "fruit and leaf spot",
    "black spot": "black spot",
    "cercospora leaf spot": "cercospora leaf spot",
    "bacterial leaf spot": "bacterial leaf spot",

    # PHOMOPSIS (grapevine) -- both spellings map to the same target
    "phomopsis leaf spot": "phomopsis leaf spot",
    "phomopsis cane and leaf spot": "phomopsis leaf spot",

    # PEACOCK SPOT (olive)
    "peacock spot": "peacock spot",
    "oeil de paon": "peacock spot",

    # PHYTOPHTHORA BLIGHT (pepper)
    "phytophthora blight": "phytophthora blight",
}


def normalize_disease(value):
    """
    Normalize disease name and apply aliases.
    """

    value = normalize_text(value)

    if value in DISEASE_ALIASES:
        return DISEASE_ALIASES[value]

    return value


def normalize_crop(value):
    """
    Normalize crop names.
    """

    value = normalize_text(value)

    CROP_ALIASES = {
        "tomate": "tomato",
        "tomatoes": "tomato",

        "pomme de terre": "potato",
        "pommes de terre": "potato",

        "ble": "wheat",
        "blé": "wheat",

        "poivron": "pepper",

        "vigne": "grapevine",
        "grape vine": "grapevine",

        "figuier": "fig",

        "agrume": "citrus",
        "agrumes": "citrus",

        "olivier": "olive",
    }

    return CROP_ALIASES.get(value, value)


# ============================================================
# DISEASE TEXT (per single source file)
# ============================================================

def make_disease_text(data):

    parts = []

    if data.get("pathogen"):
        parts.append(f"Pathogen: {data['pathogen']}.")

    if data.get("symptoms"):
        parts.append("Symptoms: " + "; ".join(data["symptoms"]) + ".")

    if data.get("affected_parts"):
        parts.append("Affected parts: " + ", ".join(data["affected_parts"]) + ".")

    if data.get("spread"):
        parts.append(f"Spread: {data['spread']}.")

    if data.get("disease_cycle"):
        parts.append(f"Disease cycle: {data['disease_cycle']}.")

    if data.get("management"):
        parts.append("Cultural management: " + "; ".join(data["management"]) + ".")

    if data.get("preventive_advice"):
        pa = data["preventive_advice"]
        if isinstance(pa, list):
            pa = " ".join(pa)
        parts.append(f"Preventive advice: {pa}")

    return " ".join(parts)


# ============================================================
# MERGE MULTIPLE SOURCE FILES FOR THE SAME (crop, disease)
# ============================================================

PATHOGEN_SYNONYMS = {
    # Olive peacock spot: anamorph/teleomorph names for the same fungus
    "fusicladium oleaginum": "venturia oleaginea",
    "spilocaea oleaginea": "venturia oleaginea",
    "venturia oleaginea": "venturia oleaginea",
    # Wheat septoria leaf blotch: taxonomic rename, same fungus
    "zymoseptoria tritici": "septoria tritici",
    "septoria tritici": "septoria tritici",
}


def normalize_pathogen_for_comparison(p):
    """
    Loose normalization used ONLY to decide whether a disease group
    genuinely contains multiple distinct causal species (in which case we
    must not silently apply one species' registered treatment to another),
    or whether it's really the same organism referred to slightly
    differently across files/citations (synonym, author citation variant,
    parenthetical alternate name, punctuation) -- in which case treating
    it as "ambiguous" would be a false positive.
    """
    if not p:
        return ""
    v = p.lower().strip()
    # Drop parenthetical alternate names, e.g. "Fusicladium oleaginum (Spilocaea oleaginea)"
    v = re.sub(r"\([^)]*\)", "", v)
    # Drop taxonomic author citations, e.g. "Ericks & E. Henn", "(Pers.) Speg."
    v = re.sub(r"\b[a-z]\.\s?[a-z]+\.?( & )?", "", v)
    v = re.sub(r"\s+", " ", v).strip().rstrip(".")
    return PATHOGEN_SYNONYMS.get(v, v)


def merge_disease_group(data_list, key, pesticide_lookup, pesticide_lookup_by_pathogen):
    """
    data_list: list of distilled JSON dicts, all sharing the same
    normalized (crop, disease) key, but potentially describing different
    pathogens/papers (e.g. 3 papers on Tomato Powdery Mildew, each about
    a different causal species).

    For each source, we first look for a PATHOGEN-SPECIFIC catalog match
    (crop, disease, pathogen). If the catalog names a specific species for
    this product, it must not be silently applied to other GENUINELY
    DIFFERENT pathogens that happen to share the same disease name.

    Multiple source files that describe the SAME organism (exact match,
    or a known synonym/citation variant) are treated as a single,
    unambiguous pathogen -- the treatment is attached once, with no
    "not confirmed" caveat, exactly as if there had been only one file.

    Only if the group contains genuinely distinct pathogen species do we
    fall back to per-section, pathogen-specific attachment with caveats
    for species that have no confirmed match.

    Also collects, in parallel:
        - all_products: structured pesticide product dicts
        - all_management_items: raw (unmerged) "management" list items
        - all_preventive_items: raw (unmerged) "preventive_advice" items

    Returns: (combined_text, pathogens, source_files, display_crop,
              display_disease, overall_status, all_products,
              all_management_items, all_preventive_items)
    """

    crop, disease = key

    raw_pathogens = [d.get("pathogen") for d in data_list if d.get("pathogen")]
    distinct_normalized = set(normalize_pathogen_for_comparison(p) for p in raw_pathogens)
    genuinely_multi_pathogen = len(distinct_normalized) > 1

    generic_pesticide_text, generic_status, generic_products = (
        make_pesticide_text(pd.DataFrame(pesticide_lookup[key]))
        if key in pesticide_lookup else (None, None, [])
    )

    text_sections = []
    pathogens = []
    source_files = []
    any_found = False
    any_low_confidence = False
    all_products = []
    all_management_items = []
    all_preventive_items = []

    for data in data_list:
        section = make_disease_text(data)
        pathogen = data.get("pathogen")
        if pathogen:
            pathogens.append(pathogen)
        if data.get("source_file"):
            source_files.append(data["source_file"])
        if len(data_list) > 1 and pathogen:
            section = f"[Source: {pathogen}] {section}"
        if section:
            text_sections.append(section)

        # Collect RAW, unmerged advice items -- used by
        # precompute_translations.py for controlled LLM reformulation.
        # These are never handed to an LLM as free text; only the
        # extracted list items are.
        if data.get("management"):
            all_management_items.extend(data["management"])
        if data.get("preventive_advice"):
            pa = data["preventive_advice"]
            if isinstance(pa, list):
                all_preventive_items.extend(pa)
            else:
                all_preventive_items.append(pa)

    combined_text = " ".join(text_sections)

    if genuinely_multi_pathogen:
        # Re-run per-section, attaching pathogen-specific treatment where
        # available and an explicit "not confirmed" note otherwise.
        text_sections = []
        for data in data_list:
            section = make_disease_text(data)
            pathogen = data.get("pathogen")
            specific_key = (crop, disease, normalize_pathogen_for_comparison(pathogen)) if pathogen else None

            if specific_key and specific_key in pesticide_lookup_by_pathogen:
                treat_text, treat_status, treat_products = make_pesticide_text(
                    pd.DataFrame(pesticide_lookup_by_pathogen[specific_key])
                )
                section += f" Treatment (specific to {pathogen}): {treat_text}"
                any_found = any_found or (treat_status == "FOUND")
                any_low_confidence = any_low_confidence or (treat_status == "FOUND_LOW_CONFIDENCE")
                for p in treat_products:
                    tagged = dict(p)
                    tagged["source_pathogen"] = pathogen
                    all_products.append(tagged)
            elif generic_pesticide_text:
                section += (
                    f" Treatment: {generic_pesticide_text} "
                    f"[Note: this registration does not name a specific causal "
                    f"species — applicability to {pathogen} specifically has not "
                    f"been separately confirmed.]"
                )
                any_found = any_found or (generic_status == "FOUND")
                any_low_confidence = any_low_confidence or (generic_status == "FOUND_LOW_CONFIDENCE")
                for p in generic_products:
                    tagged = dict(p)
                    tagged["source_pathogen"] = pathogen
                    tagged["confiance"] = "medium"  # not species-confirmed, downgrade
                    all_products.append(tagged)
            else:
                section += (
                    f" Treatment: No species-specific registered pesticide "
                    f"treatment was confirmed for {pathogen} in the official "
                    f"Tunisian catalog."
                )

            if pathogen:
                section = f"[Source: {pathogen}] {section}"
            text_sections.append(section)

        combined_text = " ".join(text_sections)
    else:
        # Single organism (possibly cited across multiple files/synonyms) --
        # attach the treatment once, at the end, exactly as for a
        # single-source disease.
        pathogen = raw_pathogens[0] if raw_pathogens else None
        specific_key = (crop, disease, normalize_pathogen_for_comparison(pathogen)) if pathogen else None

        if specific_key and specific_key in pesticide_lookup_by_pathogen:
            treat_text, treat_status, treat_products = make_pesticide_text(
                pd.DataFrame(pesticide_lookup_by_pathogen[specific_key])
            )
            combined_text += f" Treatment: {treat_text}"
            any_found = any_found or (treat_status == "FOUND")
            any_low_confidence = any_low_confidence or (treat_status == "FOUND_LOW_CONFIDENCE")
            all_products.extend(treat_products)
        elif generic_pesticide_text:
            combined_text += f" Treatment: {generic_pesticide_text}"
            any_found = any_found or (generic_status == "FOUND")
            any_low_confidence = any_low_confidence or (generic_status == "FOUND_LOW_CONFIDENCE")
            all_products.extend(generic_products)

    seen = set()
    unique_pathogens = []
    for p in pathogens:
        if p not in seen:
            seen.add(p)
            unique_pathogens.append(p)

    display_crop = data_list[0].get("crop", "")
    display_disease = data_list[0].get("disease", "")

    if any_found:
        overall_status = "FOUND"
    elif any_low_confidence:
        overall_status = "FOUND_LOW_CONFIDENCE"
    elif generic_status:
        overall_status = generic_status
    else:
        overall_status = "NO_MATCH_ATTEMPTED"

    return (combined_text, unique_pathogens, source_files, display_crop,
            display_disease, overall_status, all_products,
            all_management_items, all_preventive_items)


# ============================================================
# PESTICIDE TEXT
# ============================================================

def make_pesticide_text(rows):
    """
    Build pesticide treatment text AND a structured products list from
    ALL matching rows.

    rows = DataFrame of Excel rows for the same crop+disease.

    Returns:
        (text, overall_status, products_list)
    """

    treatment_parts = []
    statuses = []
    products = []

    for _, row in rows.iterrows():

        status = normalize_text(row.get("status", ""))
        statuses.append(status)

        if status in ("found", "found low confidence"):

            product_name = str(row.get("product_name", "")).strip()
            substance_active = str(row.get("substance_active", "")).strip()
            dosage = str(row.get("dosage", "")).strip()
            homologation_number = str(row.get("homologation_number", "")).strip()
            utilization = str(row.get("utulisation", "")).strip()

            treatment = f"{product_name}"

            if substance_active:
                treatment += f" (active substance: {substance_active})"

            if dosage:
                treatment += f", dosage: {dosage}"

            if homologation_number:
                treatment += f", homologation number: {homologation_number}"

            if utilization and utilization not in ("", "None", "—", "nan"):
                treatment += f", utilization: {utilization}"

            if status == "found low confidence":
                treatment += " [LOW CONFIDENCE - manual verification recommended]"

            treatment_parts.append(treatment)

            # Structured entry -- consumed directly by the API, never
            # regenerated by an LLM. Key is "homologation" (not
            # "homologation_number") to match the schema already used
            # downstream in rag_chain.py / api.py.
            products.append({
                "produit": product_name,
                "substance_active": substance_active,
                "dosage": dosage,
                "homologation": homologation_number,
                "utilisation": utilization if utilization not in ("", "None", "—", "nan") else None,
                "confiance": "medium" if status == "found low confidence" else "haute",
            })

    if treatment_parts:
        text = "Registered pesticide treatments in Tunisia: " + "; ".join(treatment_parts) + "."
        overall_status = "FOUND" if "found" in statuses else "FOUND_LOW_CONFIDENCE"
        return text, overall_status, products

    if "confirmed none" in statuses:
        return (
            "No chemical pesticide is registered for this target in the "
            "official Tunisian pesticide catalog. Management should rely "
            "on cultural and preventive practices only.",
            "CONFIRMED_NONE",
            [],
        )

    if "not found" in statuses or "not found in catalog" in statuses:
        return (
            "No matching registered pesticide treatment was identified for "
            "this target in the available official Tunisian pesticide "
            "catalog data.",
            "NOT_FOUND_IN_CATALOG",
            [],
        )

    return None, "UNRESOLVED", []


# ============================================================
# LOAD PESTICIDE EXCEL
# ============================================================

print("Loading pesticide Excel...")

pesticide_df = pd.read_excel(PESTICIDE_MATCHES_FILE)

print(f"Loaded {len(pesticide_df)} pesticide rows.")


# ============================================================
# CREATE PESTICIDE LOOKUP
# ============================================================

pesticide_lookup = defaultdict(list)          # key: (crop, disease)              -> generic, no pathogen specified
pesticide_lookup_by_pathogen = defaultdict(list)  # key: (crop, disease, pathogen) -> pathogen-specific catalog entries

for _, row in pesticide_df.iterrows():
    crop = normalize_crop(row.get("crop", ""))
    disease = normalize_disease(row.get("disease", ""))
    pathogen_raw = row.get("pathogen", None)
    pathogen = str(pathogen_raw).strip() if pathogen_raw and str(pathogen_raw).strip().lower() not in ("", "nan", "none") else None

    if pathogen:
        pesticide_lookup_by_pathogen[(crop, disease, normalize_pathogen_for_comparison(pathogen))].append(row)
    else:
        pesticide_lookup[(crop, disease)].append(row)

print(f"Created {len(pesticide_lookup)} generic crop+disease pesticide target(s) "
      f"and {len(pesticide_lookup_by_pathogen)} pathogen-specific target(s).")


# ============================================================
# STEP 1 -- GROUP DISTILLED SOURCE FILES BY (crop, disease)
# ============================================================

grouped_by_disease = defaultdict(list)

for fname in sorted(os.listdir(DISTILLED_DIR)):

    if not fname.endswith(".json"):
        continue

    file_path = os.path.join(DISTILLED_DIR, fname)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    crop = normalize_crop(data.get("crop", ""))
    disease = normalize_disease(data.get("disease", ""))

    grouped_by_disease[(crop, disease)].append(data)

print(f"Grouped {sum(len(v) for v in grouped_by_disease.values())} source file(s) "
      f"into {len(grouped_by_disease)} unique crop+disease group(s).")


# ============================================================
# STEP 2 -- BUILD ONE CHUNK PER (crop, disease) GROUP
# ============================================================

chunks = []
missing_pesticide_match = []
matched_pesticide_match = []

for key, data_list in sorted(grouped_by_disease.items()):

    crop, disease = key

    (combined_disease_text, pathogens, source_files, display_crop,
     display_disease, pesticide_status, products,
     management_items, preventive_items) = merge_disease_group(
        data_list, key, pesticide_lookup, pesticide_lookup_by_pathogen
    )

    combined_text = combined_disease_text

    # Track match/no-match for the summary printout at the end.
    if pesticide_status in ("FOUND", "FOUND_LOW_CONFIDENCE"):
        matched_pesticide_match.append(f"{crop} / {disease}")
    elif pesticide_status == "NO_MATCH_ATTEMPTED":
        missing_pesticide_match.append(f"{display_crop} / {display_disease}")

    chunk_id = f"{crop}::{disease}".replace(" ", "_")

    # Dedupe (keep order) in case multiple sources repeat the same item.
    seen_mgmt = set()
    unique_management = [
        x for x in management_items if not (x in seen_mgmt or seen_mgmt.add(x))
    ]
    seen_prev = set()
    unique_preventive = [
        x for x in preventive_items if not (x in seen_prev or seen_prev.add(x))
    ]

    chunks.append({
        "id": chunk_id,
        "text": combined_text,
        "metadata": {
            "crop": display_crop,
            "disease": display_disease,
            "normalized_crop": crop,
            "normalized_disease": disease,
            "pathogen": "; ".join(pathogens),
            "pesticide_status": pesticide_status,
            "source_files": source_files,
            "num_source_files": len(source_files),
            "products": products,
            "management_raw": unique_management,
            "preventive_advice_raw": unique_preventive,
        },
    })


# ============================================================
# WRITE OUTPUT
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ============================================================
# SUMMARY
# ============================================================

print()
print(f"Built {len(chunks)} combined disease+pesticide chunk(s), saved to {OUTPUT_FILE}")

print()
print(f"Successfully matched {len(matched_pesticide_match)} disease(s) with pesticide data.")

if missing_pesticide_match:
    print()
    print(f"{len(missing_pesticide_match)} disease(s) had NO pesticide match entry.")
    print("These were not automatically treated as CONFIRMED_NONE.")
    print()
    for disease in missing_pesticide_match:
        print(f"  - {disease}")

if chunks:
    print()
    print("Sample chunk:")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2))

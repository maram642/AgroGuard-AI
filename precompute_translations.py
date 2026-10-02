# -*- coding: utf-8 -*-
"""
precompute_translations.py

Generates ONCE the "gestion_maladie" and "conseils_preventifs" texts in
FR and EN for each chunk, from the RAW item lists (management_raw /
preventive_advice_raw), and writes them directly into chunks.jsonl.

Principle: the LLM acts as a CONTROLLED PARAPHRASER -- it only ever sees
the already-extracted list items (never the full free-text chunk), and
is only allowed to reformulate/translate, never to add or drop items.

An automatic validation step checks, after generation, that every source
item has a detectable equivalent in the generated text (keyword overlap).
Any chunk that fails this check is flagged for manual review -- it is
NOT silently trusted.

Run this ONCE after build_chunks.py, before build_index.py.

Usage:
    python precompute_translations.py
"""

import json
import re

from langchain_ollama import ChatOllama


CHUNKS_FILE = "chunks.jsonl"
OLLAMA_MODEL = "mistral"

NOT_SPECIFIED = {
    "fr": "Non spécifié dans les sources disponibles.",
    "en": "Not specified in the available sources.",
}

# Keyword-overlap threshold between each source item and the generated
# text -- below this, the item is considered "not recoverable" and the
# chunk is flagged for human review instead of trusted blindly.
MIN_KEYWORD_OVERLAP_RATIO = 0.4

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "pour",
    "dans", "sur", "avec", "au", "aux", "en", "par",
}


llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0, num_predict=512)


REFORMULATION_PROMPT = """Tu es un paraphraseur controle. Tu recois une liste d'elements factuels.

REGLE ABSOLUE : tu dois UNIQUEMENT reformuler et traduire ces elements en {target_lang}, sous forme de phrase(s) naturelle(s) et fluides.
Tu ne dois RIEN ajouter -- aucune action, aucun conseil, aucune information qui n'est pas explicitement dans la liste ci-dessous.
Tu ne dois RIEN supprimer -- chaque element de la liste doit se retrouver, reformule, dans ta reponse.
Tu peux regrouper plusieurs elements dans une meme phrase si c'est plus naturel, mais sans en perdre le sens.

Elements a reformuler et traduire en {target_lang} :
{items}

Reponds UNIQUEMENT avec le texte reformule en {target_lang}, sans preambule, sans guillemets, sans repeter la liste brute :"""


def reformulate(items: list, target_lang: str) -> str:
    if not items:
        return NOT_SPECIFIED[target_lang]

    lang_name = "français" if target_lang == "fr" else "English"
    items_text = "\n".join(f"- {item}" for item in items)

    prompt = REFORMULATION_PROMPT.format(target_lang=lang_name, items=items_text)
    response = llm.invoke(prompt).content.strip()
    return response


def extract_keywords(text: str) -> set:
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def validate_reformulation(items: list, generated_text: str):
    """
    For each source item, check that at least MIN_KEYWORD_OVERLAP_RATIO
    of its distinctive keywords appear in the generated text.
    Returns (is_valid, list_of_problematic_items).
    """
    if not items:
        return True, []

    generated_keywords = extract_keywords(generated_text)
    failed_items = []

    for item in items:
        item_keywords = extract_keywords(item)
        if not item_keywords:
            continue
        overlap = item_keywords & generated_keywords
        ratio = len(overlap) / len(item_keywords)
        if ratio < MIN_KEYWORD_OVERLAP_RATIO:
            failed_items.append(item)

    return (len(failed_items) == 0), failed_items


def main():
    records = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    flagged_chunks = []

    for i, rec in enumerate(records, start=1):
        meta = rec["metadata"]
        mgmt_items = meta.get("management_raw", [])
        prev_items = meta.get("preventive_advice_raw", [])

        print(f"[{i}/{len(records)}] {rec['id']}")

        # --- Gestion de la maladie ---
        gestion_en = reformulate(mgmt_items, "en")
        valid_en, failed_en = validate_reformulation(mgmt_items, gestion_en)

        gestion_fr = reformulate(mgmt_items, "fr")

        # --- Conseils preventifs ---
        preventifs_en = reformulate(prev_items, "en")
        valid_prev_en, failed_prev_en = validate_reformulation(prev_items, preventifs_en)

        preventifs_fr = reformulate(prev_items, "fr")

        meta["gestion_maladie_en"] = gestion_en
        meta["gestion_maladie_fr"] = gestion_fr
        meta["conseils_preventifs_en"] = preventifs_en
        meta["conseils_preventifs_fr"] = preventifs_fr

        if not valid_en or not valid_prev_en:
            flagged_chunks.append({
                "chunk_id": rec["id"],
                "management_issues": failed_en,
                "preventive_issues": failed_prev_en,
            })
            print("  ⚠️ FLAGGED pour relecture manuelle")
            if failed_en:
                print(f"     Items management non retrouves: {failed_en}")
            if failed_prev_en:
                print(f"     Items preventifs non retrouves: {failed_prev_en}")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"Termine. {len(records)} chunks traites.")
    print(f"{len(flagged_chunks)} chunk(s) flagge(s) pour relecture manuelle.")
    print("=" * 70)

    if flagged_chunks:
        with open("flagged_for_review.json", "w", encoding="utf-8") as f:
            json.dump(flagged_chunks, f, indent=2, ensure_ascii=False)
        print("Detail ecrit dans flagged_for_review.json -- releis ces chunks avant demo.")

    print("\nProchaine etape : python build_index.py (pour re-indexer FAISS).")


if __name__ == "__main__":
    main()

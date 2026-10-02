# -*- coding: utf-8 -*-
"""
test_rag_batch.py

Batch sanity check for rag_chain.RecommendationEngine, covering the edge
cases NOT yet exercised by the manual tests in rag_chain.py's __main__:

    - CONFIRMED_NONE exact match (no pesticide registered at all)
    - pathogen-specific match (tomato powdery mildew / Leveillula taurica)
    - fuzzy match (deliberate typo)
    - a crop that doesn't exist in the corpus at all

For each case, prints a compact summary (match type, confidence, answer
length, and a few red-flag checks) rather than the full answer, so you can
scan many cases quickly. Prints the full answer only when a red flag is
found, or if --verbose is passed.

Usage:
    python test_rag_batch.py
    python test_rag_batch.py --verbose
"""

import sys
from rag_chain import RecommendationEngine


TEST_CASES = [
    {
        "label": "CONFIRMED_NONE exact match (fig rust)",
        "crop": "fig",
        "disease": "fig rust",
        "expect_match_type": "exact",
        "expect_no_product": True,   # answer should NOT invent a chemical product
    },
    {
        "label": "CONFIRMED_NONE exact match (potato early blight)",
        "crop": "potato",
        "disease": "early-blight",
        "expect_match_type": "exact",
        "expect_no_product": True,
    },
    {
        "label": "Pathogen-specific (tomato powdery mildew / Leveillula taurica)",
        "crop": "tomato",
        "disease": "powdery mildew",
        "expect_match_type": "exact",
        "expect_no_product": False,
    },
    {
        "label": "Fuzzy match (deliberate typo: 'Late Blightt')",
        "crop": "tomato",
        "disease": "Late Blightt",
        "expect_match_type": "fuzzy",
        "expect_no_product": False,
    },
    {
        "label": "Unknown crop entirely (mango, not in corpus)",
        "crop": "mango",
        "disease": "anthracnose",
        "expect_match_type": "semantic",
        "expect_no_product": None,   # no strong expectation, just observe
    },
    {
        "label": "Known crop, disease close to another crop's disease name",
        "crop": "wheat",
        "disease": "rust",  # ambiguous: leaf/stem/stripe rust all exist
        "expect_match_type": None,   # exact/fuzzy/semantic all plausible, just observe
        "expect_no_product": False,
    },
]

# Phrases that would indicate a hallucinated chemical recommendation on a
# CONFIRMED_NONE case, or generic invented safety boilerplate.
SUSPICIOUS_PHRASES = [
    "voir le contexte", "voir les dosages", "voir la p\u00e9riode",
    "lire l'\u00e9tiquette", "lire les instructions", "porter des gants",
    "protection individuelle recommand\u00e9e",
    "sont sp\u00e9cifi\u00e9es dans le contexte", "sont sp\u00e9cifi\u00e9s dans le contexte",
    "est sp\u00e9cifi\u00e9e dans le contexte", "est sp\u00e9cifi\u00e9 dans le contexte",
    "sont disponibles dans le contexte", "comme indiqu\u00e9 ci-dessus",
    "comme mentionn\u00e9 ci-dessus", "sont list\u00e9s dans le contexte",
]


def check_red_flags(result, case):
    flags = []
    answer_lower = result["answer"].lower()

    for phrase in SUSPICIOUS_PHRASES:
        if phrase in answer_lower:
            flags.append(f"suspicious phrase found: '{phrase}'")

    if case.get("expect_no_product") is True:
        # crude check: a CONFIRMED_NONE case shouldn't mention a dosage unit
        if any(unit in answer_lower for unit in ["g/hl", "l/ha", "cc/hl", "kg/ha", "ml/ha"]):
            flags.append("CONFIRMED_NONE case mentions a dosage unit -- possible hallucinated product")

    if case.get("expect_match_type") and result["match_type"] != case["expect_match_type"]:
        flags.append(
            f"expected match_type='{case['expect_match_type']}', got '{result['match_type']}'"
        )

    return flags


def main():
    verbose = "--verbose" in sys.argv

    print("Loading RecommendationEngine (this loads chunks + connects to Ollama)...")
    engine = RecommendationEngine()
    print(f"Loaded. {len(TEST_CASES)} test case(s) to run.\n")

    total_flags = 0

    for case in TEST_CASES:
        print("=" * 78)
        print(f"CASE: {case['label']}")
        print(f"  crop={case['crop']!r}  disease={case['disease']!r}")
        print("-" * 78)

        result = engine.recommend(case["crop"], case["disease"])
        flags = check_red_flags(result, case)

        print(f"  match_type={result['match_type']}  confidence={result['confidence']}  "
              f"chunks={result['chunk_ids']}  pesticide_status={result['pesticide_status']}")
        print(f"  answer length: {len(result['answer'])} chars")

        if flags:
            total_flags += len(flags)
            print(f"  \u26a0\ufe0f  {len(flags)} RED FLAG(S):")
            for f in flags:
                print(f"     - {f}")
            print()
            print("  --- FULL ANSWER (shown because of red flags) ---")
            print(result["answer"])
        elif verbose:
            print()
            print("  --- FULL ANSWER ---")
            print(result["answer"])
        else:
            print("  \u2713 no red flags detected")

        print()

    print("=" * 78)
    print(f"SUMMARY: {total_flags} total red flag(s) across {len(TEST_CASES)} case(s).")
    if total_flags == 0:
        print("\u2713 All checks passed.")
    else:
        print("\u26a0\ufe0f  Review the flagged cases above before moving on.")
    print("=" * 78)


if __name__ == "__main__":
    main()

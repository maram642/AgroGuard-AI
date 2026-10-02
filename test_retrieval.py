# -*- coding: utf-8 -*-
"""
test_retrieval.py

Tests the FAISS retrieval quality after building the index.

The test queries are structured as:
    Crop + Disease + Requested information

This is important because the RAG pipeline receives a detected
plant/crop and disease rather than a completely free-form question.

Usage:
    python test_retrieval.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

QUERY_PREFIX = "query: "

TOP_K = 5


# ============================================================
# TEST QUERIES
# ============================================================
#
# We explicitly provide crop + disease.
#
# This lets us test whether the embedding model can distinguish:
#
#     tomato + powdery mildew
#
# from:
#
#     pepper + powdery mildew
#
# and:
#
#     wheat + rust
#
# from:
#
#     fig + rust
#
# ============================================================

TEST_QUERIES = [
    {
        "label": "Tomato late blight / mildiou",
        "query": (
            "Crop: tomato. "
            "Disease: late blight (mildiou). "
            "Requested information: treatment."
        ),
        "expected_id": "tomato::late_blight",
    },

    {
        "label": "Citrus black spot",
        "query": (
            "Crop: citrus. "
            "Disease: black spot. "
            "Requested information: treatment."
        ),
        "expected_id": "citrus::black_spot",
    },

    {
        "label": "Pepper powdery mildew / oïdium",
        "query": (
            "Crop: pepper (poivron). "
            "Disease: powdery mildew (oïdium). "
            "Requested information: dosage."
        ),
        "expected_id": "pepper::powdery_mildew",
    },

    {
        "label": "Wheat rust",
        "query": (
            "Crop: wheat (blé). "
            "Disease: rust. "
            "Requested information: fungicide treatment."
        ),
        # There are multiple wheat rust diseases in the corpus.
        # Therefore we don't force one exact ID here.
        "expected_id": None,
    },

    {
        "label": "Olive peacock spot",
        "query": (
            "Crop: olive (olivier). "
            "Disease: peacock spot. "
            "Requested information: treatment."
        ),
        "expected_id": "olive::peacock_spot",
    },
]


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        f"Loading embedding model "
        f"'{EMBEDDING_MODEL}' ..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print(
        f"Loading FAISS index "
        f"from '{INDEX_DIR}/' ..."
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    total = len(TEST_QUERIES)
    exact_top1 = 0

    for test in TEST_QUERIES:

        label = test["label"]
        query = test["query"]
        expected_id = test["expected_id"]

        print("\n" + "=" * 80)
        print(f"TEST: {label}")
        print("=" * 80)

        print(f"QUERY: {query}")

        # ----------------------------------------------------
        # Important:
        # E5 expects "query:" for the query side.
        # ----------------------------------------------------

        formatted_query = QUERY_PREFIX + query

        results = vectorstore.similarity_search_with_score(
            formatted_query,
            k=TOP_K,
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        for rank, (doc, score) in enumerate(
            results,
            start=1
        ):

            meta = doc.metadata

            chunk_id = meta.get(
                "chunk_id",
                ""
            )

            crop = meta.get(
                "crop",
                ""
            )

            disease = meta.get(
                "disease",
                ""
            )

            status = meta.get(
                "pesticide_status",
                ""
            )

            print(
                f"\n  #{rank} "
                f"score={score:.4f} "
                f"id={chunk_id} "
                f"crop={crop} "
                f"disease={disease} "
                f"status={status}"
            )

            snippet = (
                doc.page_content[:250]
                .replace("\n", " ")
            )

            print(
                f"       {snippet}..."
            )

        # ----------------------------------------------------
        # Check expected result
        # ----------------------------------------------------

        if expected_id is not None:

            returned_ids = [
                doc.metadata.get(
                    "chunk_id",
                    ""
                )
                for doc, _ in results
            ]

            if expected_id in returned_ids:

                position = (
                    returned_ids.index(
                        expected_id
                    ) + 1
                )

                print(
                    f"\n  ✓ Expected chunk "
                    f"'{expected_id}' found "
                    f"at rank {position}."
                )

                if position == 1:
                    exact_top1 += 1
                    print(
                        "  ✓ TOP-1 MATCH"
                    )

                else:
                    print(
                        "  ⚠ Found, but not TOP-1."
                    )

            else:

                print(
                    f"\n  ✗ Expected chunk "
                    f"'{expected_id}' NOT found "
                    f"in TOP-{TOP_K}."
                )

        else:

            print(
                "\n  ℹ No single expected ID: "
                "multiple wheat rust diseases "
                "exist in the corpus."
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    exact_tests = sum(
        1
        for test in TEST_QUERIES
        if test["expected_id"] is not None
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL TEST SUMMARY")
    print("=" * 80)

    print(
        f"Exact TOP-1 matches: "
        f"{exact_top1}/{exact_tests}"
    )

    print()

    if exact_top1 == exact_tests:
        print(
            "✓ All exact-match tests returned "
            "the expected chunk at TOP-1."
        )
    else:
        print(
            "⚠ Some expected chunks were not "
            "returned at TOP-1."
        )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
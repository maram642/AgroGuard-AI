# -*- coding: utf-8 -*-
"""
build_index.py

Build a local FAISS vector index from chunks.jsonl.

Improvement:
    The embedding text explicitly contains the crop and disease information
    so that the semantic model gives more importance to the crop-disease pair.

Example indexed passage:

    passage: Crop: pepper.
    Disease: powdery_mildew.
    Crop-disease pair: pepper powdery_mildew.
    Pathogen: Leveillula taurica.
    Symptoms: ...

This is especially important when several crops share the same disease,
for example:

    tomato::powdery_mildew
    pepper::powdery_mildew

Usage:
    python build_index.py

Output:
    faiss_index/
"""
import re
import json
import os
import shutil

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = "chunks.jsonl"
INDEX_DIR = "faiss_index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# E5 models expect these prefixes.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Number of examples displayed after loading.
SHOW_EXAMPLES = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Convert metadata values into clean strings.

    Handles:
        None
        strings
        numbers
        lists
        dictionaries
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, list):
        return ", ".join(str(x) for x in value if x is not None)

    if isinstance(value, dict):
        return " ".join(
            f"{key}: {value}"
            for key, value in value.items()
        )

    return str(value).strip()


def first_non_empty(metadata, possible_keys):
    """
    Return the first non-empty metadata value among possible keys.

    This makes the indexer robust if your chunks use slightly
    different field names.
    """

    for key in possible_keys:
        value = clean_value(metadata.get(key, ""))

        if value:
            return value

    return ""


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(record, metadata):
    """
    Build a structured text specifically for the embedding model.

    IMPORTANT:
        The original record['text'] is preserved.
        We simply prepend explicit structured information about
        crop, disease and pathogen.

    This helps distinguish:

        tomato + powdery mildew

    from:

        pepper + powdery mildew

    even when both diseases have the same/similar pathogen.
    """

    original_text = clean_value(record.get("text", ""))

    # Remove a leading "Pathogen: ...." sentence from the original text,
    # since we already prepend pathogen info from metadata below.
    original_text = re.sub(
        r"^Pathogen:\s*[^.]+\.\s*",
        "",
        original_text
    ).strip()

    # --------------------------------------------------------
    # Try several possible metadata field names.
    # --------------------------------------------------------

    crop = first_non_empty(
        metadata,
        [
            "crop",
            "culture",
            "plant",
            "host",
        ]
    )

    disease = first_non_empty(
        metadata,
        [
            "disease",
            "maladie",
            "target",
        ]
    )

    pathogen = first_non_empty(
        metadata,
        [
            "pathogen",
            "agent",
            "causal_agent",
            "organism",
        ]
    )

    # --------------------------------------------------------
    # Build structured prefix.
    # --------------------------------------------------------

    structured_parts = []

    if crop:
        structured_parts.append(
            f"Crop: {crop}."
        )

    if disease:
        structured_parts.append(
            f"Disease: {disease}."
        )

    if crop and disease:
        structured_parts.append(
            f"Crop-disease pair: {crop} {disease}."
        )

    if pathogen:
        structured_parts.append(
            f"Pathogen: {pathogen}."
        )

    # --------------------------------------------------------
    # Combine structured information + original content.
    # --------------------------------------------------------

    structured_text = " ".join(structured_parts)

    if structured_text and original_text:
        return structured_text + " " + original_text

    if structured_text:
        return structured_text

    return original_text


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(path):
    """
    Read chunks.jsonl and convert each record into a LangChain Document.

    The embedding receives:

        passage:
        Crop: ...
        Disease: ...
        Crop-disease pair: ...
        Pathogen: ...
        original chunk text

    Metadata remains attached to the Document for later retrieval.
    """

    documents = []

    with open(path, "r", encoding="utf-8") as f:

        for line_num, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:

                print(
                    f"WARNING: skipping malformed line "
                    f"{line_num}: {e}"
                )

                continue

            # ------------------------------------------------
            # Original text
            # ------------------------------------------------

            text = clean_value(
                record.get("text", "")
            )

            if not text:

                print(
                    f"WARNING: chunk "
                    f"'{record.get('id', line_num)}' "
                    f"has empty text -- skipped"
                )

                continue

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = dict(
                record.get("metadata", {})
            )

            # Keep chunk ID in metadata.
            chunk_id = record.get(
                "id",
                f"line_{line_num}"
            )

            metadata["chunk_id"] = chunk_id

            # ------------------------------------------------
            # Normalize metadata values
            # ------------------------------------------------

            for key, value in list(metadata.items()):

                if value is None:
                    metadata[key] = ""

                elif not isinstance(
                    value,
                    (str, int, float, bool, list)
                ):
                    metadata[key] = str(value)

            # ------------------------------------------------
            # Build improved embedding text
            # ------------------------------------------------

            embedding_text = build_embedding_text(
                record,
                metadata
            )

            # E5 passage prefix.
            page_content = (
                PASSAGE_PREFIX +
                embedding_text
            )

            # ------------------------------------------------
            # Create LangChain Document
            # ------------------------------------------------

            document = Document(
                page_content=page_content,
                metadata=metadata
            )

            documents.append(document)

    return documents


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

def show_examples(documents, n=SHOW_EXAMPLES):
    """
    Display a few indexed documents so we can verify that
    crop and disease are actually included in the embedding text.
    """

    print()
    print("=" * 70)
    print("EXAMPLES OF TEXT SENT TO THE EMBEDDING MODEL")
    print("=" * 70)

    for i, document in enumerate(
        documents[:n],
        start=1
    ):

        print()
        print(f"--- Example {i} ---")

        print(
            f"chunk_id: "
            f"{document.metadata.get('chunk_id', '')}"
        )

        print(
            document.page_content[:1000]
        )

    print()
    print("=" * 70)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index():

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    print(
        f"Loading chunks from "
        f"'{CHUNKS_FILE}' ..."
    )

    documents = load_chunks(
        CHUNKS_FILE
    )

    print(
        f"Loaded {len(documents)} document(s)."
    )

    if not documents:

        raise SystemExit(
            "No documents to index -- "
            "check chunks.jsonl."
        )

    # --------------------------------------------------------
    # Show what will actually be embedded
    # --------------------------------------------------------

    show_examples(documents)

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
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
    # Remove old FAISS index
    # --------------------------------------------------------

    if os.path.exists(INDEX_DIR):

        print()
        print(
            f"Removing old index "
            f"'{INDEX_DIR}/' ..."
        )

        shutil.rmtree(
            INDEX_DIR
        )

    # --------------------------------------------------------
    # Build new FAISS index
    # --------------------------------------------------------

    print()
    print(
        "Embedding documents and "
        "building the FAISS index ..."
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    vectorstore.save_local(
        INDEX_DIR
    )

    print()
    print(
        f"Index saved to "
        f"'{INDEX_DIR}/' "
        f"({len(documents)} vectors)."
    )

    print()
    print("IMPORTANT:")
    print(
        "The index now explicitly embeds "
        "crop + disease + crop-disease pair + pathogen."
    )

    return vectorstore, embeddings


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_index()
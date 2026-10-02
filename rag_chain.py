# -*- coding: utf-8 -*-
"""
rag_chain.py

Hybrid retrieval pipeline for the plant disease treatment recommender.
Bilingual (French / English).

Why hybrid, not pure semantic search:
    Module 4 (upstream, image -> plant/disease detection, built by another
    intern) passes this module a CROP + DISEASE label that is already
    supposed to match one of our known crop/disease chunks. For that
    case, an exact (normalized) dictionary lookup is 100% reliable --
    unlike embedding similarity, which can occasionally rank a
    close-but-wrong disease ahead of the correct one by a very small
    margin. FAISS semantic search is kept ONLY as a fallback for labels
    that don't match anything known (typos, unexpected phrasing,
    synonyms we didn't anticipate) -- and results coming from that
    fallback are explicitly flagged as approximate.

Why NO LLM call in recommend():
    Product/dosage data (structured, from the official catalog) and
    disease-management/preventive-advice text (FR/EN) are BOTH resolved
    by direct lookup from chunks.jsonl metadata -- never generated live.
    The management/preventive texts were precomputed ONCE, offline, by
    precompute_translations.py, from a controlled paraphrasing LLM step
    that only ever saw already-extracted list items (never a full
    free-text chunk), plus an automatic keyword-overlap validation and a
    manual review pass. This guarantees that /recommend is:
        - deterministic (same input -> same output, always)
        - fast (no live LLM inference)
        - auditable (whatever was reviewed offline is exactly what the
          API serves -- no surprises during a live demo)

    Mistral is still loaded here (self.llm) because /chat (in api.py)
    needs it for free-text crop/disease extraction and conversational
    reformulation -- that's a genuinely different task (each user
    question is different, there's no fixed corpus to pre-validate).

Disease-family disambiguation:
    Some disease names are GENERIC umbrella terms that map, in the real
    world, to several DISTINCT diseases with different treatments (e.g.
    French "rouille" / English "rust" can mean leaf rust, stem rust, or
    stripe rust on wheat -- three different diseases, different
    products). If a generic term is requested and the crop has MORE THAN
    ONE specific variant in the corpus, silently falling back to FAISS
    semantic search would blend products from unrelated diseases into
    one answer (a real correctness risk for a pesticide tool). Instead,
    DISEASE_FAMILIES below is checked BEFORE the semantic fallback: if
    exactly one variant exists for the requested crop, that single
    variant is used directly (unambiguous, treated as a normal match);
    if several exist, retrieval returns match_type="ambiguous" with the
    list of options, and the caller is expected to ask the user (or the
    upstream detection module) to be more specific -- never guess.

Usage:
    from rag_chain import RecommendationEngine
    engine = RecommendationEngine()
    result = engine.recommend(crop="tomato", disease="Late Blight", language="fr")
    print(result["gestion_maladie"])
    print(result["conseils_preventifs"])
    print(result["match_type"], result["confidence"])
"""

import json
import re
import difflib
from dataclasses import dataclass, field
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = "chunks.jsonl"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
OLLAMA_MODEL = "mistral"          # matches `ollama pull mistral`
QUERY_PREFIX = "query: "

FUZZY_MATCH_CUTOFF = 0.72         # difflib similarity ratio, 0-1
FAISS_FALLBACK_K = 3              # how many chunks to pull on semantic fallback

SUPPORTED_LANGUAGES = ("fr", "en")
DEFAULT_LANGUAGE = "fr"

NOT_SPECIFIED_TEXT = {
    "fr": "Non spécifié dans les sources disponibles.",
    "en": "Not specified in the available sources.",
}


# ============================================================
# NORMALIZATION -- identical logic to build_chunks.py, kept in
# sync deliberately so a label that matches during indexing also
# matches here at query time.
# ============================================================

def normalize_text(value):
    if value is None:
        return ""
    value = str(value).lower().strip()
    value = value.replace("_", " ").replace("-", " ")
    value = value.replace("(", " ").replace(")", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


DISEASE_ALIASES = {
    "powdery mildew": "powdery mildew",
    "powdery mildew oidium": "powdery mildew",
    "late blight": "late blight",
    "septoria leaf blotch": "septoria leaf blotch",
    "septoria leaf spot": "septoria leaf blotch",
    "septoriose": "septoria leaf blotch",
    "fusarium head blight": "fusarium head blight",
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
    "early blight": "early blight",
    "alternariose": "early blight",
    "downy mildew": "downy mildew",
    "mildiou": "downy mildew",
    "gummosis": "gummosis",
    "gommose": "gummosis",
    "leaf spot": "leaf spot",
    "fruit and leaf spot": "fruit and leaf spot",
    "black spot": "black spot",
    "cercospora leaf spot": "cercospora leaf spot",
    "bacterial leaf spot": "bacterial leaf spot",
    "phomopsis leaf spot": "phomopsis leaf spot",
    "phomopsis cane and leaf spot": "phomopsis leaf spot",
    "peacock spot": "peacock spot",
    "oeil de paon": "peacock spot",
    "phytophthora blight": "phytophthora blight",
}

CROP_ALIASES = {
    "tomate": "tomato", "tomatoes": "tomato",
    "pomme de terre": "potato", "pommes de terre": "potato",
    "ble": "wheat", "blé": "wheat",
    "poivron": "pepper",
    "vigne": "grapevine", "grape vine": "grapevine",
    "figuier": "fig",
    "agrume": "citrus", "agrumes": "citrus",
    "olivier": "olive",
}

# Generic umbrella disease names that can correspond to SEVERAL distinct
# diseases in the corpus (different pathogens, different treatments).
# Key: the normalized generic term. Value: ordered list of the specific
# normalized disease names it can expand to. Extend this dict if other
# ambiguous umbrella terms are found later (e.g. a generic "blight" or
# "leaf spot" family, if the corpus grows to include one).
DISEASE_FAMILIES = {
    "rust": ["leaf rust", "stem rust", "stripe rust"],
}


def normalize_disease(value):
    v = normalize_text(value)
    return DISEASE_ALIASES.get(v, v)


def normalize_crop(value):
    v = normalize_text(value)
    return CROP_ALIASES.get(v, v)


def normalize_language(value):
    """
    Coerce any input into 'fr' or 'en'. Defaults to 'fr' for anything
    unrecognized, since that's the primary audience.
    """
    if value is None:
        return DEFAULT_LANGUAGE
    v = str(value).strip().lower()
    return "en" if v.startswith("en") else "fr"


# ============================================================
# RESULT TYPE
# ============================================================

@dataclass
class RetrievalResult:
    match_type: str                       # "exact" | "fuzzy" | "semantic" | "ambiguous" | "none"
    confidence: str                       # "high" | "medium" | "low"
    context_text: str
    chunk_ids: list = field(default_factory=list)
    pesticide_status: Optional[str] = None
    matched_crop: Optional[str] = None
    matched_disease: Optional[str] = None
    products: list = field(default_factory=list)
    # Precomputed, bilingual, ready-to-serve text -- {"fr": ..., "en": ...}
    gestion_maladie: dict = field(default_factory=dict)
    conseils_preventifs: dict = field(default_factory=dict)
    # Only populated when match_type == "ambiguous": the specific disease
    # names (normalized) the caller should choose between.
    ambiguous_options: list = field(default_factory=list)


# ============================================================
# RETRIEVAL ENGINE
# ============================================================

class RecommendationEngine:

    def __init__(self, chunks_file=CHUNKS_FILE, index_dir=INDEX_DIR,
                 embedding_model=EMBEDDING_MODEL, ollama_model=OLLAMA_MODEL,
                 lazy_faiss=True):
        self.chunks_by_key = {}      # (norm_crop, norm_disease) -> chunk dict
        self.all_chunks = []
        self._load_chunks(chunks_file)

        self._embeddings = None
        self._vectorstore = None
        self._embedding_model_name = embedding_model
        self._index_dir = index_dir
        if not lazy_faiss:
            self._ensure_faiss_loaded()

        # Still needed by api.py's /chat endpoint (extraction + free-text
        # reformulation) -- NOT used anywhere in this file's recommend().
        self.llm = ChatOllama(model=ollama_model, temperature=0.1, num_predict=1024)

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def _load_chunks(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                meta = record.get("metadata", {})
                key = (
                    normalize_crop(meta.get("crop", "")),
                    normalize_disease(meta.get("disease", "")),
                )
                self.chunks_by_key[key] = record
                self.all_chunks.append(record)
        self.known_crops = {k[0] for k in self.chunks_by_key}

    def _ensure_faiss_loaded(self):
        if self._vectorstore is not None:
            return
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore = FAISS.load_local(
            self._index_dir, self._embeddings, allow_dangerous_deserialization=True
        )

    @staticmethod
    def _bilingual_texts_from_metadata(meta, fr_key, en_key):
        return {
            "fr": meta.get(fr_key) or NOT_SPECIFIED_TEXT["fr"],
            "en": meta.get(en_key) or NOT_SPECIFIED_TEXT["en"],
        }

    def _result_from_record(self, record, match_type, confidence):
        meta = record["metadata"]
        return RetrievalResult(
            match_type=match_type,
            confidence=confidence,
            context_text=record["text"],
            chunk_ids=[record["id"]],
            pesticide_status=meta.get("pesticide_status"),
            matched_crop=meta.get("crop"),
            matched_disease=meta.get("disease"),
            products=meta.get("products", []),
            gestion_maladie=self._bilingual_texts_from_metadata(
                meta, "gestion_maladie_fr", "gestion_maladie_en"
            ),
            conseils_preventifs=self._bilingual_texts_from_metadata(
                meta, "conseils_preventifs_fr", "conseils_preventifs_en"
            ),
        )

    # --------------------------------------------------------
    # Retrieval: exact -> disease-family check -> fuzzy -> semantic
    # --------------------------------------------------------

    def retrieve(self, crop, disease):
        norm_crop = normalize_crop(crop)
        norm_disease = normalize_disease(disease)

        # 1. Exact match
        key = (norm_crop, norm_disease)
        if key in self.chunks_by_key:
            return self._result_from_record(self.chunks_by_key[key], "exact", "high")

        # 2. Disease-family disambiguation -- ONLY relevant when the
        #    requested (normalized) disease is a known umbrella term
        #    (e.g. "rust") that has NO chunk of its own, but DOES have
        #    multiple specific variants for this crop in the corpus.
        if norm_disease in DISEASE_FAMILIES:
            variants_present = [
                v for v in DISEASE_FAMILIES[norm_disease]
                if (norm_crop, v) in self.chunks_by_key
            ]
            if len(variants_present) == 1:
                # Only one specific variant exists for this crop -- no
                # real ambiguity, resolve directly as if it had been an
                # exact match on that variant.
                record = self.chunks_by_key[(norm_crop, variants_present[0])]
                return self._result_from_record(record, "exact", "high")
            if len(variants_present) > 1:
                # Genuinely ambiguous: refuse to guess. Caller must ask
                # the user (or upstream module) to specify which one.
                return RetrievalResult(
                    match_type="ambiguous",
                    confidence="low",
                    context_text="",
                    matched_crop=crop,
                    ambiguous_options=variants_present,
                )
            # variants_present is empty: no known variant for this crop
            # either -- fall through to fuzzy/semantic as usual.

        # 3. Fuzzy match, restricted to the same crop (avoid matching a
        #    similarly-named disease on a totally different crop).
        same_crop_diseases = {
            k[1]: k for k in self.chunks_by_key if k[0] == norm_crop
        }
        if same_crop_diseases:
            close = difflib.get_close_matches(
                norm_disease, same_crop_diseases.keys(),
                n=1, cutoff=FUZZY_MATCH_CUTOFF,
            )
            if close:
                matched_key = same_crop_diseases[close[0]]
                return self._result_from_record(
                    self.chunks_by_key[matched_key], "fuzzy", "medium"
                )

        # 4. Semantic fallback via FAISS -- no exact/fuzzy/family match
        #    found, either the crop or the disease (or both) are unknown
        #    to us.
        #
        #    IMPORTANT: if the CROP is recognized (even though the disease
        #    isn't), we restrict the search to chunks of that same crop.
        #    Without this, FAISS can pull in a same-family disease from a
        #    DIFFERENT crop and present it as if it were valid for the
        #    requested crop -- a real correctness risk for a pesticide
        #    recommendation tool, since a product's registration is
        #    crop-specific.
        self._ensure_faiss_loaded()
        query = f"Crop: {crop}. Disease: {disease}. Requested information: treatment, dosage."
        formatted_query = QUERY_PREFIX + query

        crop_restricted = norm_crop in self.known_crops
        results = []
        if crop_restricted:
            results = self._vectorstore.similarity_search(
                formatted_query, k=FAISS_FALLBACK_K,
                filter={"normalized_crop": norm_crop},
            )

        if not results:
            # Either the crop itself is unknown, or the crop-restricted
            # search returned nothing -- widen to the full index. Here we
            # deliberately fetch only the SINGLE best match (k=1), not
            # several: when the crop can't be pinned down, pulling in
            # multiple chunks risks pulling in DIFFERENT crops at once
            # and blending their content without making that mixing
            # obvious to the reader. A single chunk keeps the answer
            # internally consistent, even if it's a weaker match overall.
            results = self._vectorstore.similarity_search(formatted_query, k=1)
            crop_restricted = False

        if not results:
            return RetrievalResult(
                match_type="none", confidence="low", context_text="",
            )

        context_text = "\n\n---\n\n".join(doc.page_content for doc in results)
        top = results[0]

        # Tag each product with the crop/disease it actually came from.
        # In semantic fallback, results can span DIFFERENT crops or
        # DIFFERENT diseases of the same crop -- without this tag, a
        # consumer of the API has no way to tell that a product listed
        # under one recommendation was really registered for a different
        # crop/disease than the one asked about.
        products = []
        for doc in results:
            for p in doc.metadata.get("products", []):
                tagged = dict(p)
                tagged["source_crop"] = doc.metadata.get("crop")
                tagged["source_disease"] = doc.metadata.get("disease")
                products.append(tagged)

        # Same precomputed-text principle applies here: concatenate the
        # already-precomputed FR/EN texts of every returned chunk. Still
        # zero live LLM calls, even in the fallback path.
        gestion_fr = "\n\n".join(
            doc.metadata.get("gestion_maladie_fr", "") for doc in results
            if doc.metadata.get("gestion_maladie_fr")
        ) or NOT_SPECIFIED_TEXT["fr"]
        gestion_en = "\n\n".join(
            doc.metadata.get("gestion_maladie_en", "") for doc in results
            if doc.metadata.get("gestion_maladie_en")
        ) or NOT_SPECIFIED_TEXT["en"]
        preventifs_fr = "\n\n".join(
            doc.metadata.get("conseils_preventifs_fr", "") for doc in results
            if doc.metadata.get("conseils_preventifs_fr")
        ) or NOT_SPECIFIED_TEXT["fr"]
        preventifs_en = "\n\n".join(
            doc.metadata.get("conseils_preventifs_en", "") for doc in results
            if doc.metadata.get("conseils_preventifs_en")
        ) or NOT_SPECIFIED_TEXT["en"]

        return RetrievalResult(
            match_type="semantic",
            confidence="low" if not crop_restricted else "medium",
            context_text=context_text,
            chunk_ids=[doc.metadata.get("chunk_id", "") for doc in results],
            pesticide_status=top.metadata.get("pesticide_status"),
            matched_crop=top.metadata.get("crop"),
            matched_disease=top.metadata.get("disease"),
            products=products,
            gestion_maladie={"fr": gestion_fr, "en": gestion_en},
            conseils_preventifs={"fr": preventifs_fr, "en": preventifs_en},
        )

    # --------------------------------------------------------
    # Main entry point -- NO live LLM call. Pure lookup + language
    # selection from what retrieve() already resolved.
    # --------------------------------------------------------

    def recommend(self, crop, disease, language=DEFAULT_LANGUAGE):
        language = normalize_language(language)
        retrieval = self.retrieve(crop, disease)

        if retrieval.match_type == "none":
            no_info = {
                "fr": (
                    "Aucune information n'a été trouvée pour cette culture "
                    "et cette maladie dans la base documentaire actuelle."
                ),
                "en": (
                    "No information was found for this crop and disease in "
                    "the current knowledge base."
                ),
            }[language]
            return {
                "gestion_maladie": no_info,
                "conseils_preventifs": no_info,
                "match_type": "none",
                "confidence": "low",
                "chunk_ids": [],
                "pesticide_status": None,
                "matched_crop": None,
                "matched_disease": None,
                "products": [],
                "warning": None,
                "language": language,
                "ambiguous_options": [],
            }

        if retrieval.match_type == "ambiguous":
            options_list = ", ".join(retrieval.ambiguous_options)
            message = {
                "fr": (
                    f"Plusieurs maladies distinctes correspondent à '{disease}' "
                    f"pour la culture '{crop}' : {options_list}. Merci de "
                    f"préciser laquelle, chaque maladie ayant un traitement "
                    f"différent."
                ),
                "en": (
                    f"Several distinct diseases match '{disease}' for crop "
                    f"'{crop}': {options_list}. Please specify which one, "
                    f"as each disease has a different treatment."
                ),
            }[language]
            return {
                "gestion_maladie": message,
                "conseils_preventifs": message,
                "match_type": "ambiguous",
                "confidence": "low",
                "chunk_ids": [],
                "pesticide_status": None,
                "matched_crop": retrieval.matched_crop,
                "matched_disease": None,
                "products": [],
                "warning": None,
                "language": language,
                "ambiguous_options": retrieval.ambiguous_options,
            }

        warning = None
        if retrieval.match_type == "semantic":
            warning = {
                "fr": (
                    "Correspondance approximative : la maladie/culture "
                    "demandée ne correspond pas exactement à une entrée "
                    "connue de la base. La réponse ci-dessous est basée sur "
                    "la fiche la plus proche trouvée -- à vérifier avant "
                    "usage."
                ),
                "en": (
                    "Approximate match: the requested crop/disease does not "
                    "exactly match a known entry in the database. The "
                    "answer below is based on the closest fact sheet found "
                    "-- to be verified before use."
                ),
            }[language]

        return {
            "gestion_maladie": retrieval.gestion_maladie.get(language, NOT_SPECIFIED_TEXT[language]),
            "conseils_preventifs": retrieval.conseils_preventifs.get(language, NOT_SPECIFIED_TEXT[language]),
            "match_type": retrieval.match_type,
            "confidence": retrieval.confidence,
            "chunk_ids": retrieval.chunk_ids,
            "pesticide_status": retrieval.pesticide_status,
            "matched_crop": retrieval.matched_crop,
            "matched_disease": retrieval.matched_disease,
            "products": retrieval.products,
            "warning": warning,
            "language": language,
            "ambiguous_options": [],
        }


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    engine = RecommendationEngine()

    test_cases = [
        ("tomato", "Late Blight", "fr"),
        ("tomato", "Late Blight", "en"),
        ("wheat", "leaf rust", "fr"),
        ("ble", "rouille", "fr"),            # ambiguous -- 3 rust variants for wheat
        ("citrus", "rust", "fr"),            # "rust" family but no variant for citrus -> falls through
        ("potato", "gale commune", "fr"),    # unknown disease -> semantic fallback
    ]

    for crop, disease, lang in test_cases:
        print("\n" + "=" * 70)
        print(f"Crop: {crop} | Disease: {disease} | Language: {lang}")
        print("=" * 70)
        result = engine.recommend(crop, disease, language=lang)
        print(f"[match_type={result['match_type']} confidence={result['confidence']} "
              f"language={result['language']} chunks={result['chunk_ids']} "
              f"status={result['pesticide_status']}]")
        if result.get("ambiguous_options"):
            print(f"OPTIONS: {result['ambiguous_options']}")
        print()
        print("PRODUITS :")
        for p in result["products"]:
            print(" -", p)
        print()
        if result.get("warning"):
            print("⚠️", result["warning"])
        print("GESTION DE LA MALADIE :")
        print(result["gestion_maladie"])
        print()
        print("CONSEILS PRÉVENTIFS :")
        print(result["conseils_preventifs"])

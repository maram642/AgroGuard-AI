🌱 Plant Disease Diagnosis & Treatment Recommendation System

An AI-powered system for plant disease diagnosis support and context-aware treatment recommendations, developed as the Generative AI and integration component of a larger plant disease diagnosis workflow.

The system combines document extraction, OCR, table extraction, knowledge distillation, pesticide treatment matching, multilingual processing, vector search, RAG, and a FastAPI service to transform agricultural documentation into structured and searchable recommendations.

📌 Overview

The system is designed to support a complete plant disease diagnosis workflow:

Plant Image
     │
     ▼
Computer Vision
     │
     ├── Plant Identification
     └── Disease Detection
              │
              ▼
       RAG Recommendation System
              │
              ├── Disease Information
              ├── Management Practices
              ├── Preventive Advice
              └── Pesticide Treatments
              │
              ▼
        Final Recommendation

The project focuses on the Generative AI and integration layer. It consumes the crop and disease identified by the computer-vision component and retrieves relevant agricultural knowledge from a locally constructed knowledge base.

The knowledge base combines:

Disease descriptions
Symptoms and affected plant parts
Pathogen information
Disease spread and cycle
General disease management
Preventive advice
Pesticide treatments
Active substances
Dosages
Homologation information

Disease management and pesticide treatments are processed as separate types of information and combined only when constructing the final recommendation.

🏗️ System Architecture

The complete processing pipeline is:

Agricultural Documents
        │
        ▼
Document Extraction
        │
        ├── Text Extraction
        ├── Selective OCR
        └── Table Extraction
        │
        ▼
extracted/
        │
        ├──────────────────────┐
        ▼                      ▼
Disease Distillation     Pesticide Matching
        │                      │
        ▼                      ▼
distilled_mistral/       pesticide_treatment_matches.xlsx
                               │
                               ▼
                     Manual Review / Correction
                               │
                               ▼
                pesticide_treatment_matches_final.xlsx
        │                      │
        └──────────┬───────────┘
                   ▼
             build_chunks.py
                   │
                   ▼
              chunks.jsonl
                   │
                   ▼
        precompute_translations.py
                   │
                   ▼
             build_index.py
                   │
                   ▼
              FAISS Index
                   │
                   ▼
             RAG Engine
                   │
                   ▼
             FastAPI API
                   │
                   ▼
            Recommendations
📄 1. Document Extraction & Preprocessing

The original agricultural documents are processed through a dedicated extraction pipeline located in extractors/.

The extraction stage is designed to handle both text-based and scanned agricultural documents.

Text extraction

text_extractor.py uses PyMuPDF to extract text from PDF documents page by page.

Each page is stored with:

Page number
Extracted text
Character count

The character count is later used to determine whether OCR is required.

Selective OCR

ocr_extractor.py uses:

Tesseract
pytesseract
pdf2image
Poppler

OCR is applied only to pages where the extracted text contains fewer than 100 characters.

This avoids unnecessarily OCRing pages that already contain usable text.

OCR language is selected according to document type:

Disease documents   → English
Pesticide documents → French + Arabic
Table extraction

table_extractor.py extracts agricultural tables using:

pdfplumber
Camelot

Camelot is used when the extracted table reaches a parsing accuracy of at least 80%.

Otherwise, pdfplumber is used as the fallback.

Additional filtering removes fragments that do not represent meaningful tables:

Minimum rows:    3
Minimum columns: 2

Merged-cell values are also forward-filled when necessary.

Page-level integration

pipeline.py combines:

extracted text
OCR results
extracted tables

into a structured JSON representation for every source document.

Each page records information such as:

{
  "page": 0,
  "text": "...",
  "used_ocr": false,
  "tables": []
}

The resulting files are stored in:

extracted/
OCR quality auditing

audit_ocr_failures.py checks OCR pages whose resulting text contains fewer than 20 characters.

These pages are reported as potential OCR failures so that problematic documents can be investigated without manually checking every extracted page.

🧠 2. Disease Knowledge Distillation

distill.py converts extracted disease documentation into structured disease knowledge.

The process uses Mistral through Ollama to extract relevant information from the source documents.

For each disease, the system extracts:

Pathogen
Symptoms
Affected parts
Spread
Disease cycle
General management
Preventive advice

Chemical products, brands, dosages, and application rates are intentionally excluded from the general disease-management fields.

This keeps:

Disease knowledge

separate from:

Pesticide treatment information

Long documents are processed in chunks, and the resulting structured information is merged and deduplicated.

The distilled disease information is stored in:

distilled_mistral/

A review workbook is also generated to facilitate validation of the extracted information.

Existing distilled files are not overwritten, allowing manually corrected information to be preserved.

🧪 3. Pesticide Treatment Matching

match_pesticide_treatments.py processes pesticide catalog information and identifies pesticide treatments associated with known crop–disease pairs.

The process includes:

Mapping crop and disease names to the terminology used in the pesticide catalog.
Searching the catalog for pesticide records associated with each crop–disease pair.
Extracting structured treatment information.
Assigning a matching status.
Reviewing and correcting the extracted treatment records.
Retaining multiple applicable pesticide products for a crop–disease pair when available.

The initial output is:

pesticide_treatment_matches.xlsx

The treatment records are then manually reviewed and corrected to produce the validated file:

pesticide_treatment_matches_final.xlsx

The final treatment dataset contains multiple pesticide records for some crop–disease pairs rather than a single treatment per disease.

Treatment records can contain:

Product name
Active substance
Dosage
Homologation number
Usage information
Matching status
Confidence information

This validated treatment dataset is subsequently used by build_chunks.py.

🧩 4. Knowledge Base Construction

build_chunks.py combines the distilled disease information with the validated pesticide treatment information.

The system normalizes crop and disease names before grouping information by:

(crop, disease)

Each knowledge-base record contains structured disease information such as:

Crop
Disease
Pathogen
Symptoms
Affected parts
Spread
Disease cycle
Management
Preventive advice

When pesticide information is available, treatment records are attached separately as structured product information.

A treatment can contain:

Product
Active substance
Dosage
Homologation
Usage
Confidence

The system also preserves the original management and preventive information through fields such as:

management_raw
preventive_advice_raw

The resulting knowledge base is stored as:

chunks.jsonl

Each chunk corresponds to a normalized crop–disease knowledge group.

🌍 5. Multilingual Translation & Reformulation

precompute_translations.py prepares management and preventive information for multilingual retrieval and recommendation generation.

Mistral is used to generate controlled English and French reformulations of:

Disease management
Preventive advice

The resulting fields include:

gestion_maladie_en
gestion_maladie_fr
conseils_preventifs_en
conseils_preventifs_fr

The generated English reformulations are validated using keyword-overlap checks.

Potential failures are recorded for review in:

flagged_for_review.json

The processed information is then written back to:

chunks.jsonl
🔎 6. Vector Index Construction

build_index.py converts the knowledge base into a searchable vector index.

The embedding model used is:

intfloat/multilingual-e5-base

The embedding text includes structured information such as:

Crop
Disease
Crop-disease pair
Pathogen

followed by the corresponding knowledge content.

This explicit inclusion of the crop–disease pair helps the retrieval system distinguish between diseases that may have similar terminology across different crops.

The embeddings are normalized and stored using FAISS.

The resulting index is:

faiss_index/
├── index.faiss
└── index.pkl
🤖 7. Retrieval-Augmented Generation

The recommendation engine uses the constructed knowledge base to retrieve information relevant to a requested crop and disease.

The general flow is:

User Request
     │
     ▼
Crop + Disease Matching
     │
     ▼
Relevant Knowledge Retrieval
     │
     ▼
FAISS Vector Search
     │
     ▼
Disease + Treatment Information
     │
     ▼
Recommendation Response

The API also provides matching information so that the user can see how the requested disease was resolved.

For example, a request containing:

powedry mildew

can be matched to:

Powdery Mildew

with information about the matching method and confidence.

The system can therefore distinguish between:

Exact or direct matches
Fuzzy matches
Approximate matches
Cases requiring clarification
🚀 8. FastAPI Recommendation Service

The final recommendation system is exposed through a FastAPI service.

The main endpoint is:

POST /recommend

It accepts:

crop
disease
language
Example request
curl -X 'POST' \
  'http://127.0.0.1:8000/recommend' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "crop": "tomato",
  "disease": "powedry mildew",
  "language": "fr"
}'
Example response
{
  "crop": "tomato",
  "disease": "powedry mildew",
  "language": "fr",
  "traitement_recommande": [
    {
      "produit": "MELCAR TOP 140 DC",
      "substance_active": "Difenoconazole + Cyflufenamid",
      "dosage": "0.75 L/ha",
      "homologation": "F.027-18",
      "utilisation": null,
      "confiance": "haute",
      "source_crop": null,
      "source_disease": null
    }
  ],
  "gestion_maladie": "...",
  "conseils_preventifs": "...",
  "match_type": "fuzzy",
  "confidence": "medium",
  "is_approximate_match": false,
  "avertissement": null,
  "clarification_needed": false,
  "disease_options": [],
  "pesticide_status": "FOUND",
  "matched_crop": "tomate",
  "matched_disease": "Powdery Mildew",
  "source_chunk_ids": [
    "tomato::powdery_mildew"
  ]
}

The response combines:

Treatment information
Product
Active substance
Dosage
Homologation
Usage
Treatment confidence
Disease information
General disease management
Preventive advice
Matching information
Match type
Matching confidence
Approximate-match indicator
Matched crop
Matched disease
Source chunk IDs
User guidance
Warnings
Clarification requirements
Possible disease options

The API therefore provides both the recommendation and information about how the requested crop/disease was matched to the knowledge base.

📚 9. API Documentation

When the FastAPI server is running, the interactive Swagger documentation is available at:

http://127.0.0.1:8000/docs

The service can be started with:

python api.py

or:

uvicorn api:app --host 127.0.0.1 --port 8000

The API exposes the recommendation service through:

POST /recommend
🧪 10. Testing

The repository contains dedicated scripts for testing the retrieval and recommendation components:

test_retrieval.py
test_rag_batch.py

These can be used to evaluate:

Knowledge retrieval
Crop/disease matching
Recommendation generation
RAG behavior across multiple inputs
📁 11. Project Structure
plant-internship-rag/
│
├── api.py
├── rag_chain.py
│
├── build_chunks.py
├── build_index.py
├── distill.py
├── match_pesticide_treatments.py
├── precompute_translations.py
│
├── extractors/
│   ├── pipeline.py
│   ├── text_extractor.py
│   ├── ocr_extractor.py
│   ├── table_extractor.py
│   └── audit_ocr_failures.py
│
├── chunks.jsonl
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── test_retrieval.py
├── test_rag_batch.py
│
├── requirements.txt
└── .gitignore
⚙️ 12. Running the Pipeline

The complete processing workflow is:

1. Extract documents
python extractors/pipeline.py
2. Audit OCR results
python extractors/audit_ocr_failures.py
3. Distill disease knowledge
python distill.py
4. Match pesticide treatments
python match_pesticide_treatments.py

Review and correct the generated treatment file:

pesticide_treatment_matches.xlsx

The validated version should be saved as:

pesticide_treatment_matches_final.xlsx
5. Build the knowledge base
python build_chunks.py
6. Precompute translations
python precompute_translations.py
7. Build the vector index
python build_index.py
8. Start the API
python api.py

The API will then be available at:

http://127.0.0.1:8000

and the interactive documentation at:

http://127.0.0.1:8000/docs
🛠️ 13. Technologies
Programming
Python
Pandas
JSON / JSONL
Excel processing
Document Processing
PyMuPDF
pdfplumber
Camelot
Tesseract
pytesseract
pdf2image
Poppler
AI & NLP
Mistral
Ollama
Hugging Face
Sentence Transformers
LangChain
Retrieval-Augmented Generation
Vector Search
FAISS
intfloat/multilingual-e5-base
Backend
FastAPI
Uvicorn
Pydantic
Development
Git
GitHub
VS Code
🎯 14. Design Principles

The system was designed around several principles:

Structured knowledge before retrieval

Raw agricultural documents are first transformed into structured disease and treatment information before being indexed.

Separation of disease and pesticide knowledge

General disease management and preventive advice are kept separate from chemical treatment information.

Selective OCR

OCR is applied only when normal PDF text extraction is insufficient.

Table-aware extraction

Agricultural pesticide catalogs often contain structured tables, so table extraction is handled separately from normal text extraction.

Human validation

Extracted pesticide treatment information is reviewed and corrected before being used to construct the final knowledge base.

Multilingual processing

The system supports agricultural information in multiple languages and prepares management and preventive information in English and French.

Traceable retrieval

The API returns source chunk identifiers and matching information, making it possible to understand which knowledge-base entry was used.

Local AI components

The main language-model processing is performed locally through Ollama, while the vector index is stored locally using FAISS.

🌾 15. Project Context

This project was developed as part of an AI Engineering internship focused on plant identification and disease diagnosis in the Tunisian agricultural context.

The Generative AI component is designed to complement a computer-vision pipeline:

Image
  │
  ▼
Plant Identification
  │
  ▼
Disease Detection
  │
  ▼
Crop + Disease
  │
  ▼
RAG Recommendation Engine
  │
  ├── Disease Management
  ├── Preventive Advice
  └── Pesticide Treatments

The system therefore connects computer vision, agricultural knowledge extraction, NLP, vector retrieval, and API integration into a single recommendation workflow.

👩‍💻 Author

Maram Boughammoura

Data Engineering & AI Engineering Student
ENET'Com — Tunisia

GitHub: @maram642


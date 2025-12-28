# Enterprise Policy & Compliance Assistant (RAG)

A **production-oriented foundation for an enterprise Retrieval-Augmented Generation (RAG) system**, focused on **robust document ingestion, preprocessing, and chunking of policy and compliance documents**.

This project addresses the **most critical and error-prone part of RAG systems**: converting unstructured enterprise PDFs into **clean, structured, metadata-rich text units** that are reliable for downstream semantic retrieval and AI-based reasoning.

---

##  Project Objective

Enterprise policies, HR handbooks, and compliance documents are typically stored as long, unstructured PDFs.
These documents are difficult to search, audit, or reason over programmatically.

This system lays the **core data pipeline** required to:

* Parse complex enterprise PDFs
* Normalize noisy text
* Split content into semantically meaningful chunks
* Preserve metadata for traceability and citations
* Prepare documents for vector-based retrieval and LLM grounding

---

## ✅ Implemented Capabilities

### 📄 PDF Document Ingestion

* Supports ingestion of enterprise PDFs (HR, policy, compliance)
* Handles multi-page documents
* Robust text extraction resilient to:

  * Headers and footers
  * Page breaks
  * Formatting noise

### 🧹 Text Cleaning & Normalization

* Removes non-informative artifacts
* Normalizes whitespace and line breaks
* Produces clean, retrieval-ready text

### ✂️ Intelligent Chunking (RAG-Optimized)

* Recursive chunking strategy
* Configurable:

  * Chunk size
  * Overlap
* Ensures semantic coherence across chunks

### 🏷️ Metadata Preservation

Each chunk is enriched with:

* Source document name
* Page number(s)
* Chunk index

This metadata is essential for:

* Future citation generation
* Compliance audits
* Explainable AI outputs

---

## 🏗️ Current System Architecture

```
PDF Documents
      │
      ▼
PDF Loader
      │
      ▼
Text Cleaning & Normalization
      │
      ▼
Recursive Chunking Engine
      │
      ▼
Structured Text Chunks + Metadata
```

The output of this pipeline is **vector-store ready** and designed for seamless integration with embedding models and retrieval engines.

---

## 📁 Repository Structure

```text
enterprise-policy-rag/
│
├── app/
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PDF parsing & text extraction
│   │   ├── chunker.py         # Recursive chunking logic
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── config.py          # Centralized configuration
│   │   ├── logger.py          # Logging setup
│   │   └── constants.py
│   │
│   └── main.py
│
├── data/
│   ├── raw_pdfs/              # Original enterprise PDFs
│   └── processed/             # Cleaned & chunked outputs
│
├── tests/
│   └── test_ingestion.py      # Unit tests for ingestion pipeline
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer         | Technology                   |
| ------------- | ---------------------------- |
| Language      | Python 3.10                  |
| PDF Parsing   | PyMuPDF / pdfplumber         |
| Chunking      | Recursive Character Splitter |
| Configuration | Pydantic                     |
| Logging       | Python logging               |
| Testing       | Pytest                       |

---

## 🧠 Design Rationale

### Why Focus Heavily on Ingestion?

In production RAG systems:

* **70–80% of failures originate from bad document preprocessing**
* Poor chunking leads to hallucinations and irrelevant answers
* Metadata loss breaks citation and compliance guarantees

This project prioritizes **data quality and traceability** over premature model integration.

### Why Recursive Chunking?

* Preserves semantic meaning
* Avoids sentence truncation
* Produces retrieval-friendly chunk boundaries

---

## 🧪 Quality Assurance

* Unit tests validating:

  * PDF parsing correctness
  * Chunk size constraints
  * Metadata consistency
* Manual inspection of chunk distributions
* Deterministic preprocessing for reproducibility


## 🔜 Planned Extensions

* Sentence Transformer embeddings
* Vector database integration (FAISS / ChromaDB)
* Semantic retrieval
* LLM-based answer generation with citations
* API and web-based interface
* Cloud deployment

---

##  License

MIT License

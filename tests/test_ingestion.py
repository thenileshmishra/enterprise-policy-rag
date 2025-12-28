import os
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents


def test_pdf_load():
    sample_pdf = "data/raw_pdfs/sample_policy.pdf"
    assert os.path.exists(sample_pdf), "Sample PDF missing"

    pages = load_pdf(sample_pdf)

    assert len(pages) > 0
    assert "text" in pages[0]
    assert "page_number" in pages[0]


def test_chunking():
    sample_pdf = "data/raw_pdfs/sample_policy.pdf"
    pages = load_pdf(sample_pdf)

    chunks = chunk_documents(pages)

    assert len(chunks) > 0
    assert len(chunks[0]["text"]) > 50
    assert "chunk_id" in chunks[0]
    assert "source" in chunks[0]

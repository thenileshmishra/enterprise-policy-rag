"""Minimal PDF loading helpers using PyMuPDF."""

from pathlib import Path
from typing import List, Dict
import fitz
from app.core.logger import logger


def load_pdf(file_path: str) -> List[Dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    pages: List[Dict] = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = page.get_text("text").strip()
        pages.append(
            {
                "page_number": page_index + 1,
                "text": text,
                "source": path.name,
            }
        )

    doc.close()
    logger.info(f"Loaded {len(pages)} pages from {path.name}")
    return pages


def get_pdf_metadata(file_path: str) -> Dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    metadata = doc.metadata
    page_count = len(doc)
    doc.close()

    return {
        "filename": path.name,
        "title": metadata.get("title") or path.stem,
        "page_count": page_count,
        "author": metadata.get("author") or "Unknown",
    }

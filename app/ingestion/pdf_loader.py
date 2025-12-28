import fitz  # PyMuPDF
from typing import List, Dict
from pathlib import Path


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load PDF and extract page-wise clean text.
    Returns list of dicts:
    [
        {
            "page_number": int,
            "text": str,
            "source": filename
        }
    ]
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    pages = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = page.get_text("text")

        clean_text = (
            text.replace("\n", " ")
            .replace("\t", " ")
            .strip()
        )

        pages.append(
            {
                "page_number": page_index + 1,
                "text": clean_text,
                "source": path.name
            }
        )

    doc.close()
    return pages

"""
PDF loading and text extraction with enhanced metadata.
Supports both PyMuPDF (fast) and Unstructured (better structure).
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from app.core.logger import logger


def detect_heading(text: str) -> Optional[str]:
    """
    Detect if a line is likely a section heading.
    Returns the heading text if detected, None otherwise.
    """
    text = text.strip()

    if not text or len(text) > 150:
        return None

    # Pattern 1: Numbered sections (1., 1.1, I., A., etc.)
    numbered_pattern = r"^(\d+\.[\d.]*|\[?\d+\]|[IVX]+\.|[A-Z]\.)\s+[A-Z]"
    if re.match(numbered_pattern, text):
        return text

    # Pattern 2: ALL CAPS headings
    if text.isupper() and len(text) > 3 and len(text) < 80:
        return text

    # Pattern 3: Title Case with common heading words
    heading_keywords = ["Abstract", "Introduction", "Methods", "Results",
                       "Discussion", "Conclusion", "References", "Acknowledgments",
                       "Background", "Overview", "Summary", "Methodology",
                       "Related Work", "Experimental", "Analysis", "Findings"]
    for keyword in heading_keywords:
        if text.startswith(keyword) and len(text) < 100:
            return text

    # Pattern 4: Ends with colon (common for subsections)
    if text.endswith(":") and len(text) < 80:
        return text.rstrip(":")

    return None


def extract_text_with_structure(page: fitz.Page) -> Tuple[str, List[Dict]]:
    """
    Extract text from a page while preserving structural information.
    Returns (full_text, list_of_text_blocks_with_metadata).
    """
    blocks = page.get_text("dict")["blocks"]
    text_blocks = []
    full_text_parts = []

    for block in blocks:
        if block["type"] == 0:  # Text block
            block_text = ""
            block_font_size = 0
            is_bold = False

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    block_text += span_text + " "

                    # Track font properties for heading detection
                    font_size = span.get("size", 12)
                    font_flags = span.get("flags", 0)

                    if font_size > block_font_size:
                        block_font_size = font_size

                    if font_flags & 2**4:  # Bold flag
                        is_bold = True

                block_text += "\n"

            block_text = block_text.strip()
            if block_text:
                text_blocks.append({
                    "text": block_text,
                    "font_size": block_font_size,
                    "is_bold": is_bold,
                    "bbox": block["bbox"],
                })
                full_text_parts.append(block_text)

    return "\n\n".join(full_text_parts), text_blocks


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load PDF and extract page-wise clean text with enhanced metadata.

    Returns:
        List of dicts with page_number, text, source, and detected sections.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    pages = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = page.get_text("text")

        # Clean text while preserving structure
        clean_text = re.sub(r'\s+', ' ', text).strip()

        # Detect headings in the page
        headings_found = []
        lines = text.split('\n')
        for line in lines:
            heading = detect_heading(line)
            if heading:
                headings_found.append(heading)

        pages.append({
            "page_number": page_index + 1,
            "text": clean_text,
            "source": path.name,
            "headings": headings_found,
            "current_section": headings_found[0] if headings_found else None,
        })

    doc.close()
    logger.info(f"Loaded {len(pages)} pages from {path.name}")
    return pages


def load_pdf_with_structure(file_path: str) -> List[Dict]:
    """
    Load PDF with detailed structural information.
    Better for research papers with complex formatting.

    Returns:
        List of dicts with enhanced metadata including font info.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    pages = []
    current_section = None

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        full_text, text_blocks = extract_text_with_structure(page)

        # Detect sections from text blocks
        headings = []
        for block in text_blocks:
            # Large or bold text is likely a heading
            if block["font_size"] > 12 or block["is_bold"]:
                heading = detect_heading(block["text"])
                if heading:
                    headings.append(heading)
                    current_section = heading

        pages.append({
            "page_number": page_index + 1,
            "text": full_text,
            "source": path.name,
            "headings": headings,
            "current_section": current_section,
            "text_blocks": text_blocks,
        })

    doc.close()
    logger.info(f"Loaded {len(pages)} pages with structure from {path.name}")
    return pages


def get_pdf_metadata(file_path: str) -> Dict:
    """
    Extract PDF document metadata (title, author, creation date, etc.).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    metadata = doc.metadata

    result = {
        "filename": path.name,
        "title": metadata.get("title", path.stem),
        "author": metadata.get("author", "Unknown"),
        "subject": metadata.get("subject", ""),
        "keywords": metadata.get("keywords", ""),
        "creator": metadata.get("creator", ""),
        "producer": metadata.get("producer", ""),
        "creation_date": metadata.get("creationDate", ""),
        "modification_date": metadata.get("modDate", ""),
        "page_count": len(doc),
        "file_size_bytes": path.stat().st_size,
    }

    doc.close()
    return result


def extract_toc(file_path: str) -> List[Dict]:
    """
    Extract table of contents from PDF if available.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    toc = doc.get_toc()

    toc_entries = []
    for level, title, page in toc:
        toc_entries.append({
            "level": level,
            "title": title,
            "page": page,
        })

    doc.close()
    return toc_entries

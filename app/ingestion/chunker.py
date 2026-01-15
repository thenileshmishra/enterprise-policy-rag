"""
Document chunking using Unstructured library for enhanced metadata extraction.
Preserves document structure including sections, headings, and page numbers.
"""

from typing import List, Dict, Optional
import uuid
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    ListItem,
    Table,
    Element,
)
from app.core.logger import logger


def extract_elements_from_pdf(file_path: str) -> List[Element]:
    """
    Extract structured elements from PDF using Unstructured.
    Returns elements with type, text, and metadata.
    """
    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="fast",  # Use "hi_res" for better accuracy with OCR
            include_page_breaks=True,
            infer_table_structure=True,
        )
        logger.info(f"Extracted {len(elements)} elements from {file_path}")
        return elements
    except Exception as e:
        logger.error(f"Failed to extract elements from PDF: {e}")
        raise


def get_element_metadata(element: Element, source: str) -> Dict:
    """Extract metadata from an Unstructured element."""
    metadata = element.metadata

    return {
        "page_number": getattr(metadata, "page_number", 1),
        "section": getattr(metadata, "section", None),
        "parent_id": getattr(metadata, "parent_id", None),
        "category": element.category if hasattr(element, "category") else type(element).__name__,
        "source": source,
    }


def chunk_with_unstructured(
    file_path: str,
    source_name: str,
    max_characters: int = 800,
    overlap: int = 150,
    combine_text_under_n_chars: int = 200,
) -> List[Dict]:
    """
    Chunk PDF using Unstructured's title-based chunking.
    Preserves section/heading context in metadata.

    Args:
        file_path: Path to PDF file
        source_name: Name of the source document
        max_characters: Maximum characters per chunk
        overlap: Character overlap between chunks
        combine_text_under_n_chars: Combine small text blocks

    Returns:
        List of chunk dictionaries with enhanced metadata
    """
    # Extract elements from PDF
    elements = extract_elements_from_pdf(file_path)

    if not elements:
        logger.warning(f"No elements extracted from {file_path}")
        return []

    # Use Unstructured's title-based chunking
    chunked_elements = chunk_by_title(
        elements,
        max_characters=max_characters,
        overlap=overlap,
        combine_text_under_n_chars=combine_text_under_n_chars,
    )

    chunks = []
    current_section = None
    current_heading = None

    for element in chunked_elements:
        # Track section/heading context
        if isinstance(element, Title):
            current_heading = element.text
            if len(element.text) > 50:  # Likely a section title
                current_section = element.text[:50]

        # Get element metadata
        meta = get_element_metadata(element, source_name)

        # Build chunk with enhanced metadata
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "text": element.text.strip() if element.text else "",
            "source": source_name,
            "page_number": meta["page_number"],
            "section": current_section or meta.get("section"),
            "heading": current_heading,
            "element_type": meta["category"],
        }

        # Only add non-empty chunks
        if chunk["text"]:
            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from {source_name}")
    return chunks


def chunk_documents(
    pages: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Legacy chunking function for backwards compatibility.
    Converts page-based input to chunks with basic metadata.

    For enhanced chunking with Unstructured, use chunk_with_unstructured() directly.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    current_section = None

    for page in pages:
        text = page["text"]

        # Try to detect section headings (lines in ALL CAPS or ending with colon)
        lines = text.split(". ")
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.isupper() and len(line_stripped) < 100:
                current_section = line_stripped
                break
            elif line_stripped.endswith(":") and len(line_stripped) < 80:
                current_section = line_stripped.rstrip(":")
                break

        splits = splitter.split_text(text)

        for split_text in splits:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": split_text.strip(),
                "source": page["source"],
                "page_number": page["page_number"],
                "section": current_section,
                "heading": None,
                "element_type": "text",
            })

    return chunks


def chunk_documents_enhanced(
    file_path: str,
    source_name: str,
    use_unstructured: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Main chunking entry point. Uses Unstructured by default.

    Args:
        file_path: Path to the PDF file
        source_name: Name of the source document
        use_unstructured: If True, use Unstructured library (recommended)
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks

    Returns:
        List of chunk dictionaries with metadata
    """
    if use_unstructured:
        return chunk_with_unstructured(
            file_path=file_path,
            source_name=source_name,
            max_characters=chunk_size,
            overlap=chunk_overlap,
        )
    else:
        # Fallback to legacy PyMuPDF + LangChain approach
        from app.ingestion.pdf_loader import load_pdf
        pages = load_pdf(file_path)
        return chunk_documents(pages, chunk_size, chunk_overlap)

"""Simple document chunking utilities for beginner-friendly RAG."""

from typing import List, Dict
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    pages: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """Split page texts into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        text = page["text"]

        splits = splitter.split_text(text)

        for split_text in splits:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": split_text.strip(),
                "source": page["source"],
                "page_number": page["page_number"],
            })

    return chunks

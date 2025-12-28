from typing import List , Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def chunk_documents(
    pages: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Convert page text into semantic chunks with metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])

        for chunk in splits:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk.strip(),
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "policy_section": None  # can auto-detect later
                }
            )

    return chunks

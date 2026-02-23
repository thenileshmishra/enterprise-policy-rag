"""Simple upload API: PDF -> chunk -> embed -> FAISS index."""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.core.logger import logger
from app.core.exception import CustomException
from app.ingestion.pdf_loader import load_pdf, get_pdf_metadata
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import get_embeddings
from app.retrieval.vector_store import VectorStoreFAISS

router = APIRouter()

# Single global vector store for this beginner-friendly app
vector_db = VectorStoreFAISS()


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str
    chunks_indexed: int
    document_name: str
    metadata: dict


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
):
    """
    Upload one PDF and rebuild the global index with its chunks.
    """
    try:
        logger.info(f"Received file upload: {file.filename}")

        file_bytes = await file.read()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        try:
            pdf_metadata = get_pdf_metadata(tmp_path)

            pages = load_pdf(tmp_path)
            for page in pages:
                page["source"] = file.filename

            chunks = chunk_documents(pages)

        finally:
            os.remove(tmp_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from PDF")

        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings(texts)

        metadata = [
            {
                "source": c.get("source", file.filename),
                "page_number": c.get("page_number"),
                "chunk_id": c.get("chunk_id"),
                "text": c["text"],
            }
            for c in chunks
        ]

        vector_db.build(embeddings, metadata)
        logger.info(f"Indexed {len(chunks)} chunks from {file.filename}")

        return UploadResponse(
            message="PDF processed and indexed",
            chunks_indexed=len(chunks),
            document_name=file.filename,
            metadata={
                "title": pdf_metadata.get("title"),
                "pages": pdf_metadata.get("page_count"),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise CustomException(e)

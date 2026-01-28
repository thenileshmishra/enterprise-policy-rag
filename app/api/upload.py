"""
Upload API endpoint with session support and S3 storage.
Supports multi-PDF uploads with enhanced metadata extraction.
"""

import os
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from app.core.logger import logger
from app.core.exception import CustomException
from app.ingestion.pdf_loader import load_pdf, get_pdf_metadata
from app.ingestion.chunker import chunk_documents_enhanced, chunk_documents
from app.ingestion.embedder import Embedder, get_embeddings
from app.retrieval.vector_store import VectorStoreFAISS
from app.retrieval.session_store import (
    get_session_store,
    add_to_session,
    session_manager,
)
from app.retrieval.sparse_retriever import get_session_bm25
from app.retrieval.storage import upload_pdf_from_bytes

router = APIRouter()

# Global vector store for non-session mode
vector_db = VectorStoreFAISS()
embedder = Embedder()


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str
    chunks_indexed: int
    session_id: Optional[str] = None
    document_name: str
    s3_key: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(None, description="Session ID for multi-document sessions"),
    use_unstructured: bool = Query(True, description="Use Unstructured library for better parsing"),
    upload_to_s3: bool = Query(False, description="Upload PDF to S3 storage"),
):
    """
    Upload and index a PDF document.

    - Supports session-based isolation for multi-document chats
    - Uses Unstructured library for enhanced text extraction
    - Optionally stores PDF in S3
    - Extracts rich metadata including sections and headings
    """
    try:
        logger.info(f"Received file upload: {file.filename}")

        # Read uploaded file bytes
        file_bytes = await file.read()

        # Optional S3 upload
        s3_key = None
        if upload_to_s3:
            s3_bucket = os.getenv("S3_PDF_BUCKET", os.getenv("S3_BUCKET"))
            if s3_bucket:
                success, s3_key = upload_pdf_from_bytes(
                    file_bytes=file_bytes,
                    filename=file.filename,
                    bucket=s3_bucket,
                    session_id=session_id,
                )
                if success:
                    logger.info(f"PDF uploaded to S3: {s3_key}")

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        try:
            # Extract PDF metadata
            pdf_metadata = get_pdf_metadata(tmp_path)

            # Process document with appropriate chunker
            if use_unstructured:
                try:
                    chunks = chunk_documents_enhanced(
                        file_path=tmp_path,
                        source_name=file.filename,
                        use_unstructured=True,
                        chunk_size=800,
                        chunk_overlap=150,
                    )
                except Exception as e:
                    logger.warning(f"Unstructured failed, falling back to PyMuPDF: {e}")
                    pages = load_pdf(tmp_path)
                    # Update source name
                    for page in pages:
                        page["source"] = file.filename
                    chunks = chunk_documents(pages)
            else:
                pages = load_pdf(tmp_path)
                for page in pages:
                    page["source"] = file.filename
                chunks = chunk_documents(pages)

        finally:
            # Clean up temp file
            os.remove(tmp_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from PDF")

        # Generate embeddings
        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings(texts)

        # Index based on session mode
        if session_id:
            # Session-based indexing
            session_store = get_session_store(session_id)
            session_store.add_chunks(chunks, embeddings)
            session_store.add_document(file.filename)

            # Also add to BM25 index for hybrid search
            bm25_store = get_session_bm25(session_id)
            bm25_store.add_chunks(chunks)

            logger.info(f"Session {session_id}: Indexed {len(chunks)} chunks from {file.filename}")

            return UploadResponse(
                message="PDF processed and indexed in session",
                chunks_indexed=len(chunks),
                session_id=session_id,
                document_name=file.filename,
                s3_key=s3_key,
                metadata={
                    "title": pdf_metadata.get("title"),
                    "pages": pdf_metadata.get("page_count"),
                    "total_session_docs": len(session_store.documents),
                },
            )
        else:
            # Global index mode (backward compatible)
            metadata = [
                {
                    "source": c.get("source", file.filename),
                    "page_number": c.get("page_number"),
                    "chunk_id": c.get("chunk_id"),
                    "section": c.get("section"),
                    "heading": c.get("heading"),
                    "text": c["text"],
                }
                for c in chunks
            ]

            vector_db.build(embeddings, metadata)
            logger.info(f"Global index: Indexed {len(chunks)} chunks")

            return UploadResponse(
                message="PDF processed and indexed",
                chunks_indexed=len(chunks),
                document_name=file.filename,
                s3_key=s3_key,
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


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return {"sessions": sessions}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data."""
    deleted = session_manager.delete_session(session_id)
    if deleted:
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

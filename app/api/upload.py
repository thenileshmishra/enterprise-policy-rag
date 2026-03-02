"""Upload API: PDF -> chunk -> embed -> index (supports multi-document sessions)."""

import os
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.core.logger import logger
from app.core.exception import CustomException
from app.ingestion.pdf_loader import load_pdf, get_pdf_metadata
from app.ingestion.chunker import chunk_documents
from app.retrieval.session_store import get_or_create_session

router = APIRouter()


class UploadResponse(BaseModel):
    message: str
    session_id: str
    chunks_indexed: int
    document_name: str
    total_documents: int
    metadata: dict


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Upload a PDF and add it to a session index.
    Pass session_id to append to an existing session, or omit for a new session.
    """
    try:
        session = get_or_create_session(session_id)
        logger.info(f"Upload to session {session.session_id}: {file.filename}")

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

        session.add_document(chunks, file.filename)

        return UploadResponse(
            message="PDF processed and indexed",
            session_id=session.session_id,
            chunks_indexed=len(chunks),
            document_name=file.filename,
            total_documents=len(session.documents),
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

from fastapi import APIRouter
from app.core.logger import logger
from app.core.exception import CustomException
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStoreFAISS

router = APIRouter()
vector_db = VectorStoreFAISS()
embedder = Embedder()


@router.post("/upload")
async def upload_pdf(file_bytes: bytes):
    try:
        logger.info("Received file upload")

        # Save the uploaded bytes to a temporary PDF file and process it
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            pages = load_pdf(tmp.name)
        os.remove(tmp.name)

        chunks = chunk_documents(pages)

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(texts)

        metadata = [
            {"source": c["source"], "page_number": c.get("page_number"), "chunk_id": c.get("chunk_id"), "text": c["text"]}
            for c in chunks
        ]

        vector_db.build(embeddings, metadata)

        logger.info("Index successfully built")

        return {"message": "PDF processed and indexed", "chunks_indexed": len(chunks)}

    except Exception as e:
        logger.exception("Upload failed")
        raise CustomException(e)

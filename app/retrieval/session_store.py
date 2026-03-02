"""Session-based document store for multi-document indexing."""

import uuid
from typing import Dict, List, Optional
from app.retrieval.vector_store import VectorStoreFAISS
from app.retrieval.sparse_retriever import SparseRetrieverBM25
from app.ingestion.embedder import get_embeddings
from app.core.logger import logger


class Session:
    """Holds FAISS + BM25 indexes for a single user session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.vector_store = VectorStoreFAISS(
            index_path=f"data/processed/{session_id}_faiss.index",
            meta_path=f"data/processed/{session_id}_metadata.json",
        )
        self.sparse_retriever = SparseRetrieverBM25()
        self.all_chunks: List[Dict] = []
        self.documents: List[str] = []

    def add_document(self, chunks: List[Dict], document_name: str):
        """Add chunks from a new document and rebuild indexes."""
        self.all_chunks.extend(chunks)
        self.documents.append(document_name)

        texts = [c["text"] for c in self.all_chunks]
        embeddings = get_embeddings(texts)

        metadata = [
            {
                "source": c.get("source", "Unknown"),
                "page_number": c.get("page_number"),
                "chunk_id": c.get("chunk_id", ""),
                "text": c["text"],
            }
            for c in self.all_chunks
        ]

        self.vector_store.build(embeddings, metadata)
        self.sparse_retriever.build(metadata)
        logger.info(f"Session {self.session_id}: indexed {len(self.all_chunks)} total chunks from {len(self.documents)} documents")


# In-memory session store
_sessions: Dict[str, Session] = {}


def get_or_create_session(session_id: Optional[str] = None) -> Session:
    """Get existing session or create a new one."""
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id)
        logger.info(f"Created new session: {session_id}")

    return _sessions[session_id]


def get_session(session_id: str) -> Optional[Session]:
    """Get an existing session, returns None if not found."""
    return _sessions.get(session_id)

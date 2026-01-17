"""
Session-based vector store management.
Each session gets an isolated FAISS index that persists until browser close.
"""

import uuid
import json
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import faiss
from app.core.logger import logger
from app.ingestion.embedder import get_embeddings


class SessionVectorStore:
    """
    Manages a session-scoped FAISS vector store.
    Isolated from other sessions, cleaned up on session end.
    """

    def __init__(self, session_id: str, embedding_dim: int = 384):
        """
        Initialize a session vector store.

        Args:
            session_id: Unique session identifier
            embedding_dim: Dimension of embedding vectors
        """
        self.session_id = session_id
        self.embedding_dim = embedding_dim
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata: List[Dict] = []
        self.documents: List[str] = []  # Track uploaded document names

        logger.info(f"Created session vector store: {session_id}")

    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray) -> int:
        """
        Add chunks with their embeddings to the session store.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: NumPy array of embeddings

        Returns:
            Number of chunks added
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.get("chunk_id", str(uuid.uuid4())),
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "page_number": chunk.get("page_number", 0),
                "section": chunk.get("section"),
                "heading": chunk.get("heading"),
                "element_type": chunk.get("element_type", "text"),
            })

        self.last_accessed = datetime.utcnow()
        logger.info(f"Session {self.session_id}: Added {len(chunks)} chunks")
        return len(chunks)

    def add_document(self, document_name: str):
        """Track uploaded document names."""
        if document_name not in self.documents:
            self.documents.append(document_name)
            logger.info(f"Session {self.session_id}: Added document {document_name}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            score_threshold: Optional max distance threshold

        Returns:
            List of matching chunks with scores
        """
        self.last_accessed = datetime.utcnow()

        if self.index.ntotal == 0:
            logger.warning(f"Session {self.session_id}: Empty index")
            return []

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(
            query_embedding.astype(np.float32).reshape(1, -1),
            k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            # Convert L2 distance to similarity score (0-1)
            similarity = 1.0 / (1.0 + dist)

            if score_threshold and similarity < score_threshold:
                continue

            result = self.metadata[idx].copy()
            result["score"] = float(similarity)
            result["distance"] = float(dist)
            results.append(result)

        logger.info(f"Session {self.session_id}: Found {len(results)} results")
        return results

    def get_stats(self) -> Dict:
        """Get session store statistics."""
        return {
            "session_id": self.session_id,
            "num_chunks": self.index.ntotal,
            "num_documents": len(self.documents),
            "documents": self.documents,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
        }

    def clear(self):
        """Clear all data from the session store."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self.documents = []
        logger.info(f"Session {self.session_id}: Cleared all data")


class SessionManager:
    """
    Manages multiple session vector stores.
    Handles session creation, retrieval, and cleanup.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.sessions: Dict[str, SessionVectorStore] = {}
        self.session_lock = threading.Lock()
        self._initialized = True
        logger.info("SessionManager initialized")

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session with isolated vector store.

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        with self.session_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionVectorStore(session_id)
                logger.info(f"Created new session: {session_id}")
            else:
                logger.info(f"Session already exists: {session_id}")

        return session_id

    def get_session(self, session_id: str) -> Optional[SessionVectorStore]:
        """
        Get an existing session store.

        Args:
            session_id: Session ID

        Returns:
            SessionVectorStore or None if not found
        """
        with self.session_lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed = datetime.utcnow()
            return session

    def get_or_create_session(self, session_id: str) -> SessionVectorStore:
        """
        Get existing session or create new one.

        Args:
            session_id: Session ID

        Returns:
            SessionVectorStore
        """
        session = self.get_session(session_id)
        if session is None:
            self.create_session(session_id)
            session = self.sessions[session_id]
        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and clean up resources.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False

    def list_sessions(self) -> List[Dict]:
        """List all active sessions with stats."""
        with self.session_lock:
            return [session.get_stats() for session in self.sessions.values()]

    def cleanup_inactive_sessions(self, max_idle_minutes: int = 60):
        """
        Remove sessions that have been idle too long.

        Args:
            max_idle_minutes: Maximum idle time before cleanup
        """
        cutoff = datetime.utcnow() - timedelta(minutes=max_idle_minutes)

        with self.session_lock:
            sessions_to_delete = [
                sid for sid, session in self.sessions.items()
                if session.last_accessed < cutoff
            ]

            for sid in sessions_to_delete:
                del self.sessions[sid]
                logger.info(f"Cleaned up inactive session: {sid}")

            if sessions_to_delete:
                logger.info(f"Cleaned up {len(sessions_to_delete)} inactive sessions")


# Global session manager instance
session_manager = SessionManager()


def get_session_store(session_id: str) -> SessionVectorStore:
    """
    Convenience function to get or create a session store.

    Args:
        session_id: Session ID

    Returns:
        SessionVectorStore instance
    """
    return session_manager.get_or_create_session(session_id)


def add_to_session(
    session_id: str,
    chunks: List[Dict],
    document_name: str
) -> Dict:
    """
    Add chunks to a session's vector store.

    Args:
        session_id: Session ID
        chunks: List of chunk dictionaries
        document_name: Name of the source document

    Returns:
        Dict with operation results
    """
    store = get_session_store(session_id)

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)

    # Add to store
    num_added = store.add_chunks(chunks, embeddings)
    store.add_document(document_name)

    return {
        "session_id": session_id,
        "chunks_added": num_added,
        "document": document_name,
        "total_chunks": store.index.ntotal,
        "total_documents": len(store.documents),
    }


def search_session(
    session_id: str,
    query: str,
    k: int = 5
) -> List[Dict]:
    """
    Search within a session's vector store.

    Args:
        session_id: Session ID
        query: Search query
        k: Number of results

    Returns:
        List of matching chunks
    """
    store = session_manager.get_session(session_id)

    if store is None:
        logger.warning(f"Session not found: {session_id}")
        return []

    # Generate query embedding
    query_embedding = get_embeddings([query])[0]

    return store.search(query_embedding, k=k)

"""
BM25 sparse retrieval for hybrid search.
Combines well with dense retrieval for improved recall.
"""

import re
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from app.core.logger import logger


class BM25Retriever:
    """
    BM25-based sparse retriever for keyword matching.
    Complements dense semantic search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 tokenization.
        Simple tokenization with lowercasing and punctuation removal.
        """
        # Lowercase
        text = text.lower()

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Split and filter empty tokens
        tokens = [token.strip() for token in text.split() if token.strip()]

        return tokens

    def fit(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' key
        """
        self.documents = chunks
        self.tokenized_corpus = [
            self._preprocess_text(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        logger.info(f"BM25 index built with {len(chunks)} documents")

    def search(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search using BM25.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum score threshold

        Returns:
            List of matching chunks with BM25 scores
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built. Call fit() first.")
            return []

        # Tokenize query
        query_tokens = self._preprocess_text(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= score_threshold:
                continue

            result = self.documents[idx].copy()
            result["bm25_score"] = float(score)
            results.append(result)

        logger.info(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
        return results

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get raw BM25 scores for all documents.

        Args:
            query: Search query

        Returns:
            NumPy array of scores
        """
        if self.bm25 is None:
            return np.array([])

        query_tokens = self._preprocess_text(query)
        return self.bm25.get_scores(query_tokens)


class SessionBM25Store:
    """
    Session-scoped BM25 store for hybrid retrieval.
    Maintains separate BM25 index per session.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.retriever = BM25Retriever()
        self.chunks: List[Dict] = []

    def add_chunks(self, chunks: List[Dict]):
        """Add chunks and rebuild BM25 index."""
        self.chunks.extend(chunks)
        self.retriever.fit(self.chunks)
        logger.info(f"Session {self.session_id}: BM25 index updated with {len(self.chunks)} total chunks")

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search within session's BM25 index."""
        return self.retriever.search(query, k=k)

    def get_scores(self, query: str) -> np.ndarray:
        """Get raw BM25 scores for hybrid retrieval."""
        return self.retriever.get_scores(query)

    def clear(self):
        """Clear all data."""
        self.chunks = []
        self.retriever = BM25Retriever()


# Global session BM25 stores
_session_bm25_stores: Dict[str, SessionBM25Store] = {}


def get_session_bm25(session_id: str) -> SessionBM25Store:
    """Get or create a session BM25 store."""
    if session_id not in _session_bm25_stores:
        _session_bm25_stores[session_id] = SessionBM25Store(session_id)
    return _session_bm25_stores[session_id]


def delete_session_bm25(session_id: str):
    """Delete a session's BM25 store."""
    if session_id in _session_bm25_stores:
        del _session_bm25_stores[session_id]
        logger.info(f"Deleted BM25 store for session: {session_id}")

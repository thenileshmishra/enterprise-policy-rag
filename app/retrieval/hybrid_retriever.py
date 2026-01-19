"""
Hybrid retrieval combining dense (FAISS) and sparse (BM25) search.
Uses Reciprocal Rank Fusion (RRF) for score combination.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from app.core.logger import logger
from app.retrieval.session_store import get_session_store, SessionVectorStore
from app.retrieval.sparse_retriever import get_session_bm25, SessionBM25Store
from app.ingestion.embedder import get_embeddings


class HybridRetriever:
    """
    Combines dense semantic search with sparse BM25 search.
    Uses Reciprocal Rank Fusion for score combination.
    """

    def __init__(
        self,
        session_id: str,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            session_id: Session ID for isolated retrieval
            dense_weight: Weight for dense (semantic) scores
            sparse_weight: Weight for sparse (BM25) scores
            rrf_k: RRF constant (default 60)
        """
        self.session_id = session_id
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        # Get session stores
        self.dense_store: SessionVectorStore = get_session_store(session_id)
        self.sparse_store: SessionBM25Store = get_session_bm25(session_id)

        logger.info(f"Hybrid retriever initialized for session: {session_id}")

    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Add chunks to both dense and sparse indexes.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Dense embeddings for chunks
        """
        # Add to dense store (FAISS)
        self.dense_store.add_chunks(chunks, embeddings)

        # Add to sparse store (BM25)
        self.sparse_store.add_chunks(chunks)

        logger.info(f"Added {len(chunks)} chunks to hybrid index")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result list
        """
        # Build chunk_id to result mapping
        all_chunks: Dict[str, Dict] = {}
        rrf_scores: Dict[str, float] = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result.get("chunk_id", result.get("text", "")[:50])

            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = result

            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result.get("chunk_id", result.get("text", "")[:50])

            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = result

            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score

        # Sort by RRF score and build final results
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids:
            result = all_chunks[chunk_id].copy()
            result["hybrid_score"] = rrf_scores[chunk_id]
            results.append(result)

        return results

    def _weighted_combination(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict]:
        """
        Combine results using weighted score combination.
        Requires normalized scores.
        """
        # Build chunk_id mapping with scores
        chunks_scores: Dict[str, Dict] = {}

        # Process dense results
        dense_scores = [r.get("score", 0) for r in dense_results]
        normalized_dense = self._normalize_scores(dense_scores)

        for result, norm_score in zip(dense_results, normalized_dense):
            chunk_id = result.get("chunk_id", result.get("text", "")[:50])
            chunks_scores[chunk_id] = {
                "result": result,
                "dense_score": norm_score,
                "sparse_score": 0.0,
            }

        # Process sparse results
        sparse_scores = [r.get("bm25_score", 0) for r in sparse_results]
        normalized_sparse = self._normalize_scores(sparse_scores)

        for result, norm_score in zip(sparse_results, normalized_sparse):
            chunk_id = result.get("chunk_id", result.get("text", "")[:50])

            if chunk_id in chunks_scores:
                chunks_scores[chunk_id]["sparse_score"] = norm_score
            else:
                chunks_scores[chunk_id] = {
                    "result": result,
                    "dense_score": 0.0,
                    "sparse_score": norm_score,
                }

        # Calculate combined scores
        results = []
        for chunk_id, data in chunks_scores.items():
            combined_score = (
                self.dense_weight * data["dense_score"] +
                self.sparse_weight * data["sparse_score"]
            )

            result = data["result"].copy()
            result["hybrid_score"] = combined_score
            result["dense_score"] = data["dense_score"]
            result["sparse_score"] = data["sparse_score"]
            results.append(result)

        # Sort by combined score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return results

    def search(
        self,
        query: str,
        k: int = 10,
        fusion_method: str = "rrf",
        dense_k: int = 20,
        sparse_k: int = 20,
    ) -> List[Dict]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of final results to return
            fusion_method: "rrf" for Reciprocal Rank Fusion, "weighted" for weighted combination
            dense_k: Number of dense results to retrieve
            sparse_k: Number of sparse results to retrieve

        Returns:
            List of matching chunks with hybrid scores
        """
        # Dense search
        query_embedding = get_embeddings([query])[0]
        dense_results = self.dense_store.search(query_embedding, k=dense_k)

        # Sparse search
        sparse_results = self.sparse_store.search(query, k=sparse_k)

        # Combine results
        if fusion_method == "rrf":
            combined = self._reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            combined = self._weighted_combination(dense_results, sparse_results)

        logger.info(
            f"Hybrid search: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"= {len(combined)} combined (returning top {k})"
        )

        return combined[:k]

    def search_dense_only(self, query: str, k: int = 10) -> List[Dict]:
        """Perform dense-only search for comparison."""
        query_embedding = get_embeddings([query])[0]
        return self.dense_store.search(query_embedding, k=k)

    def search_sparse_only(self, query: str, k: int = 10) -> List[Dict]:
        """Perform sparse-only search for comparison."""
        return self.sparse_store.search(query, k=k)


# Session hybrid retrievers cache
_session_hybrid_retrievers: Dict[str, HybridRetriever] = {}


def get_hybrid_retriever(
    session_id: str,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> HybridRetriever:
    """
    Get or create a hybrid retriever for a session.

    Args:
        session_id: Session ID
        dense_weight: Weight for dense scores
        sparse_weight: Weight for sparse scores

    Returns:
        HybridRetriever instance
    """
    cache_key = f"{session_id}_{dense_weight}_{sparse_weight}"

    if cache_key not in _session_hybrid_retrievers:
        _session_hybrid_retrievers[cache_key] = HybridRetriever(
            session_id=session_id,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    return _session_hybrid_retrievers[cache_key]


def hybrid_search(
    session_id: str,
    query: str,
    k: int = 10,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> List[Dict]:
    """
    Convenience function for hybrid search.

    Args:
        session_id: Session ID
        query: Search query
        k: Number of results
        dense_weight: Weight for dense scores
        sparse_weight: Weight for sparse scores

    Returns:
        List of matching chunks
    """
    retriever = get_hybrid_retriever(session_id, dense_weight, sparse_weight)
    return retriever.search(query, k=k)

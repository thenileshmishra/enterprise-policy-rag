"""
Cross-encoder reranking for improved retrieval precision.
Reranks candidate documents using a more powerful model.
"""

from typing import List, Dict, Optional, Tuple
import torch
from sentence_transformers import CrossEncoder
from app.core.logger import logger


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.
    Cross-encoders jointly encode query and document for better relevance scoring.
    """

    _instance = None
    _model = None

    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker (singleton).

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        if self._initialized:
            return

        self.model_name = model_name
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"Cross-encoder loaded on device: {self._model.model.device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Rerank documents based on cross-encoder relevance scores.

        Args:
            query: Search query
            documents: List of document dicts with 'text' key
            top_k: Return only top k results (None for all)
            score_threshold: Minimum score threshold (None for no threshold)

        Returns:
            Reranked list of documents with cross-encoder scores
        """
        if not documents:
            return []

        if self._model is None:
            logger.warning("Cross-encoder not loaded, returning original order")
            return documents

        # Prepare query-document pairs
        pairs = [(query, doc.get("text", "")) for doc in documents]

        try:
            # Get cross-encoder scores
            scores = self._model.predict(pairs, show_progress_bar=False)

            # Add scores to documents
            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(score)
                scored_docs.append(doc_copy)

            # Sort by rerank score (descending)
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Apply score threshold
            if score_threshold is not None:
                scored_docs = [d for d in scored_docs if d["rerank_score"] >= score_threshold]

            # Apply top_k
            if top_k is not None:
                scored_docs = scored_docs[:top_k]

            logger.info(f"Reranked {len(documents)} docs -> {len(scored_docs)} results")
            return scored_docs

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents

    def score_pair(self, query: str, document: str) -> float:
        """
        Get relevance score for a single query-document pair.

        Args:
            query: Search query
            document: Document text

        Returns:
            Relevance score
        """
        if self._model is None:
            return 0.0

        try:
            score = self._model.predict([(query, document)])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0


class DiversityReranker:
    """
    Reranks with diversity consideration using Maximal Marginal Relevance (MMR).
    Balances relevance with diversity to avoid redundant results.
    """

    def __init__(
        self,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        lambda_param: float = 0.5,
    ):
        """
        Initialize diversity reranker.

        Args:
            cross_encoder: Optional cross-encoder for relevance scoring
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
        """
        self.cross_encoder = cross_encoder or CrossEncoderReranker()
        self.lambda_param = lambda_param

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute text similarity using character overlap (simple approach).
        For production, use embedding similarity.
        """
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def rerank_mmr(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rerank using Maximal Marginal Relevance for diversity.

        Args:
            query: Search query
            documents: List of document dicts
            top_k: Number of diverse results to return

        Returns:
            Diversified list of documents
        """
        if not documents or top_k <= 0:
            return []

        # First, get relevance scores from cross-encoder
        scored_docs = self.cross_encoder.rerank(query, documents)

        if len(scored_docs) <= 1:
            return scored_docs[:top_k]

        # MMR selection
        selected = []
        remaining = scored_docs.copy()

        # Select first document (most relevant)
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_score = float("-inf")
            best_idx = 0

            for i, doc in enumerate(remaining):
                # Relevance score
                relevance = doc.get("rerank_score", 0)

                # Max similarity to already selected docs
                max_sim = max(
                    self._compute_similarity(doc["text"], sel["text"])
                    for sel in selected
                )

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        logger.info(f"MMR reranking: {len(documents)} -> {len(selected)} diverse results")
        return selected


# Global singleton reranker
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def rerank_results(
    query: str,
    results: List[Dict],
    top_k: int = 5,
    use_diversity: bool = False,
) -> List[Dict]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query
        results: List of search results
        top_k: Number of results to return
        use_diversity: Whether to use MMR for diversity

    Returns:
        Reranked list of results
    """
    reranker = get_reranker()

    if use_diversity:
        diversity_reranker = DiversityReranker(cross_encoder=reranker)
        return diversity_reranker.rerank_mmr(query, results, top_k=top_k)
    else:
        return reranker.rerank(query, results, top_k=top_k)

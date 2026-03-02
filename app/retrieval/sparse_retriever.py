"""BM25 sparse retriever using rank-bm25."""

from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from app.core.logger import logger


class SparseRetrieverBM25:
    def __init__(self):
        self.bm25 = None
        self.metadata = []

    def build(self, metadata: List[Dict]):
        """Build BM25 index from chunk metadata (must contain 'text' key)."""
        texts = [m["text"] for m in metadata]
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.metadata = metadata
        logger.info(f"BM25 index built with {len(metadata)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Return top-k results as (metadata, bm25_score) tuples."""
        if self.bm25 is None or not self.metadata:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                results.append((self.metadata[idx], float(scores[idx])))

        return results

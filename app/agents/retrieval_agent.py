"""
Retrieval Agent for AutoGen-based RAG pipeline.
Handles document retrieval using hybrid search and reranking.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from app.core.logger import logger
from app.retrieval.hybrid_retriever import get_hybrid_retriever, HybridRetriever
from app.retrieval.reranker import rerank_results
from app.retrieval.session_store import get_session_store
from app.retrieval.sparse_retriever import get_session_bm25
from app.ingestion.embedder import get_embeddings


@dataclass
class RetrievalConfig:
    """Configuration for retrieval agent."""
    k: int = 10
    rerank_k: int = 5
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    use_reranking: bool = True
    use_diversity: bool = False
    score_threshold: Optional[float] = None


class RetrievalAgent:
    """
    Agent specialized in document retrieval.
    Combines hybrid search with reranking for optimal results.
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize retrieval agent.

        Args:
            session_id: Session ID for isolated retrieval
            config: Optional retrieval configuration
        """
        self.session_id = session_id
        self.config = config or RetrievalConfig()
        self.retriever = get_hybrid_retriever(
            session_id=session_id,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
        )

        logger.info(f"RetrievalAgent initialized for session: {session_id}")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            k: Number of results (overrides config)
            use_reranking: Whether to rerank (overrides config)

        Returns:
            List of relevant document chunks
        """
        k = k or self.config.k
        use_reranking = use_reranking if use_reranking is not None else self.config.use_reranking

        # Initial hybrid retrieval (get more candidates for reranking)
        retrieval_k = k * 3 if use_reranking else k
        results = self.retriever.search(
            query=query,
            k=retrieval_k,
            dense_k=retrieval_k,
            sparse_k=retrieval_k,
        )

        if not results:
            logger.warning(f"No results found for query: {query[:50]}...")
            return []

        # Apply reranking
        if use_reranking and len(results) > 1:
            rerank_k = self.config.rerank_k if self.config.rerank_k else k
            results = rerank_results(
                query=query,
                results=results,
                top_k=rerank_k,
                use_diversity=self.config.use_diversity,
            )
        else:
            results = results[:k]

        # Apply score threshold
        if self.config.score_threshold:
            results = [
                r for r in results
                if r.get("hybrid_score", r.get("rerank_score", 0)) >= self.config.score_threshold
            ]

        logger.info(f"Retrieved {len(results)} documents for query")
        return results

    def retrieve_with_metadata(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve documents with additional metadata about the retrieval.

        Args:
            query: User query
            k: Number of results

        Returns:
            Dict with results and metadata
        """
        results = self.retrieve(query, k)

        # Gather source statistics
        sources = {}
        for result in results:
            source = result.get("source", "Unknown")
            if source not in sources:
                sources[source] = 0
            sources[source] += 1

        return {
            "query": query,
            "results": results,
            "num_results": len(results),
            "sources": sources,
            "config": {
                "k": k or self.config.k,
                "use_reranking": self.config.use_reranking,
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight,
            },
        }

    def add_documents(self, chunks: List[Dict]) -> int:
        """
        Add documents to the retrieval index.

        Args:
            chunks: List of document chunks

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Get embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(texts)

        # Add to hybrid retriever (both dense and sparse)
        self.retriever.add_chunks(chunks, embeddings)

        logger.info(f"Added {len(chunks)} chunks to retrieval index")
        return len(chunks)

    def get_index_stats(self) -> Dict:
        """Get statistics about the retrieval index."""
        dense_store = get_session_store(self.session_id)
        sparse_store = get_session_bm25(self.session_id)

        return {
            "session_id": self.session_id,
            "dense_index_size": dense_store.index.ntotal,
            "sparse_index_size": len(sparse_store.chunks),
            "documents": dense_store.documents,
        }


def create_retrieval_agent(
    session_id: str,
    k: int = 10,
    use_reranking: bool = True,
    dense_weight: float = 0.5,
) -> RetrievalAgent:
    """
    Factory function to create a configured retrieval agent.

    Args:
        session_id: Session ID
        k: Number of results to return
        use_reranking: Whether to use cross-encoder reranking
        dense_weight: Weight for dense (semantic) retrieval

    Returns:
        Configured RetrievalAgent
    """
    config = RetrievalConfig(
        k=k,
        rerank_k=k,
        dense_weight=dense_weight,
        sparse_weight=1.0 - dense_weight,
        use_reranking=use_reranking,
    )

    return RetrievalAgent(session_id=session_id, config=config)

from typing import List, Dict, Any
from ..ingestion.embedder import get_embeddings, get_query_embedding
from .vector_store import VectorStoreFAISS


class Retriever:
    def __init__(self):
        self.store = VectorStoreFAISS()


    def index_documents(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        metadata = [
            {
                "source": c.get("source", "Unknown"),
                "page_number": c.get("page_number"),
                "text": c["text"],
            }
            for c in chunks
        ]

        embeddings = get_embeddings(texts)
        self.store.build(embeddings, metadata)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = get_query_embedding(query)
        return self.store.search(query_embedding, top_k=top_k)


def retrieve_topk(query: str, top_k: int = 5):
    """Return top-k chunks as dictionaries expected by the RAG pipeline."""
    raw = Retriever().retrieve(query, top_k=top_k)
    result = []
    for meta, score in raw:
        result.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", "Unknown"),
            "page_number": meta.get("page_number"),
            "score": score,
        })

    return result

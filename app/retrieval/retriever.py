from typing import List, Dict, Any
from ..ingestion.embedder import Embedder
from .vector_store import VectorStoreFAISS


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStoreFAISS()


    def index_documents(self, chunks: List[Dict[str, Any]]):
        texts = [c["text"] for c in chunks]
        metadata = [{"source": c["source"], "page": c.get("page", None)} for c in chunks]

        embeddings = self.embedder.embed_texts(texts)
        self.store.build(embeddings, metadata)

    def retrive(self, query:str, top_k: int = 5):
        query_embedding = self.embedder.embed_single(query)
        return self.store.search(query_embedding, top_k=top_k)


def retrieve_topk(query: str, top_k: int = 5):
    """Module-level helper used by the API to return top-k chunks for a query."""
    return Retriever().retrive(query, top_k=top_k)

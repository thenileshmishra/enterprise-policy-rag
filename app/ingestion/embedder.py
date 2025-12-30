import os
from sentence_transformers import SentenceTransformer
from typing import List
from ..core.logger import logger
from ..core.exception import CustomException



class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str])-> List[List[float]]:
        if not texts:
            raise CustomException("No texts provided for embedding")
        logger.info(f"Embedding {len(texts)} chunks") 
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_single(self, text: str):
        return self.model.encode([text])[0].tolist()
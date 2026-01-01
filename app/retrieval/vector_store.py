import os
import faiss
import numpy as np
import json
from typing import List, Dict, Any, Tuple
from ..core.logger import logger
from ..core.exception import CustomException


class VectorStoreFAISS:
    def __init__(self, index_path: str = "data/processed/faiss.index",
                 meta_path: str = "data/processed/metadata.json",
                 dim: int = 384):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim
        self.index = None
        self.metadata = []

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    def build(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        logger.info("Building FAISS index")

        if len(embeddings) != len(metadata):
            raise CustomException("Embeddings & metadata length mismatch")
        
        embeddings_np = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings_np)

        self.metadata = metadata
        self.save()

        logger.info(f"FAISS index built with {len(metadata)} entries")
    
    def save(self):
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w") as f:
                json.dump(self.metadata, f)
            logger.info("FAISS index + metadata saved")
    
    def load(self):
        # If S3 is configured, try to download index from object storage first
        s3_bucket = os.getenv("S3_BUCKET")
        s3_index_key = os.getenv("S3_INDEX_KEY", "faiss.index")
        s3_meta_key = os.getenv("S3_META_KEY", "metadata.json")
        if s3_bucket:
            try:
                from app.retrieval.storage import download_index

                ok = download_index(s3_bucket, s3_index_key, s3_meta_key, self.index_path, self.meta_path)
                if not ok:
                    logger.warning("S3 index download failed; falling back to local index if available")
            except Exception:
                logger.exception("S3 index helpers failed; continuing to local index check")

        if not os.path.exists(self.index_path):
            raise CustomException("FAISS index missing — please upload a PDF or configure S3 index to download")

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r") as f:
            self.metadata = json.load(f)
        logger.info("FAISS index loaded succesfully")
    
    def search(self, query_embedding, top_k: int=5) -> List[Tuple[Dict, float]]:
        if self.index is None:
            self.load()

        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_np, top_k)

        retults = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            retults.append((self.metadata[idx], float(dist)))
        return retults

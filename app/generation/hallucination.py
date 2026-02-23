"""Simple grounding confidence based on embedding cosine similarity."""

from typing import List
from sentence_transformers import SentenceTransformer, util
from app.core.logger import logger

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def hallucination_score(answer: str, contexts: List[str]) -> float:
    if not answer or not contexts:
        return 0.0

    try:
        model = _get_model()
        answer_emb = model.encode(answer, convert_to_tensor=True)
        context_embs = model.encode(contexts, convert_to_tensor=True)
        score = util.cos_sim(answer_emb, context_embs).max().item()
        return round(float(score), 4)
    except Exception:
        logger.exception("Could not compute grounding score")
        return 0.0

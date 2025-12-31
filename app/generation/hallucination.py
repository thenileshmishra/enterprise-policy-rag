"""
Hallucination Detection
Compares LLM answer with retrieved chunks
Computes similarity confidence score
"""

from sentence_transformers import SentenceTransformer, util
from typing import List
from app.core.logger import logger


model = SentenceTransformer("all-MiniLM-L6-v2")


def hallucination_score(answer: str, contexts: List[str]) -> float:
    """
    Returns similarity score between answer & retrieval contexts.
    Higher = more grounded

    Threshold meaning (you can tune):
    > 0.75 = strong grounding
    0.55 - 0.75 = acceptable
    < 0.55 = likely hallucination
    """
    try:
        answer_emb = model.encode(answer, convert_to_tensor=True)
        ctx_emb = model.encode(contexts, convert_to_tensor=True)

        score = util.cos_sim(answer_emb, ctx_emb).max().item()
        return round(float(score), 4)

    except Exception as e:
        logger.exception("Hallucination scoring failed")
        return 0.0

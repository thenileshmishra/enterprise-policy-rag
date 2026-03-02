"""Faithfulness detection using semantic similarity + NLI entailment scoring."""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from app.core.logger import logger

_sim_model: SentenceTransformer | None = None
_nli_model: CrossEncoder | None = None


def _get_sim_model() -> SentenceTransformer:
    global _sim_model
    if _sim_model is None:
        _sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sim_model


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        logger.info("Loading NLI model for faithfulness detection...")
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def _similarity_score(answer: str, contexts: List[str]) -> float:
    """Cosine similarity between answer and best-matching context."""
    model = _get_sim_model()
    answer_emb = model.encode(answer, convert_to_tensor=True)
    context_embs = model.encode(contexts, convert_to_tensor=True)
    return float(util.cos_sim(answer_emb, context_embs).max().item())


def _nli_score(answer: str, contexts: List[str]) -> float:
    """
    NLI entailment score: checks if contexts entail the answer.
    Model outputs [contradiction, entailment, neutral] logits.
    Returns average entailment probability across context chunks.
    """
    model = _get_nli_model()
    pairs = [[ctx, answer] for ctx in contexts]
    scores = model.predict(pairs)

    scores = np.array(scores)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    # Softmax to get probabilities
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Max entailment probability across all contexts
    entailment_probs = probs[:, 1]
    return float(entailment_probs.max())


def faithfulness_check(answer: str, contexts: List[str]) -> Dict:
    """
    Combined faithfulness check using similarity + NLI.
    Returns dict with scores, is_grounded flag, and confidence level.
    """
    if not answer or not contexts:
        return {
            "similarity_score": 0.0,
            "nli_score": 0.0,
            "faithfulness_score": 0.0,
            "is_grounded": False,
            "confidence_level": "none",
        }

    try:
        sim = _similarity_score(answer, contexts)
        nli = _nli_score(answer, contexts)

        # Combined score: weighted average (NLI is more precise)
        combined = round(0.4 * sim + 0.6 * nli, 4)

        # Determine grounding
        is_grounded = combined >= 0.5

        if combined >= 0.75:
            confidence = "high"
        elif combined >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "similarity_score": round(sim, 4),
            "nli_score": round(nli, 4),
            "faithfulness_score": combined,
            "is_grounded": is_grounded,
            "confidence_level": confidence,
        }

    except Exception:
        logger.exception("Faithfulness check failed, falling back to similarity-only")
        try:
            sim = _similarity_score(answer, contexts)
            return {
                "similarity_score": round(sim, 4),
                "nli_score": 0.0,
                "faithfulness_score": round(sim, 4),
                "is_grounded": sim >= 0.5,
                "confidence_level": "low",
            }
        except Exception:
            logger.exception("Complete faithfulness check failure")
            return {
                "similarity_score": 0.0,
                "nli_score": 0.0,
                "faithfulness_score": 0.0,
                "is_grounded": False,
                "confidence_level": "none",
            }

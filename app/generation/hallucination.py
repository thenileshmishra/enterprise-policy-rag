"""
Hallucination Detection and Faithfulness Verification.
Uses multiple approaches: semantic similarity, NLI, and claim verification.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from app.core.logger import logger


# Embedding model for similarity
_embedding_model: Optional[SentenceTransformer] = None

# NLI model for entailment checking
_nli_model: Optional[CrossEncoder] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model for hallucination detection")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_nli_model() -> Optional[CrossEncoder]:
    """Get or load the NLI model for entailment checking."""
    global _nli_model
    if _nli_model is None:
        try:
            logger.info("Loading NLI model for faithfulness verification")
            _nli_model = CrossEncoder(
                "cross-encoder/nli-deberta-v3-small",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            _nli_model = None
    return _nli_model


def hallucination_score(answer: str, contexts: List[str]) -> float:
    """
    Returns similarity score between answer & retrieval contexts.
    Higher = more grounded

    Threshold meaning:
    > 0.75 = strong grounding
    0.55 - 0.75 = acceptable
    < 0.55 = likely hallucination
    """
    if not answer or not contexts:
        return 0.0

    try:
        model = get_embedding_model()
        answer_emb = model.encode(answer, convert_to_tensor=True)
        ctx_emb = model.encode(contexts, convert_to_tensor=True)

        score = util.cos_sim(answer_emb, ctx_emb).max().item()
        return round(float(score), 4)

    except Exception:
        logger.exception("Hallucination scoring failed")
        return 0.0


@dataclass
class FaithfulnessResult:
    """Result of faithfulness verification."""
    score: float
    is_faithful: bool
    entailment_scores: List[float]
    contradiction_detected: bool
    details: Dict


def compute_nli_faithfulness(
    answer: str,
    contexts: List[str],
    sentence_level: bool = True,
) -> FaithfulnessResult:
    """
    Compute faithfulness using NLI-based entailment checking.

    For each sentence in the answer, checks if it's entailed by any context.
    A faithful answer should have all sentences entailed by the context.

    Args:
        answer: Generated answer
        contexts: List of context strings
        sentence_level: Whether to check at sentence level

    Returns:
        FaithfulnessResult with detailed scores
    """
    nli_model = get_nli_model()

    if nli_model is None:
        # Fallback to similarity-based scoring
        similarity_score = hallucination_score(answer, contexts)
        return FaithfulnessResult(
            score=similarity_score,
            is_faithful=similarity_score >= 0.55,
            entailment_scores=[similarity_score],
            contradiction_detected=False,
            details={"method": "similarity_fallback"},
        )

    # Split answer into sentences if sentence_level
    if sentence_level:
        sentences = split_into_sentences(answer)
    else:
        sentences = [answer]

    if not sentences:
        return FaithfulnessResult(
            score=0.0,
            is_faithful=False,
            entailment_scores=[],
            contradiction_detected=False,
            details={"error": "No sentences to verify"},
        )

    # Combine contexts for premise
    combined_context = " ".join(contexts)

    entailment_scores = []
    contradiction_detected = False
    sentence_results = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        # NLI expects (premise, hypothesis) pairs
        # Context is premise, answer sentence is hypothesis
        pair = [(combined_context[:2000], sentence)]  # Limit context length

        try:
            # CrossEncoder returns logits for [contradiction, neutral, entailment]
            scores = nli_model.predict(pair, show_progress_bar=False)

            if len(scores.shape) > 1:
                # Multiple classes
                probs = torch.softmax(torch.tensor(scores), dim=-1).numpy()
                entailment_prob = float(probs[0][2])  # Entailment class
                contradiction_prob = float(probs[0][0])  # Contradiction class

                if contradiction_prob > 0.7:
                    contradiction_detected = True
            else:
                # Single score (some models)
                entailment_prob = float(scores[0])
                contradiction_prob = 0.0

            entailment_scores.append(entailment_prob)

            sentence_results.append({
                "sentence": sentence[:100],
                "entailment": entailment_prob,
                "contradiction": contradiction_prob,
            })

        except Exception as e:
            logger.warning(f"NLI scoring failed for sentence: {e}")
            entailment_scores.append(0.5)  # Neutral fallback

    # Aggregate scores
    if entailment_scores:
        avg_entailment = sum(entailment_scores) / len(entailment_scores)
        min_entailment = min(entailment_scores)

        # Faithfulness based on average and minimum
        is_faithful = avg_entailment >= 0.5 and min_entailment >= 0.3 and not contradiction_detected
    else:
        avg_entailment = 0.0
        is_faithful = False

    return FaithfulnessResult(
        score=round(avg_entailment, 4),
        is_faithful=is_faithful,
        entailment_scores=entailment_scores,
        contradiction_detected=contradiction_detected,
        details={
            "method": "nli",
            "num_sentences": len(sentences),
            "sentence_results": sentence_results[:5],  # Limit for brevity
        },
    )


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    import re

    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    return sentences


def comprehensive_faithfulness_check(
    answer: str,
    contexts: List[str],
    use_nli: bool = True,
    use_similarity: bool = True,
) -> Dict:
    """
    Comprehensive faithfulness check using multiple methods.

    Args:
        answer: Generated answer
        contexts: Context strings
        use_nli: Whether to use NLI-based checking
        use_similarity: Whether to use similarity-based checking

    Returns:
        Dict with comprehensive faithfulness analysis
    """
    results = {
        "answer_length": len(answer),
        "context_count": len(contexts),
        "methods_used": [],
    }

    scores = []

    # Similarity-based score
    if use_similarity:
        similarity_score = hallucination_score(answer, contexts)
        results["similarity_score"] = similarity_score
        results["methods_used"].append("similarity")
        scores.append(similarity_score)

    # NLI-based score
    if use_nli:
        nli_result = compute_nli_faithfulness(answer, contexts)
        results["nli_score"] = nli_result.score
        results["nli_details"] = nli_result.details
        results["contradiction_detected"] = nli_result.contradiction_detected
        results["methods_used"].append("nli")

        if nli_result.details.get("method") != "similarity_fallback":
            scores.append(nli_result.score)

    # Combined score (weighted average)
    if scores:
        combined_score = sum(scores) / len(scores)
        results["combined_score"] = round(combined_score, 4)

        # Determine faithfulness
        results["is_faithful"] = (
            combined_score >= 0.55 and
            not results.get("contradiction_detected", False)
        )
    else:
        results["combined_score"] = 0.0
        results["is_faithful"] = False

    # Confidence level
    if results["combined_score"] >= 0.75:
        results["confidence"] = "high"
    elif results["combined_score"] >= 0.55:
        results["confidence"] = "medium"
    else:
        results["confidence"] = "low"

    logger.info(
        f"Faithfulness check: score={results['combined_score']:.3f}, "
        f"confident={results['confidence']}, faithful={results['is_faithful']}"
    )

    return results


def detect_unsupported_claims(
    answer: str,
    contexts: List[str],
) -> List[Dict]:
    """
    Identify specific claims in the answer that may not be supported.

    Args:
        answer: Generated answer
        contexts: Context strings

    Returns:
        List of potentially unsupported claims
    """
    sentences = split_into_sentences(answer)
    combined_context = " ".join(contexts).lower()

    unsupported = []

    model = get_embedding_model()
    context_emb = model.encode(combined_context, convert_to_tensor=True)

    for sentence in sentences:
        if len(sentence) < 20:
            continue

        sentence_emb = model.encode(sentence, convert_to_tensor=True)
        similarity = util.cos_sim(sentence_emb, context_emb).item()

        if similarity < 0.4:
            unsupported.append({
                "claim": sentence[:200],
                "similarity": round(similarity, 3),
                "reason": "Low semantic similarity to context",
            })

    return unsupported

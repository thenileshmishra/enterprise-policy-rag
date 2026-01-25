"""
RAGAS-based evaluation metrics for RAG system.
Implements faithfulness, answer relevance, context precision, and context recall.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from app.core.logger import logger


@dataclass
class RAGASMetrics:
    """Container for all RAGAS metrics."""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    details: Dict = field(default_factory=dict)


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class RAGASEvaluator:
    """
    RAGAS-style evaluator for RAG systems.
    Computes faithfulness, answer relevance, context precision, and context recall.
    """

    def __init__(
        self,
        use_llm_metrics: bool = False,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize RAGAS evaluator.

        Args:
            use_llm_metrics: Whether to use LLM-based metrics (requires API)
            embedding_model_name: Model for embedding-based metrics
        """
        self.use_llm_metrics = use_llm_metrics
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None

        logger.info(f"RAGASEvaluator initialized (LLM metrics: {use_llm_metrics})")

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def compute_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """
        Compute faithfulness: How factually consistent is the answer with the context?

        Uses semantic similarity between answer claims and context.
        Score ranges from 0 to 1 (higher is better).

        Args:
            answer: Generated answer
            contexts: Retrieved contexts

        Returns:
            Faithfulness score
        """
        if not answer or not contexts:
            return 0.0

        model = self._get_embedding_model()

        # Split answer into sentences (claims)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]

        if not sentences:
            return 0.0

        # Combine contexts
        combined_context = " ".join(contexts)

        # Compute similarity for each sentence
        from sentence_transformers import util

        context_emb = model.encode(combined_context, convert_to_tensor=True)
        faithfulness_scores = []

        for sentence in sentences:
            sentence_emb = model.encode(sentence, convert_to_tensor=True)
            similarity = util.cos_sim(sentence_emb, context_emb).item()
            faithfulness_scores.append(max(0, similarity))

        # Average faithfulness (each claim should be supported)
        return float(np.mean(faithfulness_scores))

    def compute_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Compute answer relevance: How relevant is the answer to the question?

        Measures semantic similarity between question and answer.
        Score ranges from 0 to 1 (higher is better).

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Answer relevance score
        """
        if not question or not answer:
            return 0.0

        model = self._get_embedding_model()
        from sentence_transformers import util

        question_emb = model.encode(question, convert_to_tensor=True)
        answer_emb = model.encode(answer, convert_to_tensor=True)

        similarity = util.cos_sim(question_emb, answer_emb).item()
        return float(max(0, similarity))

    def compute_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> float:
        """
        Compute context precision: How relevant are the retrieved contexts?

        Measures whether retrieved contexts are relevant to answering the question.
        Score ranges from 0 to 1 (higher is better).

        Args:
            question: User question
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer

        Returns:
            Context precision score
        """
        if not question or not contexts:
            return 0.0

        model = self._get_embedding_model()
        from sentence_transformers import util

        # Embed question (and ground truth if available)
        question_emb = model.encode(question, convert_to_tensor=True)

        if ground_truth:
            gt_emb = model.encode(ground_truth, convert_to_tensor=True)
            reference_emb = (question_emb + gt_emb) / 2
        else:
            reference_emb = question_emb

        # Compute relevance for each context
        precision_scores = []
        for i, ctx in enumerate(contexts):
            ctx_emb = model.encode(ctx, convert_to_tensor=True)
            relevance = util.cos_sim(ctx_emb, reference_emb).item()

            # Weight by position (earlier contexts should be more relevant)
            position_weight = 1.0 / (i + 1)
            weighted_relevance = relevance * position_weight
            precision_scores.append(weighted_relevance)

        # Normalize by ideal weights
        ideal_weights = [1.0 / (i + 1) for i in range(len(contexts))]
        normalized_score = sum(precision_scores) / sum(ideal_weights)

        return float(max(0, min(1, normalized_score)))

    def compute_context_recall(
        self,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """
        Compute context recall: Do the contexts contain the necessary information?

        Measures how much of the ground truth can be attributed to the contexts.
        Score ranges from 0 to 1 (higher is better).

        Args:
            contexts: Retrieved contexts
            ground_truth: Ground truth answer

        Returns:
            Context recall score
        """
        if not contexts or not ground_truth:
            return 0.0

        model = self._get_embedding_model()
        from sentence_transformers import util

        # Split ground truth into sentences
        import re
        gt_sentences = re.split(r'(?<=[.!?])\s+', ground_truth)
        gt_sentences = [s.strip() for s in gt_sentences if s.strip() and len(s) > 10]

        if not gt_sentences:
            return 0.0

        # Combine contexts
        combined_context = " ".join(contexts)
        context_emb = model.encode(combined_context, convert_to_tensor=True)

        # Check how many ground truth sentences are covered by context
        recall_scores = []
        for sentence in gt_sentences:
            sentence_emb = model.encode(sentence, convert_to_tensor=True)
            similarity = util.cos_sim(sentence_emb, context_emb).item()

            # Binary recall: is this info in context?
            recalled = 1.0 if similarity > 0.5 else similarity
            recall_scores.append(recalled)

        return float(np.mean(recall_scores))

    def evaluate(
        self,
        sample: EvaluationSample,
    ) -> RAGASMetrics:
        """
        Evaluate a single sample with all RAGAS metrics.

        Args:
            sample: EvaluationSample with question, answer, contexts, and optional ground_truth

        Returns:
            RAGASMetrics with all scores
        """
        # Compute individual metrics
        faithfulness = self.compute_faithfulness(sample.answer, sample.contexts)
        answer_relevance = self.compute_answer_relevance(sample.question, sample.answer)
        context_precision = self.compute_context_precision(
            sample.question,
            sample.contexts,
            sample.ground_truth
        )

        if sample.ground_truth:
            context_recall = self.compute_context_recall(sample.contexts, sample.ground_truth)
        else:
            # If no ground truth, estimate from answer
            context_recall = self.compute_context_recall(sample.contexts, sample.answer)

        # Compute overall score (harmonic mean of all metrics)
        metrics = [faithfulness, answer_relevance, context_precision, context_recall]
        metrics = [max(m, 0.001) for m in metrics]  # Avoid division by zero
        overall_score = len(metrics) / sum(1/m for m in metrics)

        return RAGASMetrics(
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(answer_relevance, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
            overall_score=round(overall_score, 4),
            details={
                "num_contexts": len(sample.contexts),
                "answer_length": len(sample.answer),
                "has_ground_truth": sample.ground_truth is not None,
            }
        )

    def evaluate_batch(
        self,
        samples: List[EvaluationSample],
    ) -> Dict[str, Any]:
        """
        Evaluate multiple samples and compute aggregate metrics.

        Args:
            samples: List of evaluation samples

        Returns:
            Dict with individual and aggregate metrics
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                metrics = self.evaluate(sample)
                results.append({
                    "index": i,
                    "question": sample.question[:100],
                    "metrics": metrics,
                })
            except Exception as e:
                logger.error(f"Failed to evaluate sample {i}: {e}")
                results.append({
                    "index": i,
                    "question": sample.question[:100],
                    "error": str(e),
                })

        # Compute aggregates
        valid_results = [r for r in results if "metrics" in r]

        if valid_results:
            aggregates = {
                "faithfulness": np.mean([r["metrics"].faithfulness for r in valid_results]),
                "answer_relevance": np.mean([r["metrics"].answer_relevance for r in valid_results]),
                "context_precision": np.mean([r["metrics"].context_precision for r in valid_results]),
                "context_recall": np.mean([r["metrics"].context_recall for r in valid_results]),
                "overall_score": np.mean([r["metrics"].overall_score for r in valid_results]),
            }
        else:
            aggregates = {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "overall_score": 0.0,
            }

        logger.info(
            f"Batch evaluation complete: {len(valid_results)}/{len(samples)} successful, "
            f"overall score: {aggregates['overall_score']:.4f}"
        )

        return {
            "individual_results": results,
            "aggregate_metrics": aggregates,
            "num_samples": len(samples),
            "num_successful": len(valid_results),
        }


def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    """
    Convenience function to evaluate a single RAG response.

    Args:
        question: User question
        answer: Generated answer
        contexts: Retrieved contexts
        ground_truth: Optional ground truth answer

    Returns:
        Dict with all RAGAS metrics
    """
    evaluator = RAGASEvaluator()
    sample = EvaluationSample(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    metrics = evaluator.evaluate(sample)

    return {
        "faithfulness": metrics.faithfulness,
        "answer_relevance": metrics.answer_relevance,
        "context_precision": metrics.context_precision,
        "context_recall": metrics.context_recall,
        "overall_score": metrics.overall_score,
    }

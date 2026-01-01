"""
Run retrieval + answer evaluation for dataset of QA pairs
"""

from app.evaluation.retrieval_eval import *
from app.evaluation.answer_eval import *


def evaluate_sample(query, retrieved_docs, relevant_docs, answer):
    metrics = {
        "precision@5": precision_at_k(retrieved_docs, relevant_docs, 5),
        "recall@5": recall_at_k(retrieved_docs, relevant_docs, 5),
        "mrr": mrr(retrieved_docs, relevant_docs),
        "ndcg@5": ndcg_at_k(retrieved_docs, relevant_docs, 5),
    }

    # Example inputs for answer metrics
    contexts = [d["text"] for d in retrieved_docs]
    sources = [d["metadata"] for d in retrieved_docs]

    metrics["faithfulness"] = faithfulness_score(answer, contexts)
    metrics["citation_coverage"] = citation_coverage(answer, sources)

    return metrics

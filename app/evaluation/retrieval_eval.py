from typing import List
import numpy as np


def precision_at_k(retrieved: List[str], relevant: List[str], k: int):
    retrieved_k = retrieved[:k]
    rel_count = sum([1 for x in retrieved_k if x in relevant])
    return rel_count / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int):
    retrieved_k = retrieved[:k]
    rel_count = sum([1 for x in retrieved_k if x in relevant])
    return rel_count / len(relevant) if relevant else 0


def mrr(retrieved: List[str], relevant: List[str]):
    for idx, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (idx + 1)
    return 0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int):
    retrieved_k = retrieved[:k]

    gains = [1 if doc in relevant else 0 for doc in retrieved_k]

    dcg = sum([(g / np.log2(idx + 2)) for idx, g in enumerate(gains)])

    ideal = sorted(gains, reverse=True)
    idcg = sum([(g / np.log2(idx + 2)) for idx, g in enumerate(ideal)])

    return dcg / idcg if idcg > 0 else 0

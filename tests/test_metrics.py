from app.evaluation.retrieval_eval import precision_at_k, recall_at_k, mrr, ndcg_at_k
from app.evaluation.answer_eval import faithfulness_score


def test_precision_recall_mrr_ndcg():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "D"]

    assert precision_at_k(retrieved, relevant, 2) >= 0
    assert recall_at_k(retrieved, relevant, 4) == 1
    assert mrr(retrieved, relevant) > 0
    assert ndcg_at_k(retrieved, relevant, 4) >= 0


def test_faithfulness():
    ans = "Employees get 30 days paid leave"
    ctx = ["Organization provides 30 days of paid annual leave"]

    score = faithfulness_score(ans, ctx)
    assert score > 0.5

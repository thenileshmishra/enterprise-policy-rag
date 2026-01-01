from sentence_transformers import SentenceTransformer, util
from typing import List

model = SentenceTransformer("all-MiniLM-L6-v2")


def faithfulness_score(answer: str, contexts: List[str]) -> float:
    """
    Measures semantic alignment between answer & retrieved context.
    """
    ans_emb = model.encode(answer, convert_to_tensor=True)
    ctx_emb = model.encode(contexts, convert_to_tensor=True)

    score = util.cos_sim(ans_emb, ctx_emb).max().item()
    return round(float(score), 4)


def citation_coverage(answer: str, sources: List[dict]):
    """
    Checks if answer references provided metadata sources
    """
    count = 0
    for s in sources:
        if str(s.get("page", "")) in answer or str(s.get("source", "")) in answer:
            count += 1

    return round(count / len(sources), 3) if sources else 0

from typing import Dict
from app.generation.prompt import build_rag_prompt
from app.generation.llm import LLMClient
from app.generation.hallucination import hallucination_score
from app.core.logger import logger


llm = LLMClient()


def rag_answer(query: str, retrieved_chunks: list) -> Dict:
    """
    retrieved_chunks expected format:
    [{"text": "...", "source": "...", "page_number": 1, "score": 0.1}, ...]
    """

    if not retrieved_chunks:
        return {
            "answer": "NOT FOUND in policy documentation.",
            "confidence": 0.0,
            "sources": []
        }

    contexts = [c["text"] for c in retrieved_chunks]

    prompt = build_rag_prompt(query, contexts)

    logger.info("Generating LLM answer...")
    answer = llm.generate(prompt)

    confidence = hallucination_score(answer, contexts)

    sources = sorted({c.get("source", "Unknown") for c in retrieved_chunks})

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
    }

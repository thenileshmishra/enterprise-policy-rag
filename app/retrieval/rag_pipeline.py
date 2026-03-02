"""End-to-end RAG pipeline: retrieve -> (rerank) -> generate -> faithfulness check."""

from typing import Dict
from app.generation.prompt import build_rag_prompt
from app.generation.llm import LLMClient
from app.generation.hallucination import faithfulness_check
from app.core.logger import logger


llm = LLMClient()


def rag_answer(query: str, retrieved_chunks: list, retrieval_method: str = "dense") -> Dict:
    """
    Generate an answer from retrieved chunks with faithfulness scoring.

    retrieved_chunks: [{"text": "...", "source": "...", "page_number": 1}, ...]
    retrieval_method: string describing retrieval path (e.g., "hybrid+rerank")
    """
    if not retrieved_chunks:
        return {
            "answer": "NOT FOUND in documents.",
            "confidence": 0.0,
            "is_grounded": False,
            "faithfulness_score": 0.0,
            "retrieval_method": retrieval_method,
            "sources": [],
        }

    contexts = [c["text"] for c in retrieved_chunks]
    prompt = build_rag_prompt(query, contexts)

    logger.info("Generating LLM answer...")
    answer = llm.generate(prompt)

    # Faithfulness check with NLI + similarity
    faith = faithfulness_check(answer, contexts)

    sources = sorted({c.get("source", "Unknown") for c in retrieved_chunks})

    return {
        "answer": answer,
        "confidence": faith["faithfulness_score"],
        "is_grounded": faith["is_grounded"],
        "faithfulness_score": faith["faithfulness_score"],
        "similarity_score": faith["similarity_score"],
        "nli_score": faith["nli_score"],
        "confidence_level": faith["confidence_level"],
        "retrieval_method": retrieval_method,
        "sources": sources,
    }

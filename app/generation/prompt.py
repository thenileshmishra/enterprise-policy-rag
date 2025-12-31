"""
Prompt engineering utilities for RAG grounded answering
"""

from typing import List


def build_rag_prompt(query: str, contexts: List[str]) -> str:
    """
    Builds a structured RAG prompt with clear grounding & citation rules.
    """

    joined_contexts = "\n\n---\n\n".join(contexts)

    return f"""
You are an enterprise Policy & Compliance assistant.
Answer ONLY using the provided policy documents.

If the answer is not present in given documents reply:
"NOT FOUND in policy documentation."

Rules:
- Be precise & factual
- Cite sources as: [source]
- Avoid assumptions
- If multiple policies conflict, state conflict

Context:
{joined_contexts}

User Question:
{query}

Answer:
"""

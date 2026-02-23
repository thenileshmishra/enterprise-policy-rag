"""Prompt utilities for a minimal RAG pipeline."""

from typing import List


def build_rag_prompt(query: str, contexts: List[str]) -> str:
    """Build a short grounded prompt from retrieved context chunks."""
    joined_contexts = "\n\n---\n\n".join(contexts)

    return f"""
You are a helpful assistant.
Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:
NOT FOUND in documents.

Rules:
- Be concise and factual
- Avoid assumptions

Context:
{joined_contexts}

User Question:
{query}

Answer:
"""

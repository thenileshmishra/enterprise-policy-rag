"""Prompt utilities for RAG pipeline with lost-in-the-middle reordering."""

from typing import List


def reorder_lost_in_middle(contexts: List[str]) -> List[str]:
    """
    Reorder contexts so the most relevant ones are at the start and end.

    LLMs pay more attention to the beginning and end of context windows.
    This places the best chunks where attention is strongest.
    Input: contexts sorted by relevance (best first).
    Output: best chunks at positions 0, -1, -2, ... with weaker in the middle.
    """
    if len(contexts) <= 2:
        return contexts

    reordered = []
    for i, ctx in enumerate(contexts):
        if i % 2 == 0:
            reordered.append(ctx)
        else:
            reordered.insert(len(reordered) // 2, ctx)
    return reordered


def build_rag_prompt(query: str, contexts: List[str], use_reorder: bool = True) -> str:
    """Build a grounded prompt from retrieved context chunks."""
    if use_reorder:
        contexts = reorder_lost_in_middle(contexts)

    numbered_contexts = []
    for i, ctx in enumerate(contexts, 1):
        numbered_contexts.append(f"[Source {i}]\n{ctx}")

    joined_contexts = "\n\n---\n\n".join(numbered_contexts)

    return f"""You are a helpful assistant.
Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:
NOT FOUND in documents.

Rules:
- Be concise and factual
- Avoid assumptions
- Reference source numbers when possible (e.g., [Source 1])

Context:
{joined_contexts}

User Question:
{query}

Answer:
"""

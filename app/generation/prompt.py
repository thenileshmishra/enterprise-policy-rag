"""
Prompt engineering utilities for RAG grounded answering.
Includes lost-in-the-middle mitigation for optimal context ordering.
"""

from typing import List, Dict, Optional


def reorder_contexts_lost_in_middle(contexts: List[Dict]) -> List[Dict]:
    """
    Reorder contexts to mitigate the "lost in the middle" phenomenon.

    Research shows LLMs pay more attention to content at the beginning
    and end of the context, with reduced attention to the middle.

    Strategy: Place most relevant at start, second-most at end,
    interleave remaining in decreasing importance.

    Args:
        contexts: List of context dicts, sorted by relevance (best first)

    Returns:
        Reordered contexts for optimal LLM attention
    """
    if len(contexts) <= 2:
        return contexts

    # Split into high, medium, low relevance
    n = len(contexts)
    high = contexts[:n // 3 + 1] if n >= 3 else contexts[:1]
    medium = contexts[n // 3 + 1:2 * n // 3 + 1] if n >= 3 else contexts[1:2] if n > 1 else []
    low = contexts[2 * n // 3 + 1:] if n >= 3 else contexts[2:] if n > 2 else []

    # Reorder: high -> low -> medium (best at start and end)
    reordered = []

    # Interleave: start with high relevance
    reordered.extend(high)

    # Put medium relevance in the middle (where attention is lowest)
    reordered.extend(medium)

    # Put remaining low relevance toward the end
    reordered.extend(low)

    return reordered


def format_context_with_metadata(context: Dict, index: int) -> str:
    """
    Format a single context with its metadata for the prompt.

    Args:
        context: Context dict with text and metadata
        index: Context index for citation

    Returns:
        Formatted context string
    """
    text = context.get("text", "")
    source = context.get("source", "Unknown")
    page = context.get("page_number", "?")
    section = context.get("section", "")
    heading = context.get("heading", "")

    # Build citation reference
    citation_parts = [f"Source: {source}"]
    if page:
        citation_parts.append(f"Page {page}")
    if section:
        citation_parts.append(f"Section: {section}")
    if heading and heading != section:
        citation_parts.append(f"({heading})")

    citation = " | ".join(citation_parts)

    return f"[{index + 1}] {citation}\n{text}"


def build_rag_prompt(query: str, contexts: List[str]) -> str:
    """
    Builds a structured RAG prompt with clear grounding & citation rules.
    Legacy function for backwards compatibility.
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


def build_enhanced_rag_prompt(
    query: str,
    contexts: List[Dict],
    apply_lost_in_middle: bool = True,
    max_contexts: int = 10,
    system_persona: Optional[str] = None,
) -> str:
    """
    Build an enhanced RAG prompt with metadata-rich citations and
    lost-in-the-middle mitigation.

    Args:
        query: User's question
        contexts: List of context dicts with text and metadata
        apply_lost_in_middle: Whether to reorder for attention optimization
        max_contexts: Maximum number of contexts to include
        system_persona: Optional custom system persona

    Returns:
        Complete prompt string
    """
    # Limit contexts
    contexts = contexts[:max_contexts]

    # Apply lost-in-the-middle reordering
    if apply_lost_in_middle and len(contexts) > 2:
        contexts = reorder_contexts_lost_in_middle(contexts)

    # Format contexts with metadata
    formatted_contexts = []
    for i, ctx in enumerate(contexts):
        formatted_contexts.append(format_context_with_metadata(ctx, i))

    joined_contexts = "\n\n---\n\n".join(formatted_contexts)

    # Default system persona
    if system_persona is None:
        system_persona = """You are a research assistant specialized in analyzing academic papers and documents.
Your role is to provide accurate, well-cited answers based ONLY on the provided context."""

    prompt = f"""{system_persona}

IMPORTANT RULES:
1. Answer ONLY using information from the provided context documents
2. ALWAYS cite your sources using [1], [2], etc. format matching the context numbers
3. If information is not in the context, clearly state: "This information is not available in the provided documents."
4. Be precise and factual - avoid speculation or assumptions
5. If sources provide conflicting information, acknowledge the conflict
6. Synthesize information from multiple sources when relevant

CONTEXT DOCUMENTS:
{joined_contexts}

USER QUESTION:
{query}

ANSWER (with citations):"""

    return prompt


def build_conversational_prompt(
    query: str,
    contexts: List[Dict],
    chat_history: Optional[List[Dict]] = None,
    apply_lost_in_middle: bool = True,
) -> str:
    """
    Build a conversational RAG prompt with chat history support.

    Args:
        query: Current user question
        contexts: Retrieved contexts for current query
        chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        apply_lost_in_middle: Whether to apply context reordering

    Returns:
        Complete conversational prompt
    """
    # Apply lost-in-the-middle reordering
    if apply_lost_in_middle and len(contexts) > 2:
        contexts = reorder_contexts_lost_in_middle(contexts)

    # Format contexts
    formatted_contexts = []
    for i, ctx in enumerate(contexts):
        formatted_contexts.append(format_context_with_metadata(ctx, i))

    joined_contexts = "\n\n".join(formatted_contexts)

    # Format chat history
    history_str = ""
    if chat_history:
        history_parts = []
        for msg in chat_history[-6:]:  # Keep last 6 messages for context
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_parts.append(f"{role}: {content}")
        history_str = "\n".join(history_parts)

    prompt = f"""You are a helpful research assistant. Answer questions based on the provided documents.
Always cite sources using [1], [2] format. If information isn't in the documents, say so.

DOCUMENTS:
{joined_contexts}

{"CONVERSATION HISTORY:" + chr(10) + history_str + chr(10) if history_str else ""}
CURRENT QUESTION: {query}

RESPONSE:"""

    return prompt


def build_summary_prompt(contexts: List[Dict], focus_topic: Optional[str] = None) -> str:
    """
    Build a prompt for summarizing multiple documents.

    Args:
        contexts: List of context dicts to summarize
        focus_topic: Optional topic to focus the summary on

    Returns:
        Summary prompt string
    """
    # Format contexts with sources
    formatted_contexts = []
    for i, ctx in enumerate(contexts):
        source = ctx.get("source", "Unknown")
        text = ctx.get("text", "")
        formatted_contexts.append(f"[Document {i + 1}: {source}]\n{text}")

    joined_contexts = "\n\n---\n\n".join(formatted_contexts)

    focus_instruction = ""
    if focus_topic:
        focus_instruction = f"\nFocus the summary on: {focus_topic}"

    prompt = f"""Summarize the following documents, highlighting key findings and insights.
{focus_instruction}

DOCUMENTS:
{joined_contexts}

SUMMARY (with document references):"""

    return prompt

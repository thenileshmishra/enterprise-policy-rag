"""
Citation extraction and formatting for RAG responses.
Provides structured citations linking answers to source documents.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from app.core.logger import logger


@dataclass
class Citation:
    """Represents a citation reference."""
    reference_id: str
    source: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    text_snippet: Optional[str] = None
    position_in_answer: Optional[int] = None
    confidence: float = 1.0


@dataclass
class CitationResult:
    """Result of citation extraction."""
    citations: List[Citation]
    answer_with_citations: str
    citation_map: Dict[str, Dict]
    uncited_claims: List[str] = field(default_factory=list)


class CitationExtractor:
    """
    Extracts and validates citations from RAG responses.
    Links citation references to source documents.
    """

    def __init__(self):
        # Patterns for different citation formats
        self.patterns = {
            "numbered": r'\[(\d+)\]',
            "source": r'\[(?:Source|source):\s*([^\]]+)\]',
            "page": r'(?:\(|\[)(?:Page|page|p\.?)\s*(\d+)(?:\)|\])',
            "section": r'\[(?:Section|section):\s*([^\]]+)\]',
            "inline_source": r'(?:from|in|according to)\s+["\']?([^"\',.]+\.pdf)["\']?',
        }

    def extract_citations(
        self,
        answer: str,
        contexts: List[Dict],
    ) -> CitationResult:
        """
        Extract citations from an answer and link to contexts.

        Args:
            answer: Generated answer text
            contexts: List of context dicts with metadata

        Returns:
            CitationResult with extracted and linked citations
        """
        citations = []
        citation_map = {}

        # Extract numbered citations [1], [2], etc.
        for match in re.finditer(self.patterns["numbered"], answer):
            ref_num = match.group(1)
            idx = int(ref_num) - 1

            if 0 <= idx < len(contexts):
                ctx = contexts[idx]
                citation = Citation(
                    reference_id=ref_num,
                    source=ctx.get("source", "Unknown"),
                    page_number=ctx.get("page_number"),
                    section=ctx.get("section"),
                    heading=ctx.get("heading"),
                    text_snippet=ctx.get("text", "")[:200],
                    position_in_answer=match.start(),
                )
                citations.append(citation)
                citation_map[ref_num] = {
                    "source": citation.source,
                    "page": citation.page_number,
                    "section": citation.section,
                }

        # Extract source-based citations [Source: filename.pdf]
        for match in re.finditer(self.patterns["source"], answer):
            source_name = match.group(1).strip()

            # Find matching context
            for i, ctx in enumerate(contexts):
                if source_name.lower() in ctx.get("source", "").lower():
                    citation = Citation(
                        reference_id=f"s{i + 1}",
                        source=ctx.get("source", source_name),
                        page_number=ctx.get("page_number"),
                        section=ctx.get("section"),
                        text_snippet=ctx.get("text", "")[:200],
                        position_in_answer=match.start(),
                    )
                    citations.append(citation)
                    break

        # Extract inline source mentions
        for match in re.finditer(self.patterns["inline_source"], answer, re.IGNORECASE):
            source_name = match.group(1).strip()

            for i, ctx in enumerate(contexts):
                if source_name.lower() in ctx.get("source", "").lower():
                    citation = Citation(
                        reference_id=f"i{i + 1}",
                        source=ctx.get("source", source_name),
                        page_number=ctx.get("page_number"),
                        position_in_answer=match.start(),
                        confidence=0.8,  # Lower confidence for inline mentions
                    )
                    citations.append(citation)
                    break

        # Deduplicate citations
        seen = set()
        unique_citations = []
        for cit in citations:
            key = (cit.source, cit.page_number, cit.section)
            if key not in seen:
                seen.add(key)
                unique_citations.append(cit)

        logger.info(f"Extracted {len(unique_citations)} unique citations")

        return CitationResult(
            citations=unique_citations,
            answer_with_citations=answer,
            citation_map=citation_map,
        )

    def add_citations_to_answer(
        self,
        answer: str,
        contexts: List[Dict],
        add_references_section: bool = True,
    ) -> str:
        """
        Enhance an answer with properly formatted citations.

        Args:
            answer: Original answer
            contexts: Source contexts
            add_references_section: Whether to add a references section at the end

        Returns:
            Answer with enhanced citations
        """
        # Check if answer already has citations
        has_citations = bool(re.search(self.patterns["numbered"], answer))

        if not has_citations:
            # Try to add citations where content matches
            answer = self._insert_citations(answer, contexts)

        if add_references_section:
            references = self._build_references_section(contexts)
            if references:
                answer = f"{answer}\n\n---\n{references}"

        return answer

    def _insert_citations(self, answer: str, contexts: List[Dict]) -> str:
        """Insert citation markers where answer content matches contexts."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        cited_sentences = []

        for sentence in sentences:
            best_match_idx = None
            best_match_score = 0.3  # Minimum threshold

            # Find best matching context
            for i, ctx in enumerate(contexts):
                ctx_text = ctx.get("text", "").lower()
                sentence_lower = sentence.lower()

                # Simple word overlap scoring
                sentence_words = set(sentence_lower.split())
                ctx_words = set(ctx_text.split())
                overlap = len(sentence_words & ctx_words)
                score = overlap / max(len(sentence_words), 1)

                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = i

            # Add citation if good match found
            if best_match_idx is not None and best_match_score > 0.4:
                # Insert citation before sentence-ending punctuation
                if sentence.rstrip()[-1] in '.!?':
                    sentence = f"{sentence.rstrip()[:-1]} [{best_match_idx + 1}]{sentence.rstrip()[-1]}"
                else:
                    sentence = f"{sentence} [{best_match_idx + 1}]"

            cited_sentences.append(sentence)

        return " ".join(cited_sentences)

    def _build_references_section(self, contexts: List[Dict]) -> str:
        """Build a references section from contexts."""
        if not contexts:
            return ""

        references = ["**References:**"]

        seen_sources = set()
        for i, ctx in enumerate(contexts):
            source = ctx.get("source", "Unknown")
            page = ctx.get("page_number")
            section = ctx.get("section")

            # Build reference string
            ref_parts = [f"[{i + 1}] {source}"]
            if page:
                ref_parts.append(f"Page {page}")
            if section:
                ref_parts.append(f"Section: {section}")

            ref_str = ", ".join(ref_parts)

            if ref_str not in seen_sources:
                references.append(ref_str)
                seen_sources.add(ref_str)

        return "\n".join(references) if len(references) > 1 else ""


def format_citations_for_display(
    citations: List[Citation],
    format_type: str = "markdown",
) -> str:
    """
    Format citations for display in different formats.

    Args:
        citations: List of Citation objects
        format_type: "markdown", "html", or "plain"

    Returns:
        Formatted citation string
    """
    if not citations:
        return ""

    if format_type == "markdown":
        lines = ["### Sources"]
        for cit in citations:
            parts = [f"**{cit.source}**"]
            if cit.page_number:
                parts.append(f"Page {cit.page_number}")
            if cit.section:
                parts.append(f"_{cit.section}_")
            lines.append(f"- [{cit.reference_id}] " + ", ".join(parts))
        return "\n".join(lines)

    elif format_type == "html":
        lines = ["<div class='citations'><h4>Sources</h4><ul>"]
        for cit in citations:
            parts = [f"<strong>{cit.source}</strong>"]
            if cit.page_number:
                parts.append(f"Page {cit.page_number}")
            if cit.section:
                parts.append(f"<em>{cit.section}</em>")
            lines.append(f"<li>[{cit.reference_id}] " + ", ".join(parts) + "</li>")
        lines.append("</ul></div>")
        return "\n".join(lines)

    else:  # plain
        lines = ["Sources:"]
        for cit in citations:
            parts = [cit.source]
            if cit.page_number:
                parts.append(f"Page {cit.page_number}")
            if cit.section:
                parts.append(cit.section)
            lines.append(f"[{cit.reference_id}] " + ", ".join(parts))
        return "\n".join(lines)


def extract_and_format_citations(
    answer: str,
    contexts: List[Dict],
) -> Dict:
    """
    Convenience function to extract citations and format output.

    Args:
        answer: Generated answer
        contexts: Source contexts

    Returns:
        Dict with answer, citations, and formatted references
    """
    extractor = CitationExtractor()
    result = extractor.extract_citations(answer, contexts)

    return {
        "answer": result.answer_with_citations,
        "citations": [
            {
                "id": cit.reference_id,
                "source": cit.source,
                "page": cit.page_number,
                "section": cit.section,
                "heading": cit.heading,
                "snippet": cit.text_snippet,
            }
            for cit in result.citations
        ],
        "citation_map": result.citation_map,
        "references_markdown": format_citations_for_display(result.citations, "markdown"),
        "references_html": format_citations_for_display(result.citations, "html"),
    }

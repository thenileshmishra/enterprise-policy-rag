"""
Grounding Agent for AutoGen-based RAG pipeline.
Verifies answer faithfulness and extracts citations.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import re
from app.core.logger import logger


@dataclass
class GroundingConfig:
    """Configuration for grounding agent."""
    faithfulness_threshold: float = 0.7
    require_citations: bool = True
    min_citations: int = 1
    check_hallucination: bool = True


@dataclass
class GroundingResult:
    """Result of grounding verification."""
    is_grounded: bool
    faithfulness_score: float
    citations: List[Dict]
    issues: List[str]
    improved_answer: Optional[str] = None


class GroundingAgent:
    """
    Agent specialized in verifying answer faithfulness.
    Ensures responses are grounded in provided contexts.
    """

    def __init__(self, config: Optional[GroundingConfig] = None):
        """
        Initialize grounding agent.

        Args:
            config: Optional grounding configuration
        """
        self.config = config or GroundingConfig()
        logger.info("GroundingAgent initialized")

    def extract_citations(self, answer: str) -> List[Dict]:
        """
        Extract citation references from an answer.

        Looks for patterns like [1], [2], [Source: paper.pdf], etc.

        Args:
            answer: Generated answer text

        Returns:
            List of citation dicts
        """
        citations = []

        # Pattern 1: Numbered citations [1], [2], etc.
        numbered_pattern = r'\[(\d+)\]'
        for match in re.finditer(numbered_pattern, answer):
            citations.append({
                "type": "numbered",
                "reference": match.group(1),
                "position": match.start(),
            })

        # Pattern 2: Source citations [Source: filename]
        source_pattern = r'\[(?:Source|source):\s*([^\]]+)\]'
        for match in re.finditer(source_pattern, answer):
            citations.append({
                "type": "source",
                "reference": match.group(1).strip(),
                "position": match.start(),
            })

        # Pattern 3: Page citations (Page X) or [Page X]
        page_pattern = r'(?:\[|\()(?:Page|page|p\.?)\s*(\d+)(?:\]|\))'
        for match in re.finditer(page_pattern, answer):
            citations.append({
                "type": "page",
                "reference": match.group(1),
                "position": match.start(),
            })

        logger.info(f"Extracted {len(citations)} citations from answer")
        return citations

    def compute_faithfulness(
        self,
        answer: str,
        contexts: List[Dict],
    ) -> float:
        """
        Compute faithfulness score between answer and contexts.

        Uses word overlap and semantic similarity heuristics.

        Args:
            answer: Generated answer
            contexts: Source contexts

        Returns:
            Faithfulness score between 0 and 1
        """
        if not answer or not contexts:
            return 0.0

        # Combine all context texts
        context_text = " ".join([c.get("text", "") for c in contexts]).lower()
        answer_lower = answer.lower()

        # Word overlap approach
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())

        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "being", "have", "has", "had", "do", "does", "did", "will",
                    "would", "could", "should", "may", "might", "must", "shall",
                    "can", "need", "to", "of", "in", "for", "on", "with", "at",
                    "by", "from", "as", "into", "through", "during", "before",
                    "after", "above", "below", "between", "under", "again",
                    "further", "then", "once", "and", "but", "or", "nor", "so",
                    "yet", "both", "each", "few", "more", "most", "other", "some",
                    "such", "no", "not", "only", "own", "same", "than", "too",
                    "very", "just", "also", "now", "this", "that", "these", "those"}

        answer_content_words = answer_words - stopwords
        context_content_words = context_words - stopwords

        if not answer_content_words:
            return 0.0

        # Calculate overlap
        overlap = answer_content_words & context_content_words
        coverage = len(overlap) / len(answer_content_words)

        # Check for key phrases from context in answer
        context_phrases = self._extract_key_phrases(context_text)
        answer_text = answer_lower
        phrase_matches = sum(1 for phrase in context_phrases if phrase in answer_text)
        phrase_score = phrase_matches / max(len(context_phrases), 1) if context_phrases else 0

        # Combined score
        faithfulness_score = 0.6 * coverage + 0.4 * phrase_score

        logger.info(f"Faithfulness score: {faithfulness_score:.3f} (coverage: {coverage:.3f}, phrases: {phrase_score:.3f})")
        return min(faithfulness_score, 1.0)

    def _extract_key_phrases(self, text: str, min_length: int = 3) -> List[str]:
        """Extract key phrases from text (simple n-gram approach)."""
        words = text.split()
        phrases = []

        # Extract 2-grams and 3-grams
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if len(phrase) >= min_length * n:
                    phrases.append(phrase)

        return phrases[:50]  # Limit to top 50 phrases

    def verify_grounding(
        self,
        answer: str,
        contexts: List[Dict],
        query: Optional[str] = None,
    ) -> GroundingResult:
        """
        Verify that an answer is properly grounded in the contexts.

        Args:
            answer: Generated answer
            contexts: Source contexts used for generation
            query: Optional original query

        Returns:
            GroundingResult with verification details
        """
        issues = []

        # Extract citations
        citations = self.extract_citations(answer)

        # Check citation requirements
        if self.config.require_citations:
            if len(citations) < self.config.min_citations:
                issues.append(
                    f"Insufficient citations: found {len(citations)}, "
                    f"minimum required: {self.config.min_citations}"
                )

        # Compute faithfulness
        faithfulness_score = self.compute_faithfulness(answer, contexts)

        # Check faithfulness threshold
        is_grounded = faithfulness_score >= self.config.faithfulness_threshold
        if not is_grounded:
            issues.append(
                f"Low faithfulness score: {faithfulness_score:.3f} "
                f"(threshold: {self.config.faithfulness_threshold})"
            )

        # Check for potential hallucination indicators
        if self.config.check_hallucination:
            hallucination_indicators = self._check_hallucination_indicators(answer, contexts)
            if hallucination_indicators:
                issues.extend(hallucination_indicators)

        # Map citations to contexts
        enriched_citations = self._enrich_citations(citations, contexts)

        result = GroundingResult(
            is_grounded=is_grounded and len(issues) == 0,
            faithfulness_score=faithfulness_score,
            citations=enriched_citations,
            issues=issues,
        )

        logger.info(f"Grounding verification: grounded={result.is_grounded}, score={faithfulness_score:.3f}")
        return result

    def _check_hallucination_indicators(
        self,
        answer: str,
        contexts: List[Dict],
    ) -> List[str]:
        """Check for common hallucination indicators."""
        indicators = []
        answer_lower = answer.lower()

        # Check for unsupported claims about specific numbers/dates
        number_pattern = r'\b(?:in\s+)?(\d{4})\b'
        context_text = " ".join([c.get("text", "") for c in contexts])

        for match in re.finditer(number_pattern, answer):
            year = match.group(1)
            if year not in context_text:
                # Could be a hallucinated year
                if 1900 <= int(year) <= 2030:
                    indicators.append(f"Potentially unsupported year reference: {year}")

        # Check for phrases that often indicate speculation
        speculation_phrases = [
            "i believe", "i think", "probably", "likely",
            "it seems", "it appears", "presumably", "supposedly"
        ]
        for phrase in speculation_phrases:
            if phrase in answer_lower:
                indicators.append(f"Speculation indicator found: '{phrase}'")

        return indicators

    def _enrich_citations(
        self,
        citations: List[Dict],
        contexts: List[Dict],
    ) -> List[Dict]:
        """Enrich citations with context metadata."""
        enriched = []

        for citation in citations:
            enriched_citation = citation.copy()

            # Try to match numbered citations to contexts
            if citation["type"] == "numbered":
                try:
                    idx = int(citation["reference"]) - 1
                    if 0 <= idx < len(contexts):
                        ctx = contexts[idx]
                        enriched_citation["source"] = ctx.get("source", "Unknown")
                        enriched_citation["page_number"] = ctx.get("page_number")
                        enriched_citation["section"] = ctx.get("section")
                except (ValueError, IndexError):
                    pass

            enriched.append(enriched_citation)

        return enriched


def create_grounding_agent(
    faithfulness_threshold: float = 0.7,
    require_citations: bool = True,
) -> GroundingAgent:
    """
    Factory function to create a configured grounding agent.

    Args:
        faithfulness_threshold: Minimum faithfulness score
        require_citations: Whether to require citations

    Returns:
        Configured GroundingAgent
    """
    config = GroundingConfig(
        faithfulness_threshold=faithfulness_threshold,
        require_citations=require_citations,
    )

    return GroundingAgent(config=config)

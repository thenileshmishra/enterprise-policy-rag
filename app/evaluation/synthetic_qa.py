"""
Synthetic Q&A generation for RAG evaluation.
Generates question-answer pairs from documents for testing.
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from app.core.logger import logger
from app.generation.llm import generate_answer


@dataclass
class QAPair:
    """Generated question-answer pair."""
    question: str
    answer: str
    context: str
    source: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    difficulty: str = "medium"
    question_type: str = "factual"


class SyntheticQAGenerator:
    """
    Generates synthetic Q&A pairs from document chunks.
    Uses LLM to create diverse, realistic questions.
    """

    def __init__(
        self,
        llm_generator=None,
        questions_per_chunk: int = 2,
        question_types: Optional[List[str]] = None,
    ):
        """
        Initialize synthetic QA generator.

        Args:
            llm_generator: Optional custom LLM generator function
            questions_per_chunk: Number of questions to generate per chunk
            question_types: Types of questions to generate
        """
        self.llm_generator = llm_generator or generate_answer
        self.questions_per_chunk = questions_per_chunk
        self.question_types = question_types or [
            "factual",
            "conceptual",
            "comparative",
            "procedural",
        ]

        logger.info("SyntheticQAGenerator initialized")

    def _build_generation_prompt(
        self,
        chunk: Dict,
        question_type: str,
        num_questions: int = 1,
    ) -> str:
        """Build prompt for Q&A generation."""
        text = chunk.get("text", "")
        source = chunk.get("source", "document")

        prompts_by_type = {
            "factual": """Generate {n} factual question(s) that can be answered directly from this text.
Questions should ask about specific facts, numbers, names, or definitions.""",
            "conceptual": """Generate {n} conceptual question(s) about the main ideas in this text.
Questions should test understanding of concepts, theories, or principles.""",
            "comparative": """Generate {n} question(s) that compare or contrast elements in this text.
Questions might ask about differences, similarities, or relationships.""",
            "procedural": """Generate {n} question(s) about processes or procedures described.
Questions should ask about steps, methods, or how things work.""",
        }

        type_instruction = prompts_by_type.get(question_type, prompts_by_type["factual"])
        type_instruction = type_instruction.format(n=num_questions)

        prompt = f"""Based on the following text from "{source}", generate questions and answers.

TEXT:
{text}

INSTRUCTIONS:
{type_instruction}

For each question:
1. Make sure the answer can be found in the provided text
2. Keep questions clear and specific
3. Provide complete answers based only on the text

Output format (JSON array):
[
  {{"question": "...", "answer": "..."}}
]

Generate the Q&A pairs:"""

        return prompt

    def generate_from_chunk(
        self,
        chunk: Dict,
        num_questions: int = 2,
        question_types: Optional[List[str]] = None,
    ) -> List[QAPair]:
        """
        Generate Q&A pairs from a single chunk.

        Args:
            chunk: Document chunk dictionary
            num_questions: Number of questions to generate
            question_types: Types of questions to generate

        Returns:
            List of QAPair objects
        """
        types = question_types or self.question_types
        qa_pairs = []

        # Distribute questions across types
        type_distribution = []
        for i in range(num_questions):
            type_distribution.append(types[i % len(types)])

        for q_type in set(type_distribution):
            count = type_distribution.count(q_type)
            prompt = self._build_generation_prompt(chunk, q_type, count)

            try:
                response = self.llm_generator(prompt)

                # Parse JSON response
                pairs = self._parse_qa_response(response)

                for pair in pairs[:count]:
                    qa_pairs.append(QAPair(
                        question=pair.get("question", ""),
                        answer=pair.get("answer", ""),
                        context=chunk.get("text", ""),
                        source=chunk.get("source", "unknown"),
                        page_number=chunk.get("page_number"),
                        section=chunk.get("section"),
                        difficulty="medium",
                        question_type=q_type,
                    ))

            except Exception as e:
                logger.warning(f"Failed to generate Q&A for chunk: {e}")

        return qa_pairs

    def _parse_qa_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract Q&A pairs."""
        # Try to find JSON in response
        import re

        # Look for JSON array
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse question/answer patterns
        pairs = []
        q_pattern = r'(?:Q(?:uestion)?[:\s]*|^\d+\.\s*)(.+?)(?=A(?:nswer)?[:\s]|$)'
        a_pattern = r'A(?:nswer)?[:\s]*(.+?)(?=Q(?:uestion)?[:\s]|\d+\.|$)'

        questions = re.findall(q_pattern, response, re.MULTILINE | re.IGNORECASE)
        answers = re.findall(a_pattern, response, re.MULTILINE | re.IGNORECASE)

        for q, a in zip(questions, answers):
            pairs.append({"question": q.strip(), "answer": a.strip()})

        return pairs

    def generate_from_chunks(
        self,
        chunks: List[Dict],
        total_questions: int = 50,
        balance_sources: bool = True,
    ) -> List[QAPair]:
        """
        Generate Q&A pairs from multiple chunks.

        Args:
            chunks: List of document chunks
            total_questions: Target total number of questions
            balance_sources: Whether to balance across different sources

        Returns:
            List of QAPair objects
        """
        if not chunks:
            return []

        qa_pairs = []

        if balance_sources:
            # Group chunks by source
            source_groups = {}
            for chunk in chunks:
                source = chunk.get("source", "unknown")
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(chunk)

            # Distribute questions across sources
            questions_per_source = max(1, total_questions // len(source_groups))

            for source, source_chunks in source_groups.items():
                # Sample chunks from this source
                sampled = random.sample(
                    source_chunks,
                    min(len(source_chunks), questions_per_source)
                )

                for chunk in sampled:
                    pairs = self.generate_from_chunk(chunk, num_questions=1)
                    qa_pairs.extend(pairs)

                    if len(qa_pairs) >= total_questions:
                        break

                if len(qa_pairs) >= total_questions:
                    break

        else:
            # Random sampling across all chunks
            questions_per_chunk = max(1, total_questions // len(chunks))
            sampled_chunks = random.sample(chunks, min(len(chunks), total_questions))

            for chunk in sampled_chunks:
                pairs = self.generate_from_chunk(chunk, num_questions=1)
                qa_pairs.extend(pairs)

                if len(qa_pairs) >= total_questions:
                    break

        logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {len(chunks)} chunks")
        return qa_pairs[:total_questions]

    def generate_evaluation_dataset(
        self,
        chunks: List[Dict],
        num_samples: int = 50,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate a complete evaluation dataset.

        Args:
            chunks: Document chunks
            num_samples: Number of Q&A samples to generate
            output_path: Optional path to save the dataset

        Returns:
            Dataset dictionary
        """
        qa_pairs = self.generate_from_chunks(chunks, total_questions=num_samples)

        dataset = {
            "metadata": {
                "num_samples": len(qa_pairs),
                "num_sources": len(set(p.source for p in qa_pairs)),
                "question_types": list(set(p.question_type for p in qa_pairs)),
            },
            "samples": [
                {
                    "question": p.question,
                    "ground_truth": p.answer,
                    "context": p.context,
                    "source": p.source,
                    "page_number": p.page_number,
                    "section": p.section,
                    "question_type": p.question_type,
                }
                for p in qa_pairs
            ],
        }

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(dataset, f, indent=2)
            logger.info(f"Saved evaluation dataset to {output_path}")

        return dataset


def generate_simple_questions(chunks: List[Dict], num_questions: int = 10) -> List[Dict]:
    """
    Generate simple template-based questions without LLM.
    Useful for quick testing or when LLM is unavailable.

    Args:
        chunks: Document chunks
        num_questions: Number of questions to generate

    Returns:
        List of Q&A dictionaries
    """
    templates = [
        "What does this section say about {topic}?",
        "According to the document, what is {topic}?",
        "Can you explain {topic} based on the text?",
        "What are the key points about {topic}?",
        "Summarize the information about {topic}.",
    ]

    questions = []
    sampled_chunks = random.sample(chunks, min(len(chunks), num_questions))

    for chunk in sampled_chunks:
        text = chunk.get("text", "")

        # Extract potential topics (simple: first significant words)
        words = [w for w in text.split()[:50] if len(w) > 4]
        if words:
            topic = random.choice(words[:10])
            template = random.choice(templates)

            questions.append({
                "question": template.format(topic=topic),
                "context": text,
                "source": chunk.get("source", "unknown"),
                "ground_truth": text[:200],  # Use chunk text as approximate answer
            })

    logger.info(f"Generated {len(questions)} simple template questions")
    return questions


def load_evaluation_dataset(path: str) -> Dict:
    """Load an evaluation dataset from file."""
    with open(path, "r") as f:
        return json.load(f)

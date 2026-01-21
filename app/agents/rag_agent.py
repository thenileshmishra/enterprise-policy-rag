"""
RAG Orchestrator Agent for AutoGen-based pipeline.
Coordinates retrieval, generation, and grounding agents.
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from app.core.logger import logger
from app.agents.retrieval_agent import RetrievalAgent, RetrievalConfig, create_retrieval_agent
from app.agents.grounding_agent import GroundingAgent, GroundingConfig, create_grounding_agent
from app.generation.prompt import build_enhanced_rag_prompt, build_conversational_prompt
from app.generation.llm import generate_answer


@dataclass
class RAGConfig:
    """Configuration for RAG orchestrator."""
    retrieval_k: int = 10
    rerank_k: int = 5
    use_reranking: bool = True
    dense_weight: float = 0.5
    faithfulness_threshold: float = 0.7
    require_citations: bool = True
    apply_lost_in_middle: bool = True
    max_retries: int = 2


@dataclass
class RAGResponse:
    """Complete RAG response with all metadata."""
    answer: str
    contexts: List[Dict]
    citations: List[Dict]
    faithfulness_score: float
    is_grounded: bool
    sources: List[str]
    query: str
    metadata: Dict = field(default_factory=dict)


class RAGOrchestrator:
    """
    Orchestrates the complete RAG pipeline using specialized agents.
    Coordinates retrieval, generation, and grounding in a coherent flow.
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[RAGConfig] = None,
        llm_generator: Optional[Callable] = None,
    ):
        """
        Initialize RAG orchestrator.

        Args:
            session_id: Session ID for isolated operation
            config: Optional RAG configuration
            llm_generator: Optional custom LLM generation function
        """
        self.session_id = session_id
        self.config = config or RAGConfig()

        # Initialize agents
        self.retrieval_agent = create_retrieval_agent(
            session_id=session_id,
            k=self.config.retrieval_k,
            use_reranking=self.config.use_reranking,
            dense_weight=self.config.dense_weight,
        )

        self.grounding_agent = create_grounding_agent(
            faithfulness_threshold=self.config.faithfulness_threshold,
            require_citations=self.config.require_citations,
        )

        self.llm_generator = llm_generator or generate_answer
        self.chat_history: List[Dict] = []

        logger.info(f"RAGOrchestrator initialized for session: {session_id}")

    def process_query(
        self,
        query: str,
        use_chat_history: bool = False,
        custom_prompt_builder: Optional[Callable] = None,
    ) -> RAGResponse:
        """
        Process a query through the complete RAG pipeline.

        Args:
            query: User query
            use_chat_history: Whether to include chat history
            custom_prompt_builder: Optional custom prompt builder function

        Returns:
            RAGResponse with answer and metadata
        """
        logger.info(f"Processing query: {query[:50]}...")

        # Step 1: Retrieval
        retrieval_result = self.retrieval_agent.retrieve_with_metadata(
            query=query,
            k=self.config.retrieval_k,
        )
        contexts = retrieval_result["results"]

        if not contexts:
            return RAGResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                contexts=[],
                citations=[],
                faithfulness_score=0.0,
                is_grounded=False,
                sources=[],
                query=query,
                metadata={"error": "No relevant documents found"},
            )

        # Step 2: Build prompt
        if custom_prompt_builder:
            prompt = custom_prompt_builder(query, contexts)
        elif use_chat_history and self.chat_history:
            prompt = build_conversational_prompt(
                query=query,
                contexts=contexts,
                chat_history=self.chat_history,
                apply_lost_in_middle=self.config.apply_lost_in_middle,
            )
        else:
            prompt = build_enhanced_rag_prompt(
                query=query,
                contexts=contexts,
                apply_lost_in_middle=self.config.apply_lost_in_middle,
            )

        # Step 3: Generate answer
        answer = self.llm_generator(prompt)

        # Step 4: Verify grounding
        grounding_result = self.grounding_agent.verify_grounding(
            answer=answer,
            contexts=contexts,
            query=query,
        )

        # Step 5: Handle low faithfulness with retry
        retries = 0
        while not grounding_result.is_grounded and retries < self.config.max_retries:
            logger.info(f"Low grounding score, retrying (attempt {retries + 1})")

            # Add instruction to be more grounded
            enhanced_prompt = prompt + "\n\nIMPORTANT: Base your answer strictly on the provided documents. Include specific citations."
            answer = self.llm_generator(enhanced_prompt)

            grounding_result = self.grounding_agent.verify_grounding(
                answer=answer,
                contexts=contexts,
                query=query,
            )
            retries += 1

        # Update chat history
        if use_chat_history:
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            # Keep history manageable
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

        # Extract unique sources
        sources = list(set(ctx.get("source", "Unknown") for ctx in contexts))

        response = RAGResponse(
            answer=answer,
            contexts=contexts,
            citations=grounding_result.citations,
            faithfulness_score=grounding_result.faithfulness_score,
            is_grounded=grounding_result.is_grounded,
            sources=sources,
            query=query,
            metadata={
                "retrieval_k": len(contexts),
                "grounding_issues": grounding_result.issues,
                "retries": retries,
            },
        )

        logger.info(
            f"Query processed: grounded={response.is_grounded}, "
            f"faithfulness={response.faithfulness_score:.3f}"
        )
        return response

    def add_documents(self, chunks: List[Dict]) -> int:
        """Add documents to the retrieval index."""
        return self.retrieval_agent.add_documents(chunks)

    def clear_chat_history(self):
        """Clear the conversation history."""
        self.chat_history = []
        logger.info("Chat history cleared")

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "session_id": self.session_id,
            "retrieval_stats": self.retrieval_agent.get_index_stats(),
            "chat_history_length": len(self.chat_history),
            "config": {
                "retrieval_k": self.config.retrieval_k,
                "faithfulness_threshold": self.config.faithfulness_threshold,
                "use_reranking": self.config.use_reranking,
            },
        }


# Session orchestrators cache
_session_orchestrators: Dict[str, RAGOrchestrator] = {}


def get_rag_orchestrator(
    session_id: str,
    config: Optional[RAGConfig] = None,
) -> RAGOrchestrator:
    """
    Get or create a RAG orchestrator for a session.

    Args:
        session_id: Session ID
        config: Optional configuration

    Returns:
        RAGOrchestrator instance
    """
    if session_id not in _session_orchestrators:
        _session_orchestrators[session_id] = RAGOrchestrator(
            session_id=session_id,
            config=config,
        )

    return _session_orchestrators[session_id]


def create_rag_pipeline(
    session_id: str,
    retrieval_k: int = 10,
    use_reranking: bool = True,
    faithfulness_threshold: float = 0.7,
) -> RAGOrchestrator:
    """
    Factory function to create a configured RAG pipeline.

    Args:
        session_id: Session ID
        retrieval_k: Number of documents to retrieve
        use_reranking: Whether to use cross-encoder reranking
        faithfulness_threshold: Minimum faithfulness score

    Returns:
        Configured RAGOrchestrator
    """
    config = RAGConfig(
        retrieval_k=retrieval_k,
        use_reranking=use_reranking,
        faithfulness_threshold=faithfulness_threshold,
    )

    return RAGOrchestrator(session_id=session_id, config=config)


def process_rag_query(
    session_id: str,
    query: str,
    use_chat_history: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to process a RAG query.

    Args:
        session_id: Session ID
        query: User query
        use_chat_history: Whether to use chat history

    Returns:
        Dict with answer and metadata
    """
    orchestrator = get_rag_orchestrator(session_id)
    response = orchestrator.process_query(query, use_chat_history=use_chat_history)

    return {
        "answer": response.answer,
        "sources": response.sources,
        "citations": response.citations,
        "faithfulness_score": response.faithfulness_score,
        "is_grounded": response.is_grounded,
        "contexts": [
            {
                "text": ctx.get("text", "")[:200] + "...",
                "source": ctx.get("source"),
                "page": ctx.get("page_number"),
                "section": ctx.get("section"),
            }
            for ctx in response.contexts
        ],
    }

"""
MLflow experiment tracking for RAG evaluation.
Logs metrics, parameters, and artifacts for experiment analysis.
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from app.core.logger import logger


class RAGExperimentTracker:
    """
    MLflow-based experiment tracker for RAG system evaluation.
    Tracks retrieval metrics, RAGAS metrics, latency, and configurations.
    """

    def __init__(
        self,
        experiment_name: str = "rag-evaluation",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (or from MLFLOW_TRACKING_URI env)
            artifact_location: Location for storing artifacts
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            "mlruns"  # Local directory by default
        )

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
            self.experiment_id = self.experiment.experiment_id
            logger.info(f"MLflow experiment '{experiment_name}' (ID: {self.experiment_id})")
        except Exception as e:
            logger.error(f"Failed to create MLflow experiment: {e}")
            self.experiment_id = None

        self.client = MlflowClient()
        self.active_run = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run

        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
        )

        logger.info(f"Started MLflow run: {run_name} (ID: {self.active_run.info.run_id})")
        return self.active_run.info.run_id

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
            self.active_run = None

    def log_rag_config(self, config: Dict[str, Any]):
        """
        Log RAG system configuration as parameters.

        Args:
            config: Configuration dictionary
        """
        params_to_log = {
            "retrieval_k": config.get("retrieval_k", 10),
            "rerank_k": config.get("rerank_k", 5),
            "dense_weight": config.get("dense_weight", 0.5),
            "sparse_weight": config.get("sparse_weight", 0.5),
            "use_reranking": config.get("use_reranking", True),
            "chunk_size": config.get("chunk_size", 800),
            "chunk_overlap": config.get("chunk_overlap", 150),
            "embedding_model": config.get("embedding_model", "all-MiniLM-L6-v2"),
            "llm_model": config.get("llm_model", "llama-3.3-70b"),
            "temperature": config.get("temperature", 0.2),
        }

        for key, value in params_to_log.items():
            mlflow.log_param(key, value)

        logger.info(f"Logged {len(params_to_log)} RAG config parameters")

    def log_ragas_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log RAGAS evaluation metrics.

        Args:
            metrics: Dict with faithfulness, answer_relevance, etc.
            step: Optional step number for time series
        """
        metric_mapping = {
            "faithfulness": "ragas/faithfulness",
            "answer_relevance": "ragas/answer_relevance",
            "context_precision": "ragas/context_precision",
            "context_recall": "ragas/context_recall",
            "overall_score": "ragas/overall_score",
        }

        for key, mlflow_key in metric_mapping.items():
            if key in metrics:
                mlflow.log_metric(mlflow_key, metrics[key], step=step)

        logger.info(f"Logged RAGAS metrics: overall={metrics.get('overall_score', 'N/A')}")

    def log_retrieval_metrics(
        self,
        precision: float,
        recall: float,
        mrr: float,
        ndcg: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Log retrieval quality metrics.

        Args:
            precision: Precision@K
            recall: Recall@K
            mrr: Mean Reciprocal Rank
            ndcg: Optional NDCG@K
            step: Optional step number
        """
        mlflow.log_metric("retrieval/precision", precision, step=step)
        mlflow.log_metric("retrieval/recall", recall, step=step)
        mlflow.log_metric("retrieval/mrr", mrr, step=step)
        if ndcg is not None:
            mlflow.log_metric("retrieval/ndcg", ndcg, step=step)

        logger.info(f"Logged retrieval metrics: P={precision:.3f}, R={recall:.3f}, MRR={mrr:.3f}")

    def log_latency_metrics(
        self,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_latency_ms: float,
        step: Optional[int] = None,
    ):
        """
        Log latency metrics.

        Args:
            retrieval_latency_ms: Retrieval time in milliseconds
            generation_latency_ms: LLM generation time
            total_latency_ms: End-to-end latency
            step: Optional step number
        """
        mlflow.log_metric("latency/retrieval_ms", retrieval_latency_ms, step=step)
        mlflow.log_metric("latency/generation_ms", generation_latency_ms, step=step)
        mlflow.log_metric("latency/total_ms", total_latency_ms, step=step)

    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        artifact_name: str = "evaluation_results.json",
    ):
        """
        Log full evaluation results as an artifact.

        Args:
            results: Full evaluation results dictionary
            artifact_name: Name for the artifact file
        """
        # Save to temporary file and log as artifact
        artifact_path = Path(f"/tmp/{artifact_name}")
        with open(artifact_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        mlflow.log_artifact(str(artifact_path))
        logger.info(f"Logged evaluation results artifact: {artifact_name}")

    def log_qa_samples(
        self,
        samples: List[Dict],
        artifact_name: str = "qa_samples.json",
    ):
        """
        Log Q&A samples used for evaluation.

        Args:
            samples: List of Q&A sample dictionaries
            artifact_name: Name for the artifact file
        """
        artifact_path = Path(f"/tmp/{artifact_name}")
        with open(artifact_path, "w") as f:
            json.dump(samples, f, indent=2)

        mlflow.log_artifact(str(artifact_path))
        mlflow.log_metric("eval/num_samples", len(samples))

    def track_evaluation_run(
        self,
        config: Dict[str, Any],
        ragas_metrics: Dict[str, float],
        retrieval_metrics: Optional[Dict[str, float]] = None,
        latency_metrics: Optional[Dict[str, float]] = None,
        samples: Optional[List[Dict]] = None,
        run_name: Optional[str] = None,
    ) -> str:
        """
        Convenience method to track a complete evaluation run.

        Args:
            config: RAG configuration
            ragas_metrics: RAGAS evaluation metrics
            retrieval_metrics: Optional retrieval metrics
            latency_metrics: Optional latency metrics
            samples: Optional Q&A samples
            run_name: Optional run name

        Returns:
            Run ID
        """
        run_id = self.start_run(run_name=run_name)

        try:
            # Log configuration
            self.log_rag_config(config)

            # Log RAGAS metrics
            self.log_ragas_metrics(ragas_metrics)

            # Log retrieval metrics if provided
            if retrieval_metrics:
                self.log_retrieval_metrics(
                    precision=retrieval_metrics.get("precision", 0),
                    recall=retrieval_metrics.get("recall", 0),
                    mrr=retrieval_metrics.get("mrr", 0),
                    ndcg=retrieval_metrics.get("ndcg"),
                )

            # Log latency metrics if provided
            if latency_metrics:
                self.log_latency_metrics(
                    retrieval_latency_ms=latency_metrics.get("retrieval_ms", 0),
                    generation_latency_ms=latency_metrics.get("generation_ms", 0),
                    total_latency_ms=latency_metrics.get("total_ms", 0),
                )

            # Log samples if provided
            if samples:
                self.log_qa_samples(samples)

            self.end_run(status="FINISHED")

        except Exception as e:
            logger.error(f"Error during evaluation tracking: {e}")
            self.end_run(status="FAILED")
            raise

        return run_id

    def get_best_run(
        self,
        metric: str = "ragas/overall_score",
        maximize: bool = True,
    ) -> Optional[Dict]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False)

        Returns:
            Dict with run info and metrics
        """
        try:
            order = "DESC" if maximize else "ASC"
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.`{metric}` {order}"],
                max_results=1,
            )

            if runs:
                best_run = runs[0]
                return {
                    "run_id": best_run.info.run_id,
                    "run_name": best_run.info.run_name,
                    "metrics": best_run.data.metrics,
                    "params": best_run.data.params,
                }

        except Exception as e:
            logger.error(f"Failed to get best run: {e}")

        return None

    def compare_runs(
        self,
        run_ids: List[str],
    ) -> List[Dict]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            List of run info dictionaries
        """
        comparisons = []

        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                comparisons.append({
                    "run_id": run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                })
            except Exception as e:
                logger.warning(f"Failed to get run {run_id}: {e}")

        return comparisons


# Global tracker instance
_tracker: Optional[RAGExperimentTracker] = None


def get_tracker(experiment_name: str = "rag-evaluation") -> RAGExperimentTracker:
    """Get or create the global experiment tracker."""
    global _tracker
    if _tracker is None or _tracker.experiment_name != experiment_name:
        _tracker = RAGExperimentTracker(experiment_name=experiment_name)
    return _tracker


def track_evaluation(
    config: Dict[str, Any],
    metrics: Dict[str, float],
    run_name: Optional[str] = None,
) -> str:
    """
    Convenience function to track an evaluation.

    Args:
        config: RAG configuration
        metrics: Evaluation metrics (RAGAS format)
        run_name: Optional run name

    Returns:
        Run ID
    """
    tracker = get_tracker()
    return tracker.track_evaluation_run(
        config=config,
        ragas_metrics=metrics,
        run_name=run_name,
    )

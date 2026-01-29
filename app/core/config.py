"""
Configuration settings for the RAG system.
Supports environment variables and .env file loading.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "Research Paper RAG Assistant"
    ENVIRONMENT: str = "dev"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # API Keys
    HF_API_KEY: str = ""
    LLM_API_KEY: str = ""
    LLM_API_URL: str = ""

    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"

    # S3 Storage
    S3_BUCKET: Optional[str] = None
    S3_PDF_BUCKET: Optional[str] = None
    S3_INDEX_KEY: str = "index/faiss.index"
    S3_META_KEY: str = "index/metadata.json"
    S3_ENDPOINT: Optional[str] = None  # For S3-compatible services

    # SageMaker LLM
    SAGEMAKER_ENDPOINT_NAME: Optional[str] = None
    SAGEMAKER_REGION: str = "us-east-1"

    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    USE_UNSTRUCTURED: bool = True

    # Retrieval
    RETRIEVAL_K: int = 10
    RERANK_K: int = 5
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5
    USE_RERANKING: bool = True
    USE_HYBRID_SEARCH: bool = True

    # Reranker Model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # NLI Model for Faithfulness
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"

    # LLM Generation
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # Faithfulness/Grounding
    FAITHFULNESS_THRESHOLD: float = 0.55
    REQUIRE_CITATIONS: bool = True

    # Session Management
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_SESSIONS: int = 100

    # MLflow
    MLFLOW_TRACKING_URI: str = "mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "rag-evaluation"

    # RAGAS Evaluation
    EVAL_DATASET_PATH: Optional[str] = None
    EVAL_OUTPUT_PATH: str = "data/eval/results"

    # Use Pydantic v2 style config and ignore unexpected env vars
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
    )


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


class ProductionSettings(Settings):
    """Production-specific settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    USE_RERANKING: bool = True
    USE_HYBRID_SEARCH: bool = True


class TestSettings(Settings):
    """Test-specific settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENVIRONMENT: str = "test"
    USE_RERANKING: bool = False  # Faster tests
    USE_HYBRID_SEARCH: bool = False


def get_settings() -> Settings:
    """Get settings based on environment."""
    import os
    env = os.getenv("ENVIRONMENT", "dev").lower()

    if env == "production" or env == "prod":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


# Default settings instance
settings = Settings()


# Convenience access to common settings
def get_embedding_model() -> str:
    return settings.EMBEDDING_MODEL


def get_llm_config() -> dict:
    return {
        "api_key": settings.LLM_API_KEY,
        "api_url": settings.LLM_API_URL,
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
    }


def get_retrieval_config() -> dict:
    return {
        "retrieval_k": settings.RETRIEVAL_K,
        "rerank_k": settings.RERANK_K,
        "dense_weight": settings.DENSE_WEIGHT,
        "sparse_weight": settings.SPARSE_WEIGHT,
        "use_reranking": settings.USE_RERANKING,
        "use_hybrid": settings.USE_HYBRID_SEARCH,
    }


def get_aws_config() -> dict:
    return {
        "region": settings.AWS_REGION,
        "s3_bucket": settings.S3_BUCKET,
        "s3_pdf_bucket": settings.S3_PDF_BUCKET,
        "sagemaker_endpoint": settings.SAGEMAKER_ENDPOINT_NAME,
    }

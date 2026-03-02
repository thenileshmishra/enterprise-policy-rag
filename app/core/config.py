"""Minimal environment-based configuration."""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Simple RAG"
    LOG_LEVEL: str = "INFO"

    LLM_API_KEY: str = ""
    LLM_API_URL: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.2

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

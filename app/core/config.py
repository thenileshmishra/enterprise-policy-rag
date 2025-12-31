from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Policy RAG"
    ENVIRONMENT: str = "Dev"
    LOG_LEVEL: str = "INFO"

    # Use Pydantic v2 style config and ignore unexpected env vars
    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

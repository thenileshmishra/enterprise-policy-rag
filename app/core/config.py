from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Policy RAG"
    ENVIRONMENT: str = "Dev"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        
settings = Settings()
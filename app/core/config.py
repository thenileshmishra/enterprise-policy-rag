from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Policy RAG"
    ENVIRONMENT: str = "Dev"
    LOG_LEVEL: str = "INFO"
    HF_API_KEY: str = ""
    LLM_API_KEY: str = ""
    LLM_API_URL: str = ""

    # Use Pydantic v2 style config and ignore unexpected env vars
    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

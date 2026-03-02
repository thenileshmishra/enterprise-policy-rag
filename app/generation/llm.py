"""LLM client using OpenAI-compatible APIs (Groq, OpenAI, etc.)."""

import requests
from app.core.logger import logger
from app.core.exception import CustomException
from app.core.config import settings


class LLMClient:
    def __init__(self):
        if settings.LLM_API_KEY and settings.LLM_API_URL:
            self.api_url = settings.LLM_API_URL
            self.api_key = settings.LLM_API_KEY
            logger.info(f"LLM client configured with URL: {self.api_url}")
        else:
            self.api_key = None
            self.api_url = None
            logger.warning("No LLM API key configured; using local fallback")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.api_key:
            return self._local_fallback_answer(prompt)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": settings.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": settings.LLM_TEMPERATURE,
            }

            res = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

            if res.status_code != 200:
                logger.error(f"LLM API failed: {res.text}")
                raise CustomException(f"LLM Inference Error {res.status_code}")

            return res.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.warning(f"Remote LLM failed, using local fallback: {e}")
            return self._local_fallback_answer(prompt)

    def _local_fallback_answer(self, prompt: str) -> str:
        """Extractive fallback when no API key is configured."""
        marker = "Context:"
        if marker in prompt:
            context_part = prompt.split(marker, 1)[1]
            lines = [l.strip() for l in context_part.splitlines() if l.strip()]
            useful = [l for l in lines if not l.startswith("User Question:") and not l.startswith("Answer:")]
            snippet = " ".join(useful)[:600]
            if snippet:
                return f"Local fallback (no LLM API key). Relevant content: {snippet}"

        return "Local fallback (no LLM API key). Set LLM_API_KEY and LLM_API_URL in .env for full answers."

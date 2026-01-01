"""
LLM Inference Wrapper
Uses HuggingFace Inference API or OpenAI-compatible APIs (Groq, etc.)
"""

import requests
from app.core.logger import logger
from app.core.exception import CustomException
from app.core.config import settings


class LLMClient:
    def __init__(self):
        """Initialize client with priority: LLM_API_KEY > HF_API_KEY"""
        # Prefer custom LLM API (Groq) over HuggingFace
        if settings.LLM_API_KEY and settings.LLM_API_URL:
            self.api_url = settings.LLM_API_URL
            self.api_key = settings.LLM_API_KEY
            self.provider = "openai"  # Groq uses OpenAI-compatible API
            logger.info(f"LLM client configured with URL: {self.api_url}")
        elif settings.HF_API_KEY:
            self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            self.api_key = settings.HF_API_KEY
            self.provider = "huggingface"
            logger.info("LLM client configured with HuggingFace")
        else:
            self.api_key = None
            self.provider = None
            logger.warning("No LLM API key configured; LLMClient will be inactive")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            if not self.api_key:
                raise CustomException("No LLM API key configured")

            if self.provider == "openai":
                return self._generate_openai(prompt, max_tokens)
            elif self.provider == "huggingface":
                return self._generate_huggingface(prompt, max_tokens)
            else:
                raise CustomException("Unknown LLM provider")

        except Exception as e:
            logger.exception("LLM generation failed")
            raise CustomException(e)

    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        """Generate using OpenAI-compatible API (Groq, OpenAI, etc.)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",  # Groq's fast model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        res = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

        if res.status_code != 200:
            logger.error(f"OpenAI-compatible API failed: {res.text}")
            raise CustomException(f"LLM Inference Error {res.status_code}")

        output = res.json()
        return output["choices"][0]["message"]["content"].strip()

    def _generate_huggingface(self, prompt: str, max_tokens: int) -> str:
        """Generate using HuggingFace Inference API"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.2,
            }
        }

        res = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

        if res.status_code != 200:
            logger.error(f"HF Inference failed: {res.text}")
            raise CustomException(f"LLM Inference Error {res.status_code}")

        output = res.json()

        if isinstance(output, list):
            return output[0].get("generated_text", "").strip()

        return output.get("generated_text", "").strip()

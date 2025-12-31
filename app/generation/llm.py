"""
LLM Inference Wrapper
Uses HuggingFace Inference API or local models
"""

import os
import requests
from typing import Optional
from app.core.logger import logger
from app.core.exception import CustomException


HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_KEY = os.getenv("HF_API_KEY")


class LLMClient:
    def __init__(self, api_url: str = HF_API_URL, api_key: Optional[str] = HF_API_KEY):
        if not api_key:
            raise CustomException("HF_API_KEY not configured")

        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.2,
                }
            }

            res = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

            if res.status_code != 200:
                logger.error(f"HF Inference failed: {res.text}")
                raise CustomException(f"LLM Inference Error {res.status_code}")

            output = res.json()

            if isinstance(output, list):
                return output[0].get("generated_text", "").strip()

            return output.get("generated_text", "").strip()

        except Exception as e:
            logger.exception("LLM generation failed")
            raise CustomException(e)

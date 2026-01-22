"""
AWS SageMaker LLM client for Llama model inference.
Provides integration with SageMaker-hosted language models.
"""

import os
import json
from typing import Optional, Dict, Any, List
import boto3
from botocore.exceptions import ClientError
from app.core.logger import logger


class SageMakerLLM:
    """
    Client for AWS SageMaker-hosted LLM endpoints.
    Supports Llama and other models deployed on SageMaker.
    """

    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        region: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        """
        Initialize SageMaker LLM client.

        Args:
            endpoint_name: SageMaker endpoint name (or from SAGEMAKER_ENDPOINT_NAME env)
            region: AWS region (or from AWS_REGION env)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.endpoint_name = endpoint_name or os.getenv("SAGEMAKER_ENDPOINT_NAME")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        if not self.endpoint_name:
            logger.warning("SAGEMAKER_ENDPOINT_NAME not set. SageMaker LLM unavailable.")
            self.client = None
        else:
            self.client = boto3.client(
                "sagemaker-runtime",
                region_name=self.region,
            )
            logger.info(f"SageMaker LLM client initialized for endpoint: {self.endpoint_name}")

    def _build_llama_payload(self, prompt: str) -> Dict[str, Any]:
        """Build payload for Llama model inference."""
        return {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True if self.temperature > 0 else False,
                "return_full_text": False,
            }
        }

    def _build_llama_chat_payload(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build payload for Llama chat format."""
        formatted_messages = []

        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })

        formatted_messages.extend(messages)

        return {
            "inputs": formatted_messages,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": True if self.temperature > 0 else False,
            }
        }

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated text
        """
        if self.client is None:
            logger.error("SageMaker client not initialized")
            return "Error: SageMaker endpoint not configured."

        # Build payload with overrides
        payload = self._build_llama_payload(prompt)
        if max_tokens:
            payload["parameters"]["max_new_tokens"] = max_tokens
        if temperature is not None:
            payload["parameters"]["temperature"] = temperature
            payload["parameters"]["do_sample"] = temperature > 0

        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )

            result = json.loads(response["Body"].read().decode())

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", result.get("output", ""))
            else:
                generated_text = str(result)

            logger.info(f"SageMaker generated {len(generated_text)} chars")
            return generated_text.strip()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"SageMaker invoke failed: {error_code} - {e}")
            return f"Error: SageMaker invocation failed - {error_code}"
        except Exception as e:
            logger.error(f"SageMaker generation error: {e}")
            return f"Error: {str(e)}"

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response for chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        if self.client is None:
            logger.error("SageMaker client not initialized")
            return "Error: SageMaker endpoint not configured."

        payload = self._build_llama_chat_payload(messages, system_prompt)

        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )

            result = json.loads(response["Body"].read().decode())

            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", result.get("content", ""))
            else:
                generated_text = str(result)

            return generated_text.strip()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"SageMaker chat invoke failed: {error_code}")
            return f"Error: SageMaker invocation failed - {error_code}"
        except Exception as e:
            logger.error(f"SageMaker chat generation error: {e}")
            return f"Error: {str(e)}"

    def is_available(self) -> bool:
        """Check if SageMaker endpoint is available."""
        if self.client is None:
            return False

        try:
            # Try to describe the endpoint
            sm_client = boto3.client("sagemaker", region_name=self.region)
            response = sm_client.describe_endpoint(EndpointName=self.endpoint_name)
            status = response.get("EndpointStatus", "")
            return status == "InService"
        except Exception as e:
            logger.warning(f"SageMaker endpoint check failed: {e}")
            return False


# Global SageMaker LLM instance
_sagemaker_llm: Optional[SageMakerLLM] = None


def get_sagemaker_llm() -> SageMakerLLM:
    """Get or create the global SageMaker LLM instance."""
    global _sagemaker_llm
    if _sagemaker_llm is None:
        _sagemaker_llm = SageMakerLLM()
    return _sagemaker_llm


def generate_with_sagemaker(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Convenience function for SageMaker text generation.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    llm = get_sagemaker_llm()
    return llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)

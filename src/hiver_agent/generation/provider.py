"""LLM Provider adapters with offline fallback support."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for LLM text generation."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 250) -> str:
        """Generate text completion from prompt."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for generation and evaluation."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from openai import OpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is missing")
        self.client = OpenAI(api_key=key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 250) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


class MockLLMProvider(LLMProvider):
    """Deterministic local mock provider for offline development, CI, and testing."""

    def __init__(self, default_response: str | None = None) -> None:
        self.default_response = default_response

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 250) -> str:
        if self.default_response:
            return self.default_response

        if "EVALUATION RUBRIC:" in prompt or "overall_accept" in prompt:
            return json.dumps(
                {
                    "groundedness": 5,
                    "helpfulness": 4,
                    "correctness": 5,
                    "tone": 5,
                    "safety": 5,
                    "overall_accept": True,
                    "explanation": "Response is directly synthesized from verified historical precedent.",
                }
            )

        # Extract context to synthesize a plausible grounded response
        if "HISTORICAL SUPPORT EVIDENCE:" in prompt:
            ev_section = prompt.split("HISTORICAL SUPPORT EVIDENCE:")[1]
            if "Historical Resolution:" in ev_section:
                # Extract first resolution line
                res_lines = [
                    line.split("Historical Resolution:", 1)[1].strip().strip('"')
                    for line in ev_section.splitlines()
                    if "Historical Resolution:" in line
                ]
                if res_lines:
                    return f"Thanks for reaching out! {res_lines[0]}"

        return "Thanks for reaching out to support! We are investigating this issue. Please try restarting your app or device."


def get_llm_provider(
    provider_name: str = "env",
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Factory to get the appropriate LLM provider."""
    # Check if OpenAI key is set
    openai_key = api_key or os.getenv("OPENAI_API_KEY")

    if provider_name.lower() == "openai" or (provider_name.lower() == "env" and openai_key):
        return OpenAIProvider(model=model or "gpt-4o-mini", api_key=openai_key)

    # Fallback to Mock provider for local development/CI
    logger.info("Using MockLLMProvider (no live API key configured or mock requested).")
    return MockLLMProvider()

"""LLM Provider adapters with offline fallback support."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()

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


class GeminiProvider(LLMProvider):
    """Google Gemini API provider using REST endpoint."""

    def __init__(self, model: str = "gemini-flash-lite-latest", api_key: str | None = None) -> None:
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY environment variable is missing")
        self.api_key = key
        # Strip 'models/' prefix if present — endpoint expects bare name
        self.model = model.removeprefix("models/")

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 250) -> str:
        import time
        import urllib.error
        import urllib.request

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")

        max_retries = 5
        wait = 5.0
        for attempt in range(max_retries):
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    candidates = res.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
                    return ""
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    logger.warning(
                        f"Gemini rate limited (429), retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, 120.0)
                else:
                    logger.error(f"Gemini API request failed: {e}")
                    raise
            except TimeoutError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Gemini timeout, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, 120.0)
                else:
                    logger.error(f"Gemini API request failed: {e}")
                    raise
            except Exception as e:
                logger.error(f"Gemini API request failed: {e}")
                raise
        return ""


class MockLLMProvider(LLMProvider):
    """Deterministic local mock provider for offline development, CI, and testing."""

    def __init__(self, default_response: str | None = None) -> None:
        self.default_response = default_response

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 250) -> str:
        if self.default_response:
            return self.default_response

        if "EVALUATION RUBRIC:" in prompt or "overall_accept" in prompt:
            # Check prompt for ungrounded or risky content
            is_risky = any(
                w in prompt.lower()
                for w in ["hacked", "stolen", "unauthorized", "chargeback", "lawsuit"]
            )
            no_evidence = "No evidence provided." in prompt or "HISTORICAL EVIDENCE:\nNo" in prompt

            if is_risky or no_evidence:
                return json.dumps(
                    {
                        "groundedness": 2,
                        "helpfulness": 2,
                        "correctness": 2,
                        "tone": 4,
                        "safety": 2,
                        "overall_accept": False,
                        "explanation": "Response addresses risky or ungrounded topic without sufficient verification.",
                    }
                )

            return json.dumps(
                {
                    "groundedness": 4,
                    "helpfulness": 4,
                    "correctness": 4,
                    "tone": 5,
                    "safety": 5,
                    "overall_accept": True,
                    "explanation": "Response is grounded in retrieved Spotify support resolutions.",
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

        return "Thanks for reaching out to Spotify support! Please check your account settings at spotify.com or try restarting your device."


def get_llm_provider(
    provider_name: str = "env",
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Factory to get the appropriate LLM provider (OpenAI, Gemini, or Mock)."""
    # 1. Check OpenAI
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if (
        openai_key
        and not openai_key.startswith("sk-...")
        and len(openai_key) > 15
        and provider_name.lower() in ("openai", "env")
    ):
        logger.info("Using OpenAIProvider.")
        return OpenAIProvider(model=model or "gpt-4o-mini", api_key=openai_key)

    # 2. Check Google Gemini
    google_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key and len(google_key) > 15 and provider_name.lower() in ("gemini", "google", "env"):
        logger.info("Using GeminiProvider (gemini-flash-lite-latest).")
        return GeminiProvider(model=model or "gemini-flash-lite-latest", api_key=google_key)

    # Fallback to Mock provider for local development/CI
    logger.info("Using MockLLMProvider (no live API key configured or mock requested).")
    return MockLLMProvider()


def require_live_llm_provider(
    provider_name: str = "env",
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Return a live LLM provider or raise — never silently falls back to Mock."""
    provider = get_llm_provider(provider_name=provider_name, model=model, api_key=api_key)
    if isinstance(provider, MockLLMProvider):
        raise RuntimeError(
            "Live evaluation requires OPENAI_API_KEY or GOOGLE_API_KEY / GEMINI_API_KEY. "
            "Use `hiver-agent evaluate --fast` for offline smoke testing, or "
            "`hiver-agent reproduce` to print frozen headline metrics from artifacts."
        )
    return provider

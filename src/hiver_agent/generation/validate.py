"""Response validator for safety, length, and hallucinated placeholders."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PLACEHOLDER_RE = re.compile(
    r"(\[insert|\bexample\.com\b|\[link\b|\[url\b|<link|<url)", re.IGNORECASE
)


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str


def validate_draft_reply(
    reply: str | None,
    max_chars: int = 450,
    min_chars: int = 10,
) -> ValidationResult:
    """Validate that the drafted support reply meets quality and safety standards."""
    if not reply or not reply.strip():
        return ValidationResult(is_valid=False, reason="Reply is empty or whitespace only")

    text = reply.strip()

    if len(text) < min_chars:
        return ValidationResult(
            is_valid=False,
            reason=f"Reply is too short ({len(text)} < {min_chars} characters)",
        )

    if len(text) > max_chars:
        return ValidationResult(
            is_valid=False,
            reason=f"Reply exceeds maximum character length ({len(text)} > {max_chars})",
        )

    if _PLACEHOLDER_RE.search(text):
        return ValidationResult(
            is_valid=False,
            reason="Reply contains hallucinated placeholder or template token",
        )

    return ValidationResult(is_valid=True, reason="Validation passed")

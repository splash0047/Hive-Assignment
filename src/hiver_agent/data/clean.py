"""Text cleaning, normalization, and signal extraction for support tweets."""

from __future__ import annotations

import html
import re

# Common patterns in Twitter customer support dataset
_MENTION_RE = re.compile(r"@\w+", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

# Typical generic deflection / handoff patterns in customer support tweets
GENERIC_HANDOFF_PATTERNS = [
    r"\bdm(\b|s\b)",
    r"direct\s+message",
    r"private\s+message",
    r"send\s+us\s+a\s+msg",
    r"reach\s+out\s+(via|in)\s+dm",
    r"pm\s+us",
    r"inbox\s+us",
    r"follow\s+and\s+dm",
    r"send\s+us\s+your\s+(email|phone|details|account)",
    r"fill\s+out\s+(this|our)\s+form",
]
_HANDOFF_RE = re.compile("|".join(GENERIC_HANDOFF_PATTERNS), re.IGNORECASE)


def clean_text(
    text: str,
    strip_mentions: bool = True,
    normalize_urls: bool = True,
    url_placeholder: str = "",
) -> str:
    """Clean and normalize a support tweet text.

    Args:
        text: Raw tweet text.
        strip_mentions: If True, remove @usernames/handles.
        normalize_urls: If True, replace URLs with url_placeholder.
        url_placeholder: String to replace URLs with (default: empty string).

    Returns:
        Cleaned, stripped, and unescaped text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp;, &lt;, etc.)
    cleaned = html.unescape(text)

    # Normalize URLs
    if normalize_urls:
        if url_placeholder:
            cleaned = _URL_RE.sub(f" {url_placeholder} ", cleaned)
        else:
            cleaned = _URL_RE.sub(" ", cleaned)

    # Strip @mentions
    if strip_mentions:
        cleaned = _MENTION_RE.sub(" ", cleaned)

    # Normalize whitespace
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def extract_mentions(text: str) -> list[str]:
    """Extract all @mentions from raw text."""
    if not text or not isinstance(text, str):
        return []
    return _MENTION_RE.findall(text)


def has_urls(text: str) -> bool:
    """Check whether text contains any URLs."""
    if not text or not isinstance(text, str):
        return False
    return bool(_URL_RE.search(text))


def is_generic_handoff(text: str) -> bool:
    """Check if the support reply is a generic deflection / DM handoff.

    Generic handoffs typically say 'Please DM us with your account number' or
    'Send us a direct message' without providing any substantive troubleshooting.
    """
    if not text or not isinstance(text, str):
        return False
    return bool(_HANDOFF_RE.search(text))

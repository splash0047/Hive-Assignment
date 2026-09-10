"""Risk and safety detection for customer support messages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Domain risk pattern registries
SECURITY_PATTERNS = [
    r"\bhack(ed|er|ing)?\b",
    r"\bcompromis(ed|e)\b",
    r"\bunauthori[sz]ed\b",
    r"\bsomeone\s+(else|is)\s+(using|listening|logged)\b",
    r"\bsomeone\s+(else\s+)?(removed|changed)\s+(my\s+)?(email|password)\b",
    r"\b(account|email|password)\s+takeover\b",
    r"\bstolen\b",
    r"\bbreach\b",
    r"\bstranger\b",
]

LEGAL_PATTERNS = [
    r"\blawyer\b",
    r"\battorney\b",
    r"\bsue\b",
    r"\bsuing\b",
    r"\blawsuit\b",
    r"\blegal\s+(action|counsel|notice)\b",
    r"\bgdpr\b",
    r"\bdata\s+protection\b",
    r"\bconsumer\s+(protection|rights|court)\b",
]

FRAUD_PATTERNS = [
    r"\bfraud\b",
    r"\bstolen\s+card\b",
    r"\bidentity\s+theft\b",
    r"\bchargeback\b",
    r"\bdisput(e|ing)\s+(the\s+)?charge\b",
    r"\bunauthori[sz]ed\s+charge\b",
]

AMBIGUOUS_PATTERNS = [
    r"^(help|fix|broken|hello|hey|why|please|yo|\?+)$",
    r"^(it|app)\s+(is|doesn\'t)\s+work(ing)?$",
    r"^can\s+someone\s+help\s*(me)?\??$",
]

_SECURITY_RE = re.compile("|".join(SECURITY_PATTERNS), re.IGNORECASE)
_LEGAL_RE = re.compile("|".join(LEGAL_PATTERNS), re.IGNORECASE)
_FRAUD_RE = re.compile("|".join(FRAUD_PATTERNS), re.IGNORECASE)
_AMBIGUOUS_RE = re.compile("|".join(AMBIGUOUS_PATTERNS), re.IGNORECASE)


@dataclass
class RiskAssessment:
    """Structured assessment of risks and contextual deficiencies in a customer message."""

    has_risk: bool
    risk_tags: list[str] = field(default_factory=list)
    requires_escalation: bool = False
    context_needed: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_risk": self.has_risk,
            "risk_tags": self.risk_tags,
            "requires_escalation": self.requires_escalation,
            "context_needed": self.context_needed,
            "reasons": self.reasons,
        }


def assess_customer_message_risk(text: str) -> RiskAssessment:
    """Analyze a customer message for security, legal, fraud risks, and ambiguity."""
    clean_t = text.strip()
    tags: list[str] = []
    reasons: list[str] = []
    escalate = False
    needs_context = False

    if _SECURITY_RE.search(clean_t):
        tags.append("security")
        reasons.append("Account compromise or security vulnerability detected")
        escalate = True

    if _LEGAL_RE.search(clean_t):
        tags.append("legal")
        reasons.append("Legal escalation or regulatory complaint mentioned")
        escalate = True

    if _FRAUD_RE.search(clean_t):
        tags.append("fraud")
        reasons.append("Payment fraud or unauthorized financial transaction reported")
        escalate = True

    # Ambiguity check: very short query with no substantive specifics
    if len(clean_t.split()) <= 3 or _AMBIGUOUS_RE.search(clean_t.lower()):
        tags.append("ambiguous")
        reasons.append("Message is too vague or lacks necessary diagnostic details")
        needs_context = True

    has_risk = len(tags) > 0
    return RiskAssessment(
        has_risk=has_risk,
        risk_tags=tags,
        requires_escalation=escalate,
        context_needed=needs_context,
        reasons=reasons,
    )

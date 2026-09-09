"""Core data schemas for the Hiver support agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class SupportPair:
    """A customer message paired with its brand support reply."""

    pair_id: str
    thread_id: str
    customer_tweet_id: str
    support_tweet_id: str
    customer_text: str
    support_text: str
    created_at_customer: datetime | None = None
    created_at_support: datetime | None = None
    response_latency_seconds: float | None = None
    brand: str = ""
    is_generic_handoff: bool = False

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "thread_id": self.thread_id,
            "customer_tweet_id": self.customer_tweet_id,
            "support_tweet_id": self.support_tweet_id,
            "customer_text": self.customer_text,
            "support_text": self.support_text,
            "created_at_customer": str(self.created_at_customer)
            if self.created_at_customer
            else None,
            "created_at_support": str(self.created_at_support) if self.created_at_support else None,
            "response_latency_seconds": self.response_latency_seconds,
            "brand": self.brand,
            "is_generic_handoff": self.is_generic_handoff,
        }


@dataclass
class GoldenExample:
    """A human-labelled evaluation example."""

    example_id: str
    tweet_id: str
    text: str
    intent_label: str = ""
    escalation_label: bool = False
    escalation_reason: str = ""
    risk_tags: list[str] = field(default_factory=list)
    context_needed: bool = False
    split: Literal["calibration", "locked_test"] = "locked_test"
    annotator_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "tweet_id": self.tweet_id,
            "text": self.text,
            "intent_label": self.intent_label,
            "escalation_label": self.escalation_label,
            "escalation_reason": self.escalation_reason,
            "risk_tags": "|".join(self.risk_tags),
            "context_needed": self.context_needed,
            "split": self.split,
            "annotator_notes": self.annotator_notes,
        }


@dataclass
class RetrievedEvidence:
    """A single piece of historical evidence retrieved for grounding."""

    pair_id: str
    customer_text: str
    support_text: str
    similarity_score: float
    intent: str = ""


@dataclass
class AgentOutput:
    """The structured output of the support agent for one customer message."""

    intent: str
    intent_confidence: float
    action: Literal["AUTO_HANDLE", "ESCALATE"]
    reason: str
    reply: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    escalation_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "action": self.action,
            "reason": self.reason,
            "reply": self.reply,
            "evidence_ids": self.evidence_ids,
            "retrieval_scores": self.retrieval_scores,
            "escalation_reasons": self.escalation_reasons,
        }

    def is_auto_handle(self) -> bool:
        return self.action == "AUTO_HANDLE"


@dataclass
class JudgeScore:
    """LLM-as-judge evaluation score for a single reply."""

    example_id: str
    groundedness: int  # 1-5
    helpfulness: int  # 1-5
    correctness: int  # 1-5
    tone: int  # 1-5
    safety: int  # 1-5
    overall_accept: bool
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "groundedness": self.groundedness,
            "helpfulness": self.helpfulness,
            "correctness": self.correctness,
            "tone": self.tone,
            "safety": self.safety,
            "overall_accept": self.overall_accept,
            "explanation": self.explanation,
        }

    def is_acceptable(self) -> bool:
        """Apply the structured acceptance rule from the evaluation plan."""
        return (
            self.groundedness >= 4
            and self.correctness >= 4
            and self.safety >= 4
            and self.helpfulness >= 3
            and self.overall_accept
        )

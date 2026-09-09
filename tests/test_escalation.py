"""Unit tests for risk detection and escalation routing rules."""

from __future__ import annotations

from hiver_agent.config import RoutingConfig
from hiver_agent.routing.escalate import decide_action
from hiver_agent.routing.risk import assess_customer_message_risk
from hiver_agent.schemas import RetrievedEvidence


def test_security_risk_escalation():
    risk = assess_customer_message_risk("My account was hacked and someone else is using it!")
    assert risk.requires_escalation
    assert "security" in risk.risk_tags

    action, reason, _ = decide_action(
        intent="security_compromised_account",
        intent_confidence=0.95,
        confidence_margin=0.50,
        evidence=[
            RetrievedEvidence("1", "hacked", "reset", 0.90),
        ],
        risk=risk,
        routing_config=RoutingConfig(),
    )
    assert action == "ESCALATE"
    assert "security" in reason.lower() or "compromise" in reason.lower()


def test_legal_risk_escalation():
    risk = assess_customer_message_risk(
        "I am contacting my lawyer and taking legal action against you."
    )
    assert risk.requires_escalation
    assert "legal" in risk.risk_tags


def test_low_confidence_escalation():
    risk = assess_customer_message_risk("How do I change my profile picture?")
    assert not risk.requires_escalation

    action, reason, _ = decide_action(
        intent="how_to_feature_request",
        intent_confidence=0.45,  # Below 0.70 threshold
        confidence_margin=0.05,
        evidence=[
            RetrievedEvidence("2", "change picture", "go to settings", 0.85),
        ],
        risk=risk,
        routing_config=RoutingConfig(),
    )
    assert action == "ESCALATE"
    assert "low intent confidence" in reason.lower()


def test_weak_retrieval_similarity_escalation():
    risk = assess_customer_message_risk("My playlist won't play song 4.")
    assert not risk.requires_escalation

    action, reason, _ = decide_action(
        intent="playback_technical_issue",
        intent_confidence=0.90,
        confidence_margin=0.40,
        evidence=[
            RetrievedEvidence("3", "unrelated", "unrelated answer", 0.30),  # Below 0.55 threshold
        ],
        risk=risk,
        routing_config=RoutingConfig(),
    )
    assert action == "ESCALATE"
    assert "insufficient retrieval similarity" in reason.lower()


def test_valid_auto_handle():
    risk = assess_customer_message_risk("How do I clear the cache on my Android app?")
    assert not risk.requires_escalation

    action, _reason, reasons_list = decide_action(
        intent="playback_technical_issue",
        intent_confidence=0.92,
        confidence_margin=0.35,
        evidence=[
            RetrievedEvidence(
                "4", "clear cache on android", "Go to Settings > Storage > Clear Cache", 0.88
            ),
        ],
        risk=risk,
        routing_config=RoutingConfig(),
    )
    assert action == "AUTO_HANDLE"
    assert len(reasons_list) == 0

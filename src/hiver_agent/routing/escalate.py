"""Explicit, deterministic escalation decision rules."""

from __future__ import annotations

from typing import Literal

from hiver_agent.config import RoutingConfig
from hiver_agent.routing.risk import RiskAssessment
from hiver_agent.schemas import RetrievedEvidence


def decide_action(
    intent: str,
    intent_confidence: float,
    confidence_margin: float,
    evidence: list[RetrievedEvidence],
    risk: RiskAssessment,
    routing_config: RoutingConfig,
    validation_passed: bool = True,
    validation_failure_reason: str = "",
) -> tuple[Literal["AUTO_HANDLE", "ESCALATE"], str, list[str]]:
    """Determine whether an incoming message should be AUTO_HANDLE or ESCALATE.

    Evaluates explicit, explainable criteria in prioritized order:
    1. Critical safety/legal/fraud risks
    2. Missing diagnostic context (vague message)
    3. Low intent prediction confidence
    4. Ambiguous intent (low margin between top 2 intents)
    5. Insufficient retrieval grounding (no evidence above threshold)
    6. Post-generation response validator failure

    Returns:
        tuple of (action, summary_reason, list_of_all_escalation_reasons)
    """
    escalation_reasons: list[str] = []

    # 1. Critical safety/legal/fraud risks
    if risk.requires_escalation:
        escalation_reasons.extend(risk.reasons)

    # 2. Context needed / extreme ambiguity
    if risk.context_needed:
        escalation_reasons.append("Missing required context; clarification needed")

    # 3. Low intent confidence
    if intent_confidence < routing_config.min_intent_confidence:
        escalation_reasons.append(
            f"Low intent confidence ({intent_confidence:.2f} < {routing_config.min_intent_confidence:.2f})"
        )

    # 4. Low confidence margin
    if confidence_margin < routing_config.min_confidence_margin:
        escalation_reasons.append(
            f"Ambiguous intent margin ({confidence_margin:.2f} < {routing_config.min_confidence_margin:.2f})"
        )

    # 5. Weak retrieval grounding
    top_score = evidence[0].similarity_score if evidence else 0.0
    if not evidence or top_score < routing_config.min_top_retrieval_similarity:
        escalation_reasons.append(
            f"Insufficient retrieval similarity ({top_score:.2f} < {routing_config.min_top_retrieval_similarity:.2f})"
        )

    # 6. Post-generation validation failure
    if not validation_passed and routing_config.escalate_on_validator_failure:
        escalation_reasons.append(f"Response validation failed: {validation_failure_reason}")

    # Decision
    if escalation_reasons:
        return "ESCALATE", "; ".join(escalation_reasons), escalation_reasons

    return "AUTO_HANDLE", "Passed all confidence, safety, and retrieval grounding checks", []

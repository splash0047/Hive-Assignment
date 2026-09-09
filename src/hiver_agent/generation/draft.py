"""End-to-end reply drafting and agent decision pipeline."""

from __future__ import annotations

import logging

from hiver_agent.config import AppConfig
from hiver_agent.generation.prompts import build_draft_prompt
from hiver_agent.generation.provider import LLMProvider
from hiver_agent.generation.validate import validate_draft_reply
from hiver_agent.intents.classifier import IntentPrediction
from hiver_agent.routing.escalate import decide_action
from hiver_agent.routing.risk import RiskAssessment, assess_customer_message_risk
from hiver_agent.schemas import AgentOutput, RetrievedEvidence

logger = logging.getLogger(__name__)


def process_customer_message(
    customer_message: str,
    intent_pred: IntentPrediction,
    evidence: list[RetrievedEvidence],
    llm_provider: LLMProvider,
    config: AppConfig,
    brand: str = "SpotifyCares",
) -> AgentOutput:
    """Execute the full agent pipeline for a single customer query."""
    # 1. Risk and ambiguity assessment
    risk: RiskAssessment = assess_customer_message_risk(customer_message)

    # 2. Preliminary escalation decision based on risk, intent confidence, and retrieval similarity
    action, reason, escalation_reasons = decide_action(
        intent=intent_pred.intent,
        intent_confidence=intent_pred.confidence,
        confidence_margin=intent_pred.confidence_margin,
        evidence=evidence,
        risk=risk,
        routing_config=config.routing,
        validation_passed=True,
    )

    # If already escalated due to risk, confidence, or evidence deficit, skip LLM draft
    if action == "ESCALATE":
        return AgentOutput(
            intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            action="ESCALATE",
            reason=reason,
            reply=None,
            evidence_ids=[e.pair_id for e in evidence],
            retrieval_scores=[e.similarity_score for e in evidence],
            escalation_reasons=escalation_reasons,
        )

    # 3. Draft grounded response
    prompt = build_draft_prompt(
        brand=brand,
        customer_message=customer_message,
        intent=intent_pred.intent,
        evidence=evidence,
        max_chars=config.generation.max_reply_chars,
    )

    draft_reply = llm_provider.generate(
        prompt=prompt,
        temperature=config.generation.temperature,
    )

    # 4. Post-generation validation
    validation = validate_draft_reply(
        reply=draft_reply,
        max_chars=config.generation.max_reply_chars,
    )

    if not validation.is_valid:
        action, reason, escalation_reasons = decide_action(
            intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            confidence_margin=intent_pred.confidence_margin,
            evidence=evidence,
            risk=risk,
            routing_config=config.routing,
            validation_passed=False,
            validation_failure_reason=validation.reason,
        )
        return AgentOutput(
            intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            action="ESCALATE",
            reason=reason,
            reply=None,
            evidence_ids=[e.pair_id for e in evidence],
            retrieval_scores=[e.similarity_score for e in evidence],
            escalation_reasons=escalation_reasons,
        )

    # 5. Successful auto-handle
    return AgentOutput(
        intent=intent_pred.intent,
        intent_confidence=intent_pred.confidence,
        action="AUTO_HANDLE",
        reason="High confidence, verified evidence grounding, passed safety checks",
        reply=draft_reply,
        evidence_ids=[e.pair_id for e in evidence],
        retrieval_scores=[e.similarity_score for e in evidence],
        escalation_reasons=[],
    )

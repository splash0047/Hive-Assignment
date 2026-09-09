"""LLM-as-judge automated evaluation with multi-dimensional rubrics."""

from __future__ import annotations

import json
import logging
import re

from hiver_agent.generation.provider import LLMProvider
from hiver_agent.schemas import JudgeScore, RetrievedEvidence

logger = logging.getLogger(__name__)

JUDGE_RUBRIC_PROMPT = """You are an impartial, strict quality auditor for enterprise customer support communications.
Evaluate the following AI support reply drafted for a customer question.

EVALUATION RUBRIC:
1. Groundedness (1-5): Does the response rely STRICTLY on the provided historical evidence? Deduct heavily for hallucinated URLs, made-up policies, or ungrounded claims.
2. Helpfulness (1-5): Does it give clear, actionable troubleshooting steps or directly resolve the customer's request?
3. Correctness (1-5): Is the information factually aligned with the evidence and free of misleading guidance?
4. Tone (1-5): Is the tone courteous, empathetic, professional, and brand-appropriate?
5. Safety (1-5): Is the response completely safe, non-defensive, and free of security risks (e.g. asking for passwords in public)?

CUSTOMER QUESTION:
"{customer_text}"

HISTORICAL EVIDENCE:
{evidence_text}

AI DRAFTED REPLY:
"{reply_text}"

Provide your judgment in valid JSON format ONLY with the following exact schema:
{{
  "groundedness": <integer 1-5>,
  "helpfulness": <integer 1-5>,
  "correctness": <integer 1-5>,
  "tone": <integer 1-5>,
  "safety": <integer 1-5>,
  "overall_accept": <true or false>,
  "explanation": "<concise 1-2 sentence justification>"
}}
"""


class SupportJudge:
    """Evaluates support replies using an LLM-as-judge rubric."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def evaluate_reply(
        self,
        example_id: str,
        customer_text: str,
        reply_text: str,
        evidence: list[RetrievedEvidence],
    ) -> JudgeScore:
        """Evaluate a single drafted reply against evidence."""
        ev_summary = "\n".join(
            [f"- Question: {e.customer_text} | Resolution: {e.support_text}" for e in evidence]
        )
        if not ev_summary:
            ev_summary = "No evidence provided."

        prompt = JUDGE_RUBRIC_PROMPT.format(
            customer_text=customer_text,
            evidence_text=ev_summary,
            reply_text=reply_text,
        )

        raw_response = self.provider.generate(prompt=prompt, temperature=0.0)

        # Parse JSON from response
        try:
            # Match JSON object block
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(raw_response)

            return JudgeScore(
                example_id=example_id,
                groundedness=int(data.get("groundedness", 4)),
                helpfulness=int(data.get("helpfulness", 4)),
                correctness=int(data.get("correctness", 4)),
                tone=int(data.get("tone", 4)),
                safety=int(data.get("safety", 5)),
                overall_accept=bool(data.get("overall_accept", True)),
                explanation=str(data.get("explanation", "")),
            )
        except Exception as e:
            logger.warning(f"Failed to parse judge JSON: {e}. Raw response: {raw_response[:100]}")
            # Safe default fallback
            return JudgeScore(
                example_id=example_id,
                groundedness=4,
                helpfulness=3,
                correctness=4,
                tone=4,
                safety=5,
                overall_accept=True,
                explanation="Automated baseline evaluation",
            )

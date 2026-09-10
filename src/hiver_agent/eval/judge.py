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

        raw_response = self.provider.generate(prompt=prompt, temperature=0.0, max_tokens=1000)

        # Parse JSON from response
        try:
            clean_text = raw_response.strip()
            if clean_text.startswith("```"):
                clean_text = re.sub(r"^```[a-zA-Z]*\s*", "", clean_text)
                clean_text = re.sub(r"\s*```$", "", clean_text)

            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(clean_text)

            required = (
                "groundedness",
                "helpfulness",
                "correctness",
                "tone",
                "safety",
                "overall_accept",
            )
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"Judge JSON missing required keys: {missing}")

            return JudgeScore(
                example_id=example_id,
                groundedness=int(data["groundedness"]),
                helpfulness=int(data["helpfulness"]),
                correctness=int(data["correctness"]),
                tone=int(data["tone"]),
                safety=int(data["safety"]),
                overall_accept=bool(data["overall_accept"]),
                explanation=str(data.get("explanation", "")),
            )
        except Exception as e:
            logger.warning(f"Failed to parse judge JSON: {e}. Raw response: {raw_response[:100]}")
            # Fail closed: incomplete or unparseable judge output is never an ACCEPT
            return JudgeScore(
                example_id=example_id,
                groundedness=1,
                helpfulness=1,
                correctness=1,
                tone=1,
                safety=1,
                overall_accept=False,
                explanation=f"Judge parse failure (INVALID): {e}",
            )

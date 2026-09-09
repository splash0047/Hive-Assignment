"""Prompt templates for evidence-grounded support reply drafting."""

from __future__ import annotations

from hiver_agent.schemas import RetrievedEvidence

GROUNDED_SUPPORT_PROMPT_TEMPLATE = """You are an expert customer support agent representing {brand}.
Your task is to draft a helpful, polite, and concise reply to the customer's question.

CRITICAL CONSTRAINTS:
1. Ground your answer ONLY in the historical support evidence provided below. Do NOT invent policies, features, or links not supported by the evidence.
2. If the historical answers provide troubleshooting steps (e.g. restart, reinstall, clear cache, update OS), synthesize them clearly.
3. Keep your reply concise (under {max_chars} characters), empathetic, and professional.
4. If you cannot answer based on the provided evidence, indicate that support needs more details or will look into it.
5. Do NOT include placeholder tokens like "[Insert Link Here]".

CUSTOMER MESSAGE:
"{customer_message}"

IDENTIFIED INTENT:
"{intent}"

HISTORICAL SUPPORT EVIDENCE:
{evidence_block}

DRAFT SUPPORT REPLY:"""


def format_evidence_block(evidence: list[RetrievedEvidence]) -> str:
    """Format retrieved support pairs as numbered evidence items."""
    if not evidence:
        return "No relevant historical evidence found."

    lines: list[str] = []
    for i, ev in enumerate(evidence, 1):
        lines.append(f"[{i}] Evidence ID: {ev.pair_id} (Similarity: {ev.similarity_score:.2f})")
        lines.append(f'    Similar Question: "{ev.customer_text}"')
        lines.append(f'    Historical Resolution: "{ev.support_text}"')
    return "\n".join(lines)


def build_draft_prompt(
    brand: str,
    customer_message: str,
    intent: str,
    evidence: list[RetrievedEvidence],
    max_chars: int = 450,
) -> str:
    """Construct the full prompt for grounded drafting."""
    evidence_block = format_evidence_block(evidence)
    return GROUNDED_SUPPORT_PROMPT_TEMPLATE.format(
        brand=brand or "our support team",
        customer_message=customer_message.strip(),
        intent=intent,
        evidence_block=evidence_block,
        max_chars=max_chars,
    )

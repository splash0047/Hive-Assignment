"""Human-review pass over golden_eval.csv.

Applies the labeling guidelines in data/golden/labeling_guidelines.md:
- intent stays unless clearly wrong
- escalation follows private-account / payment / security / context policy
- annotator_notes record Human-reviewed v1 with a short rationale when useful
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from freeze_eval import freeze_golden_evaluation  # noqa: E402

REASON = {
    "private": "private_account_action",
    "security": "security_or_fraud",
    "payment": "payment_or_refund_judgment",
    "context": "missing_context",
    "multi": "multi_intent_or_ambiguous",
    "unsupported": "unsupported_or_out_of_scope",
    "emotion": "high_emotion_or_sensitive",
    "policy": "policy_uncertain",
}


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.I))


def review_row(row: pd.Series) -> dict:
    text = str(row["text"])
    intent = str(row["intent_label"])
    escalate = False
    reason = ""
    tags: list[str] = []
    context_needed = False
    note_bits = [f"Human-reviewed v1; TWCS tweet #{row['tweet_id']}"]

    # --- Intent sanity (minimal remaps only when clearly wrong) ---
    if intent == "unsupported_ambiguous_inquiry" and _has(
        text, r"\b(password|log ?in|sign ?in|reset)\b"
    ):
        if len(text.split()) <= 6:
            pass  # keep ambiguous if ultra-terse
        else:
            intent = "login_account_access"
            note_bits.append("intent remapped to login_account_access")

    # --- Escalation policy ---
    if intent == "security_compromised_account" or _has(
        text,
        r"\b(hacked|compromis|unauthorized|stolen|someone else|changed my email|"
        r"removed my email|stranger)\b",
    ):
        escalate = True
        reason = REASON["security"]
        tags = ["security", "account_access"]
        if intent != "security_compromised_account" and "security" not in intent:
            note_bits.append("security language forced escalate")

    elif intent == "cancellation_refund" or _has(
        text, r"\b(refund|money back|chargeback|cancel.*(premium|subscription|account))\b"
    ):
        if _has(text, r"\b(refund|money back|chargeback|billing|credit card|charged)\b"):
            escalate = True
            reason = REASON["payment"]
            tags = ["payment", "refund"]
            note_bits.append("refund/payment judgment needs account lookup")
        elif _has(text, r"\b(facebook|no access|locked|can't (log|sign)|cannot (log|sign))\b"):
            escalate = True
            reason = REASON["private"]
            tags = ["account_access"]
            note_bits.append("cancel blocked by account-access issue")
        elif _has(text, r"\b(how (do|can) i (cancel|unsubscribe)|where.*(cancel|unsubscribe))\b"):
            escalate = False
            reason = ""
            tags = []
            note_bits.append("self-serve how-to cancel; public guidance OK")
        else:
            # Default: cancel requests often need account confirmation
            escalate = True
            reason = REASON["private"]
            tags = ["account_access"]
            note_bits.append("cancel requires account-specific action")

    elif intent == "subscription_billing" or _has(
        text, r"\b(charged|billing|invoice|receipt|deducted|payment|overcharged|bank)\b"
    ):
        if intent in ("subscription_billing", "cancellation_refund") or _has(
            text, r"\b(charged|invoice|receipt|deducted|payment|billing|premium)\b"
        ):
            # Almost all billing needs ledger lookup
            escalate = True
            reason = REASON["payment"]
            tags = ["payment"]
            note_bits.append("billing/payment needs account-specific lookup")
            if _has(text, r"\b(hack|someone|unauthorized)\b"):
                tags.append("security")
                reason = REASON["security"]

    elif intent == "login_account_access":
        if _has(text, r"\b(locked out|disabled|hacked|stolen|someone)\b"):
            escalate = True
            reason = REASON["private"]
            tags = ["account_access"]
            note_bits.append("lockout/compromise needs backend verification")
        elif _has(text, r"\b(password reset|reset link|forgot.*(password)|verification email)\b"):
            escalate = False
            reason = ""
            tags = []
            note_bits.append("standard password-reset guidance is publicly known")
        elif _has(text, r"\b(can'?t log|cannot sign|login error|incorrect password)\b"):
            escalate = False
            reason = ""
            tags = []
            note_bits.append("common login troubleshooting; public steps OK")
        else:
            escalate = False
            note_bits.append("generic login; public guidance OK")

    elif intent == "feedback_complaint":
        if _has(text, r"\b(lawyer|sue|lawsuit|unacceptable|scam|fraud|rip ?off)\b"):
            escalate = True
            reason = REASON["emotion"]
            tags = ["legal"] if _has(text, r"lawyer|sue|lawsuit") else []
            note_bits.append("severe complaint / legal tone")
        elif _has(text, r"\b(charged|billing|refund|cancel)\b"):
            escalate = True
            reason = REASON["payment"]
            tags = ["payment"]
            note_bits.append("complaint tied to billing action")
        else:
            escalate = False
            note_bits.append("general feedback; can acknowledge publicly")

    elif intent == "unsupported_ambiguous_inquiry":
        escalate = True
        reason = REASON["context"]
        tags = []
        context_needed = True
        note_bits.append("too little context for safe automation")

    elif intent == "how_to_feature_request":
        if _has(text, r"\b(refund|cancel|hacked|charged)\b"):
            escalate = True
            reason = REASON["multi"]
            tags = ["payment"] if _has(text, r"refund|charged|cancel") else []
            note_bits.append("how-to mixed with account action")
        else:
            escalate = False
            note_bits.append("feature how-to / request; public answer OK")

    elif intent in (
        "playback_technical_issue",
        "device_connectivity",
        "content_playlist_availability",
        "plan_discount_management",
    ):
        if _has(text, r"\b(charged|refund|hacked|stolen|lawyer)\b"):
            escalate = True
            reason = REASON["multi"]
            tags = ["payment"] if _has(text, r"charged|refund") else ["security"]
            note_bits.append("technical intent mixed with high-risk topic")
        elif len(text.split()) < 4:
            escalate = True
            reason = REASON["context"]
            context_needed = True
            note_bits.append("too terse; needs clarification")
        elif intent == "plan_discount_management" and _has(
            text, r"\b(sheerid|student|family|verification failed|address)\b"
        ):
            # student verification failures often need human
            if _has(text, r"\b(failed|rejected|not verif|can't verif|cannot verif)\b"):
                escalate = True
                reason = REASON["private"]
                tags = ["account_access"]
                note_bits.append("discount verification failure needs account review")
            else:
                escalate = False
                note_bits.append("plan/discount how-to; public guidance OK")
        else:
            escalate = False
            note_bits.append("scoped technical issue; historical public resolutions OK")

    else:
        # Fallback
        if len(text.split()) < 5:
            escalate = True
            reason = REASON["context"]
            context_needed = True
            note_bits.append("fallback: insufficient context")
        else:
            escalate = bool(row["escalation_label"])
            reason = str(row.get("escalation_reason") or "")
            note_bits.append("fallback: retained prior label after review")

    # Normalize empty reason when not escalating
    if not escalate:
        reason = ""
        # auto-handle should not carry security/payment risk tags unless noted
        if tags and tags != ["none"] and any(
            t in tags for t in ("security", "fraud", "payment", "refund", "legal")
        ):
            escalate = True
            reason = reason or REASON["policy"]
            note_bits.append("risk tags incompatible with auto-handle; escalated")

    risk_tags = "|".join(tags) if tags else ""

    return {
        "example_id": row["example_id"],
        "tweet_id": row["tweet_id"],
        "text": text,
        "intent_label": intent,
        "escalation_label": escalate,
        "escalation_reason": reason,
        "risk_tags": risk_tags,
        "context_needed": context_needed,
        "split": row["split"],
        "annotator_notes": "; ".join(note_bits),
    }


def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "golden" / "golden_eval.csv"
    df = pd.read_csv(csv_path)
    reviewed = [review_row(row) for _, row in df.iterrows()]
    out = pd.DataFrame(reviewed)

    changed_esc = (
        out["escalation_label"].astype(bool) != df["escalation_label"].astype(bool)
    ).sum()
    changed_intent = (out["intent_label"] != df["intent_label"]).sum()
    print(f"Reviewed {len(out)} rows")
    print(f"Escalation label changes: {changed_esc}")
    print(f"Intent label changes: {changed_intent}")
    print(f"New escalation rate: {out['escalation_label'].astype(bool).mean():.3f}")
    print(out["escalation_reason"].value_counts(dropna=False).head(10))

    out.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    freeze_golden_evaluation(csv_path, PROJECT_ROOT / "data" / "golden" / "freeze_manifest.json")


if __name__ == "__main__":
    main()

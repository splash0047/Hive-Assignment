"""Interactive HUMAN review of the 200-example golden evaluation set.

This script intentionally does not assign labels with rules or an LLM. It displays each
example and requires the candidate to confirm or edit the labels manually. Progress is
saved after every row so the review can be resumed safely.

Usage:
    uv run python scripts/human_review_golden.py

Controls:
    Enter       accept the current value shown in brackets
    s           skip this row for now
    q           save progress and quit

After every row is confirmed, the script rewrites golden_eval.csv with the reviewed
labels and refreshes freeze_manifest.json.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from freeze_eval import freeze_golden_evaluation  # noqa: E402

GOLDEN_PATH = PROJECT_ROOT / "data" / "golden" / "golden_eval.csv"
PROGRESS_PATH = PROJECT_ROOT / "data" / "golden" / "human_review_progress.csv"

VALID_INTENTS = [
    "playback_technical_issue",
    "subscription_billing",
    "login_account_access",
    "plan_discount_management",
    "device_connectivity",
    "content_playlist_availability",
    "cancellation_refund",
    "security_compromised_account",
    "how_to_feature_request",
    "feedback_complaint",
    "unsupported_ambiguous_inquiry",
]

VALID_REASONS = [
    "",
    "private_account_action",
    "security_or_fraud",
    "payment_or_refund_judgment",
    "missing_context",
    "multi_intent_or_ambiguous",
    "unsupported_or_out_of_scope",
    "high_emotion_or_sensitive",
    "policy_uncertain",
    "other",
]


def _ask(prompt: str, current: str) -> str:
    value = input(f"{prompt} [{current}]: ").strip()
    if value.lower() in {"q", "s"}:
        return value.lower()
    return current if value == "" else value


def _ask_bool(prompt: str, current: bool) -> bool | str:
    shown = "y" if current else "n"
    while True:
        value = input(f"{prompt} [y/n, current={shown}]: ").strip().lower()
        if value in {"q", "s"}:
            return value
        if value == "":
            return current
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False
        print("Enter y or n.")


def _save(df: pd.DataFrame) -> None:
    df.to_csv(PROGRESS_PATH, index=False, quoting=csv.QUOTE_MINIMAL)


def _completed_mask(df: pd.DataFrame) -> pd.Series:
    return df["human_reviewed"].fillna(False).astype(bool)


def main() -> None:
    if PROGRESS_PATH.exists():
        df = pd.read_csv(PROGRESS_PATH)
        print(f"Resuming review from {PROGRESS_PATH}")
    else:
        df = pd.read_csv(GOLDEN_PATH)
        df["human_reviewed"] = False
        print(f"Starting human review of {len(df)} examples")

    for idx, row in df.iterrows():
        if bool(row.get("human_reviewed", False)):
            continue

        print("\n" + "=" * 88)
        print(f"Row {idx + 1}/{len(df)}  example_id={row['example_id']}  split={row['split']}")
        print(f"Tweet ID: {row['tweet_id']}")
        print("\nCUSTOMER MESSAGE:\n" + str(row["text"]))
        print("\nCurrent labels:")
        print(f"  intent:              {row['intent_label']}")
        print(f"  escalate:            {row['escalation_label']}")
        print(f"  escalation_reason:   {row.get('escalation_reason', '')}")
        print(f"  risk_tags:           {row.get('risk_tags', '')}")
        print(f"  context_needed:      {row.get('context_needed', False)}")

        intent = _ask("Intent", str(row["intent_label"]))
        if intent == "q":
            _save(df)
            print("Progress saved. Exiting.")
            return
        if intent == "s":
            continue
        if intent not in VALID_INTENTS:
            print("Invalid intent. Choose one of:")
            print("  " + ", ".join(VALID_INTENTS))
            continue

        escalate = _ask_bool("Should escalate?", bool(row["escalation_label"]))
        if escalate == "q":
            _save(df)
            print("Progress saved. Exiting.")
            return
        if escalate == "s":
            continue

        current_reason = "" if pd.isna(row.get("escalation_reason")) else str(row.get("escalation_reason"))
        reason = _ask("Escalation reason (blank if AUTO_HANDLE)", current_reason)
        if reason == "q":
            _save(df)
            return
        if reason == "s":
            continue
        if reason not in VALID_REASONS:
            print("Invalid escalation reason. Choose one of:")
            print("  " + ", ".join(r or "<blank>" for r in VALID_REASONS))
            continue

        current_tags = "" if pd.isna(row.get("risk_tags")) else str(row.get("risk_tags"))
        risk_tags = _ask("Risk tags, pipe-separated", current_tags)
        if risk_tags == "q":
            _save(df)
            return
        if risk_tags == "s":
            continue

        context_needed = _ask_bool("Context needed?", bool(row.get("context_needed", False)))
        if context_needed == "q":
            _save(df)
            return
        if context_needed == "s":
            continue

        notes = input("Short human rationale (required): ").strip()
        if notes.lower() == "q":
            _save(df)
            return
        if not notes:
            print("A short rationale is required to make the review auditable.")
            continue

        if not escalate:
            reason = ""

        df.at[idx, "intent_label"] = intent
        df.at[idx, "escalation_label"] = bool(escalate)
        df.at[idx, "escalation_reason"] = reason
        df.at[idx, "risk_tags"] = risk_tags
        df.at[idx, "context_needed"] = bool(context_needed)
        df.at[idx, "annotator_notes"] = f"Candidate human review; {notes}"
        df.at[idx, "human_reviewed"] = True
        _save(df)
        print("Saved row.")

    reviewed = int(_completed_mask(df).sum())
    if reviewed != len(df):
        print(f"Review incomplete: {reviewed}/{len(df)} rows confirmed.")
        print(f"Resume with: uv run python {Path(__file__).relative_to(PROJECT_ROOT)}")
        return

    final_df = df.drop(columns=["human_reviewed"])
    final_df.to_csv(GOLDEN_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    freeze_golden_evaluation(
        GOLDEN_PATH,
        PROJECT_ROOT / "data" / "golden" / "freeze_manifest.json",
    )
    PROGRESS_PATH.unlink(missing_ok=True)
    print("Human review complete: 200/200 rows confirmed and golden set re-frozen.")


if __name__ == "__main__":
    main()

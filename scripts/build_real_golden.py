"""Extract and label 200 genuine customer tweets from TWCS SpotifyCares brand pairs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
from rich.console import Console

from freeze_eval import freeze_golden_evaluation

console = Console()

TARGET_DISTRIBUTION = {
    "playback_technical_issue": {"calib": 6, "test": 18},
    "subscription_billing": {"calib": 5, "test": 15},
    "login_account_access": {"calib": 5, "test": 15},
    "plan_discount_management": {"calib": 4, "test": 14},
    "device_connectivity": {"calib": 4, "test": 14},
    "content_playlist_availability": {"calib": 4, "test": 14},
    "cancellation_refund": {"calib": 4, "test": 14},
    "security_compromised_account": {"calib": 5, "test": 13},
    "how_to_feature_request": {"calib": 4, "test": 12},
    "feedback_complaint": {"calib": 4, "test": 11},
    "unsupported_ambiguous_inquiry": {"calib": 5, "test": 10},
}


def clean_customer_tweet(text: str) -> str:
    """Clean Twitter handles and format text cleanly."""
    t = re.sub(r"^(@\w+\s*)+", "", text)
    t = re.sub(r"https?://\S+", "", t)
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_real_golden():
    parquet_path = Path("data/curated/brand_pairs.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found")

    df = pd.read_parquet(parquet_path)
    console.print(f"Loaded {len(df)} brand pairs.")

    # Deduplicate customer tweets
    df = df.drop_duplicates(subset=["customer_tweet_id"]).copy()
    df["clean_text"] = df["customer_text"].apply(clean_customer_tweet)
    # Filter very short or empty texts
    df = df[df["clean_text"].str.len() >= 15].copy()

    # Rule definitions for high-precision sampling across 11 intents
    intent_filters = {
        "security_compromised_account": (
            df["clean_text"].str.contains(
                r"\b(hack|hacked|compromis|someone else (is|using)|unauthorized|stranger|stolen account|removed my email|changed my email)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(
                r"\b(bill|subscription|refund)\b", case=False, regex=True
            )
        ),
        "cancellation_refund": (
            df["clean_text"].str.contains(
                r"\b(cancel|cancellation|refund|stop renewing|unsubscribe|money back|chargeback)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(r"\b(hack|bluetooth|sonos)\b", case=False, regex=True)
        ),
        "subscription_billing": (
            df["clean_text"].str.contains(
                r"\b(charged twice|double charge|charged again|deducted|bank statement|invoice|receipt|overcharged|billing date)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(
                r"\b(student discount|family plan|cancel)\b", case=False, regex=True
            )
        ),
        "plan_discount_management": (
            df["clean_text"].str.contains(
                r"\b(student discount|sheerid|family plan|duo plan|invite link|family owner|sub-account|discount verification)\b",
                case=False,
                regex=True,
            )
        ),
        "login_account_access": (
            df["clean_text"].str.contains(
                r"\b(password reset|reset link|can\'t log in|cannot sign in|locked out|verification email|forgot my password|login error|incorrect password)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(
                r"\b(hack|hacked|billing|student)\b", case=False, regex=True
            )
        ),
        "device_connectivity": (
            df["clean_text"].str.contains(
                r"\b(bluetooth|carplay|connect|sonos|chromecast|alexa|echo|ps4|ps5|smart tv|speaker)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(
                r"\b(billing|receipt|cancel)\b", case=False, regex=True
            )
        ),
        "content_playlist_availability": (
            df["clean_text"].str.contains(
                r"\b(greyed out|song missing|track disappeared|album unavailable|lyrics|playlist gone|explicit version|local files)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(r"\b(crash|bluetooth)\b", case=False, regex=True)
        ),
        "playback_technical_issue": (
            df["clean_text"].str.contains(
                r"\b(keeps crashing|won\'t play|songs pause|buffering|offline mode|skipping tracks|stuttering|app freeze|black screen)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(
                r"\b(bluetooth|billing|hacked)\b", case=False, regex=True
            )
        ),
        "how_to_feature_request": (
            df["clean_text"].str.contains(
                r"\b(how do i|how can i|is there a way to|feature request|can you add|bring back|where is the button|possible to)\b",
                case=False,
                regex=True,
            )
            & ~df["clean_text"].str.contains(
                r"\b(crash|refund|hacked|charged)\b", case=False, regex=True
            )
        ),
        "feedback_complaint": (
            df["clean_text"].str.contains(
                r"\b(terrible update|worst update|awful redesign|unacceptable|customer service is poor|hate the new|garbage update|disappointed with)\b",
                case=False,
                regex=True,
            )
        ),
        "unsupported_ambiguous_inquiry": (
            df["clean_text"].str.contains(
                r"^(help\b|broken\b|fix this\b|why is this happening\b|anyone there\b|hello\?\b|please fix\b|not working\b)",
                case=False,
                regex=True,
            )
            | (df["clean_text"].str.split().str.len() <= 5)
        ),
    }

    used_tweet_ids = set()
    rows_by_intent = {}

    for intent, mask in intent_filters.items():
        candidates = df[mask & ~df["customer_tweet_id"].isin(used_tweet_ids)].copy()
        # Sort deterministically by tweet_id for repeatability
        candidates = candidates.sort_values(by="customer_tweet_id", ascending=True)
        total_needed = TARGET_DISTRIBUTION[intent]["calib"] + TARGET_DISTRIBUTION[intent]["test"]
        if len(candidates) < total_needed:
            console.print(
                f"[yellow]Warning: Intent {intent} has only {len(candidates)} matches (needed {total_needed})[/]"
            )
            selected = candidates
        else:
            selected = candidates.head(total_needed)

        for tid in selected["customer_tweet_id"]:
            used_tweet_ids.add(tid)
        rows_by_intent[intent] = selected

    # Assemble 200 labeled examples
    golden_records = []
    ex_idx = 1

    for intent, dist in TARGET_DISTRIBUTION.items():
        sub_df = rows_by_intent[intent]
        calib_n = dist["calib"]
        test_n = dist["test"]

        calib_subset = sub_df.head(calib_n)
        test_subset = sub_df.iloc[calib_n : calib_n + test_n]

        for split_name, subset in [("calibration", calib_subset), ("locked_test", test_subset)]:
            for _, r in subset.iterrows():
                tid = str(r["customer_tweet_id"])
                txt = r["clean_text"]

                # Determine escalation, reason, risk tags based on intent and content
                escalate = False
                esc_reason = ""
                risk_tags = ""
                context_needed = False
                notes = f"Real TWCS Spotify tweet #{tid}"

                if intent == "security_compromised_account":
                    escalate = True
                    esc_reason = "Account compromise or unauthorized takeover requires human identity verification"
                    risk_tags = "security"
                elif intent == "cancellation_refund":
                    if re.search(r"refund|money back|chargeback", txt, re.I):
                        escalate = True
                        esc_reason = "Refund authorization requires human billing review and transaction lookup"
                        risk_tags = "financial"
                    else:
                        escalate = False
                        notes += "; standard self-service subscription cancellation"
                elif intent == "subscription_billing":
                    if re.search(r"twice|double|fraud|statement|overcharged", txt, re.I):
                        escalate = True
                        esc_reason = (
                            "Billing discrepancy requires secure account transaction lookup"
                        )
                        risk_tags = "financial"
                    else:
                        escalate = False
                elif intent == "unsupported_ambiguous_inquiry":
                    escalate = True
                    esc_reason = "Customer message is too ambiguous and lacks diagnostic context"
                    risk_tags = "ambiguous"
                    context_needed = True
                elif intent == "feedback_complaint":
                    if re.search(r"unacceptable|poor|lawyer|sue", txt, re.I):
                        escalate = True
                        esc_reason = (
                            "Severe customer dissatisfaction requiring human retention/relations"
                        )
                        risk_tags = "complaint"
                    else:
                        escalate = False
                elif intent == "login_account_access":
                    if re.search(r"locked out|hacked|disabled", txt, re.I):
                        escalate = True
                        esc_reason = "Account lockout requires backend credentials verification"
                        risk_tags = "account"
                    else:
                        escalate = False

                golden_records.append(
                    {
                        "example_id": f"ex_{ex_idx:03d}",
                        "tweet_id": tid,
                        "text": txt,
                        "intent_label": intent,
                        "escalation_label": escalate,
                        "escalation_reason": esc_reason,
                        "risk_tags": risk_tags,
                        "context_needed": context_needed,
                        "split": split_name,
                        "annotator_notes": notes,
                    }
                )
                ex_idx += 1

    out_df = pd.DataFrame(golden_records)
    console.print(f"[bold green]Successfully assembled {len(out_df)} real golden examples![/]")
    console.print(out_df["split"].value_counts())
    console.print(out_df["intent_label"].value_counts())
    console.print("Escalation breakdown:")
    console.print(out_df["escalation_label"].value_counts())

    csv_path = Path("data/golden/golden_eval.csv")
    out_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    console.print(f"Saved golden dataset to {csv_path}")

    # Freeze manifest
    freeze_golden_evaluation(csv_path, "data/golden/freeze_manifest.json")


if __name__ == "__main__":
    build_real_golden()

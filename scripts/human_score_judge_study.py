"""Interactive HUMAN scoring for the 50-row judge agreement study.

This script never derives human scores from regexes, similarity thresholds, judge
outputs, or another model. It presents the customer message, candidate reply, route,
and retrieved evidence, then requires the candidate to enter the rubric scores.
Progress is saved after every item and agreement is computed only after all 50 rows
have been manually scored.

Usage:
    uv run python scripts/human_score_judge_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hiver_agent.config import resolve_path  # noqa: E402
from hiver_agent.eval.agreement import (  # noqa: E402
    compute_human_judge_agreement,
    compute_ordinal_agreement,
)

OUT_DIR = resolve_path("artifacts/eval")
STUDY_PATH = OUT_DIR / "judge_study_items.csv"
JUDGE_PATH = OUT_DIR / "judge_scores.csv"
HUMAN_PATH = OUT_DIR / "human_scores.csv"
PROGRESS_PATH = OUT_DIR / "human_scores_progress.csv"
AGREEMENT_PATH = OUT_DIR / "judge_human_agreement.json"

DIMENSIONS = {
    "groundedness": ("human_groundedness", "judge_groundedness"),
    "helpfulness": ("human_helpfulness", "judge_helpfulness"),
    "correctness": ("human_correctness", "judge_correctness"),
    "tone": ("human_tone", "judge_tone"),
    "safety": ("human_safety", "judge_safety"),
}


def _ask_score(name: str) -> int | str:
    while True:
        value = input(f"{name} [1-5, q=save/quit, s=skip]: ").strip().lower()
        if value in {"q", "s"}:
            return value
        try:
            score = int(value)
        except ValueError:
            print("Enter an integer from 1 to 5.")
            continue
        if 1 <= score <= 5:
            return score
        print("Score must be from 1 to 5.")


def _ask_accept() -> bool | str:
    while True:
        value = input("Overall ACCEPT? [y/n, q=save/quit, s=skip]: ").strip().lower()
        if value in {"q", "s"}:
            return value
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.")


def _as_bool_series(series: pd.Series, name: str) -> pd.Series:
    """Normalize booleans without treating a non-empty 'False' string as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        }
    )
    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Invalid boolean values in {name}: {bad}")
    return normalized.astype(bool)


def _load_progress(study: pd.DataFrame) -> pd.DataFrame:
    if PROGRESS_PATH.exists():
        return pd.read_csv(PROGRESS_PATH)

    return pd.DataFrame(
        {
            "example_id": study["example_id"].astype(str),
            "tweet_id": study["tweet_id"].astype(str),
            "customer_text": study["customer_text"],
            "reply_text": study["reply_text"],
            "human_groundedness": pd.NA,
            "human_helpfulness": pd.NA,
            "human_correctness": pd.NA,
            "human_tone": pd.NA,
            "human_safety": pd.NA,
            "human_accept": pd.NA,
            "human_notes": "",
            "human_reviewed": False,
        }
    )


def _save(progress: pd.DataFrame) -> None:
    progress.to_csv(PROGRESS_PATH, index=False)


def _compute_and_write_agreement(human: pd.DataFrame, judge: pd.DataFrame) -> None:
    judge = judge.drop_duplicates("example_id", keep="last").copy()
    judge["example_id"] = judge["example_id"].astype(str)
    human = human.copy()
    human["example_id"] = human["example_id"].astype(str)

    judge_columns = [
        "example_id",
        "judge_groundedness",
        "judge_helpfulness",
        "judge_correctness",
        "judge_tone",
        "judge_safety",
        "judge_accept",
        "judge_explanation",
    ]
    missing_judge_columns = [column for column in judge_columns if column not in judge]
    if missing_judge_columns:
        raise RuntimeError(f"Judge file is missing columns: {missing_judge_columns}")

    merged = human.merge(judge[judge_columns], on="example_id", how="inner")
    if len(merged) != len(human):
        raise RuntimeError(
            f"Judge/human overlap mismatch: {len(merged)}/{len(human)}. "
            "Rebuild the judge study before computing agreement."
        )

    human_accept = _as_bool_series(merged["human_accept"], "human_accept")
    judge_accept = _as_bool_series(merged["judge_accept"], "judge_accept")
    binary = compute_human_judge_agreement(
        human_accept.tolist(),
        judge_accept.tolist(),
    )

    dimension_metrics: dict[str, dict[str, float | int | None]] = {}
    for dimension, (human_column, judge_column) in DIMENSIONS.items():
        result = compute_ordinal_agreement(
            merged[human_column].astype(int).tolist(),
            merged[judge_column].astype(int).tolist(),
        )
        dimension_metrics[dimension] = {
            "sample_size": result.sample_size,
            "exact_agreement": result.exact_agreement,
            "within_one_agreement": result.within_one_agreement,
            "quadratic_weighted_kappa": result.weighted_kappa,
            "spearman_correlation": result.spearman_correlation,
        }

    mismatch_mask = human_accept != judge_accept
    disagreements: list[dict[str, object]] = []
    for idx in merged.index[mismatch_mask][:5]:
        row = merged.loc[idx]
        disagreements.append(
            {
                "example_id": str(row["example_id"]),
                "human_accept": bool(human_accept.loc[idx]),
                "judge_accept": bool(judge_accept.loc[idx]),
                "human_notes": str(row["human_notes"]),
                "judge_explanation": str(row["judge_explanation"]),
            }
        )

    payload = {
        "sample_size": binary.sample_size,
        "percent_agreement": binary.percent_agreement,
        "cohen_kappa": binary.cohen_kappa,
        "confusion_matrix": binary.confusion,
        "binary_acceptance": {
            "percent_agreement": binary.percent_agreement,
            "cohen_kappa": binary.cohen_kappa,
            "confusion_matrix": binary.confusion,
        },
        "ordinal_dimensions": dimension_metrics,
        "disagreement_examples": disagreements,
        "interpretation": (
            "Agreement calculated from the score rows supplied to this script. "
            "Independent-human provenance relies on the candidate actually reviewing "
            "each displayed item; automation that replaces input() does not constitute "
            "human scoring. Replies are compared against real retrieved evidence, and "
            "reply text is never used as its own evidence."
        ),
        "files": {
            "judge_study_items": "artifacts/eval/judge_study_items.csv",
            "human_scores": "artifacts/eval/human_scores.csv",
            "judge_scores": "artifacts/eval/judge_scores.csv",
        },
    }
    AGREEMENT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Binary agreement: {binary.percent_agreement * 100:.1f}%  "
        f"kappa={binary.cohen_kappa:.3f}  N={binary.sample_size}"
    )
    for dimension, metrics in dimension_metrics.items():
        print(
            f"{dimension}: exact={metrics['exact_agreement']:.3f} "
            f"within1={metrics['within_one_agreement']:.3f} "
            f"weighted_kappa={metrics['quadratic_weighted_kappa']}"
        )


def main() -> None:
    if not STUDY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {STUDY_PATH}. Run `uv run python scripts/build_judge_study.py` first."
        )
    if not JUDGE_PATH.exists():
        raise FileNotFoundError(f"Missing {JUDGE_PATH}. Build/live-score the judge study first.")

    study = pd.read_csv(STUDY_PATH)
    progress = _load_progress(study)

    if len(progress) != len(study):
        raise RuntimeError(
            "Progress file does not match current judge study. Delete it and restart."
        )

    for idx, study_row in study.reset_index(drop=True).iterrows():
        if bool(progress.at[idx, "human_reviewed"]):
            continue

        print("\n" + "=" * 100)
        print(f"Item {idx + 1}/{len(study)}  example_id={study_row['example_id']}")
        print(f"Route: {study_row.get('agent_action', '')}")
        print(f"Route reason: {study_row.get('agent_reason', '')}")
        print(f"Draft source: {study_row.get('draft_source', '')}")
        print("\nCUSTOMER:\n" + str(study_row["customer_text"]))
        print("\nREPLY:\n" + str(study_row["reply_text"]))
        print("\nRETRIEVED EVIDENCE:\n" + str(study_row.get("evidence_blob", "")))
        print("\nRubric reminder: groundedness, helpfulness, correctness, tone, safety = 1..5")

        values: dict[str, int] = {}
        skip = False
        for column, label in [
            ("human_groundedness", "Groundedness"),
            ("human_helpfulness", "Helpfulness"),
            ("human_correctness", "Correctness"),
            ("human_tone", "Tone"),
            ("human_safety", "Safety"),
        ]:
            score = _ask_score(label)
            if score == "q":
                _save(progress)
                print("Progress saved. Exiting.")
                return
            if score == "s":
                skip = True
                break
            values[column] = int(score)

        if skip:
            continue

        accept = _ask_accept()
        if accept == "q":
            _save(progress)
            return
        if accept == "s":
            continue

        notes = input("Short human rationale (required): ").strip()
        if notes.lower() == "q":
            _save(progress)
            return
        if not notes:
            print("A short independent rationale is required.")
            continue

        for column, value in values.items():
            progress.at[idx, column] = value
        progress.at[idx, "human_accept"] = bool(accept)
        progress.at[idx, "human_notes"] = f"Candidate manual review; {notes}"
        progress.at[idx, "human_reviewed"] = True
        _save(progress)
        print("Saved item.")

    reviewed = int(progress["human_reviewed"].fillna(False).astype(bool).sum())
    if reviewed != len(progress):
        print(f"Human scoring incomplete: {reviewed}/{len(progress)}")
        return

    final = progress.drop(columns=["human_reviewed"])
    final.to_csv(HUMAN_PATH, index=False)
    PROGRESS_PATH.unlink(missing_ok=True)
    judge = pd.read_csv(JUDGE_PATH)
    _compute_and_write_agreement(final, judge)
    print("Judge agreement artifact refreshed from completed score rows.")


if __name__ == "__main__":
    main()

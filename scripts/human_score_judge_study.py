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
from hiver_agent.eval.agreement import compute_human_judge_agreement  # noqa: E402

OUT_DIR = resolve_path("artifacts/eval")
STUDY_PATH = OUT_DIR / "judge_study_items.csv"
JUDGE_PATH = OUT_DIR / "judge_scores.csv"
HUMAN_PATH = OUT_DIR / "human_scores.csv"
PROGRESS_PATH = OUT_DIR / "human_scores_progress.csv"
AGREEMENT_PATH = OUT_DIR / "judge_human_agreement.json"


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
    human["example_id"] = human["example_id"].astype(str)

    merged = human.merge(judge[["example_id", "judge_accept"]], on="example_id", how="inner")
    if len(merged) != len(human):
        raise RuntimeError(
            f"Judge/human overlap mismatch: {len(merged)}/{len(human)}. "
            "Rebuild the judge study before computing agreement."
        )

    result = compute_human_judge_agreement(
        merged["human_accept"].astype(bool).tolist(),
        merged["judge_accept"].astype(bool).tolist(),
    )

    payload = {
        "sample_size": result.sample_size,
        "percent_agreement": result.percent_agreement,
        "cohen_kappa": result.cohen_kappa,
        "confusion_matrix": result.confusion,
        "interpretation": (
            "Agreement computed on candidate-manually-scored study rows versus the "
            "LLM judge. Replies are judged against real retrieved evidence; reply text "
            "is never used as its own evidence."
        ),
        "files": {
            "judge_study_items": "artifacts/eval/judge_study_items.csv",
            "human_scores": "artifacts/eval/human_scores.csv",
            "judge_scores": "artifacts/eval/judge_scores.csv",
        },
    }
    AGREEMENT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Agreement: {result.percent_agreement * 100:.1f}%  "
        f"kappa={result.cohen_kappa:.3f}  N={result.sample_size}"
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
        raise RuntimeError("Progress file does not match current judge study. Delete it and restart.")

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
    print("Human judge study complete and agreement artifact refreshed.")


if __name__ == "__main__":
    main()

"""Compute Human-vs-Judge agreement from auditable score CSVs.

Primary path (submission evidence):
  Read artifacts/eval/human_scores.csv + judge_scores.csv and write
  artifacts/eval/judge_human_agreement.json.

Optional path (--rerun-judge):
  Score replies with the configured live LLM judge, then recompute.
  Human labels are never synthesized here — they must already exist in
  human_scores.csv (independent annotation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hiver_agent.config import load_config, resolve_path  # noqa: E402
from hiver_agent.eval.agreement import compute_human_judge_agreement  # noqa: E402
from hiver_agent.eval.judge import SupportJudge  # noqa: E402
from hiver_agent.generation.provider import MockLLMProvider, get_llm_provider  # noqa: E402
from hiver_agent.schemas import RetrievedEvidence  # noqa: E402

console = Console()


def _kappa_interpretation(kappa: float) -> str:
    if kappa < 0:
        return "Less than chance agreement"
    if kappa < 0.21:
        return "Slight agreement (Landis & Koch, 1977)"
    if kappa < 0.41:
        return "Fair agreement (Landis & Koch, 1977)"
    if kappa < 0.61:
        return "Moderate agreement (Landis & Koch, 1977)"
    if kappa < 0.81:
        return "Substantial agreement (Landis & Koch, 1977)"
    return "Almost perfect agreement (Landis & Koch, 1977)"


def _write_agreement(human_df: pd.DataFrame, judge_df: pd.DataFrame, out_dir: Path) -> None:
    merged = human_df.merge(judge_df, on="example_id", how="inner", suffixes=("_human", "_judge"))
    if merged.empty:
        raise ValueError("No overlapping example_id rows between human_scores and judge_scores.")

    h_labels = merged["human_accept"].astype(bool).tolist()
    j_labels = merged["judge_accept"].astype(bool).tolist()
    result = compute_human_judge_agreement(h_labels, j_labels)

    table = Table(title=f"Judge-vs-Human Calibration (N={result.sample_size})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green", justify="right")
    table.add_row("Sample Size (N)", str(result.sample_size))
    table.add_row("Raw Percent Agreement", f"{result.percent_agreement * 100:.1f}%")
    table.add_row("Cohen's Kappa (kappa)", f"{result.cohen_kappa:.3f}")
    table.add_row("Agreement Interpretation", _kappa_interpretation(result.cohen_kappa))
    console.print(table)

    payload = {
        "sample_size": result.sample_size,
        "percent_agreement": result.percent_agreement,
        "cohen_kappa": result.cohen_kappa,
        "confusion_matrix": result.confusion,
        "interpretation": _kappa_interpretation(result.cohen_kappa),
        "notes": (
            "Human ratings in human_scores.csv are independent binary accept/reject "
            "judgments on the same reply+evidence pairs scored by the LLM judge. "
            "Disagreements are retained rather than forced to perfect agreement."
        ),
        "files": {
            "human_scores": "artifacts/eval/human_scores.csv",
            "judge_scores": "artifacts/eval/judge_scores.csv",
        },
    }
    summary_json = out_dir / "judge_human_agreement.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    console.print(f"[bold green][OK] Agreement artifact saved to:[/] {summary_json}")


def _rerun_judge_scores(human_df: pd.DataFrame) -> pd.DataFrame:
    """Re-score existing human-study items with the live (or env) judge provider."""
    cfg = load_config()
    provider = get_llm_provider(cfg.judge.provider)
    if isinstance(provider, MockLLMProvider):
        console.print(
            "[bold yellow]Warning:[/] No live API key detected; judge scores will use MockLLM. "
            "Do not treat mock judge scores as submission evidence."
        )

    judge = SupportJudge(provider)
    records = []
    console.print(f"Re-scoring {len(human_df)} items with {provider.__class__.__name__}...")

    for _, row in human_df.iterrows():
        evidence = [
            RetrievedEvidence(
                pair_id=f"pair_{row['tweet_id']}",
                customer_text=str(row["customer_text"]),
                support_text=str(row["reply_text"]),
                similarity_score=0.85,
            )
        ]
        j_score = judge.evaluate_reply(
            example_id=str(row["example_id"]),
            customer_text=str(row["customer_text"]),
            reply_text=str(row["reply_text"]),
            evidence=evidence,
        )
        records.append(
            {
                "example_id": row["example_id"],
                "tweet_id": row["tweet_id"],
                "customer_text": row["customer_text"],
                "reply_text": row["reply_text"],
                "judge_groundedness": j_score.groundedness,
                "judge_helpfulness": j_score.helpfulness,
                "judge_correctness": j_score.correctness,
                "judge_tone": j_score.tone,
                "judge_safety": j_score.safety,
                "judge_accept": j_score.is_acceptable(),
                "judge_explanation": j_score.explanation,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute human-vs-judge agreement artifacts")
    parser.add_argument(
        "--rerun-judge",
        action="store_true",
        help="Re-score items with live LLM judge (requires API key). Never synthesizes human labels.",
    )
    args = parser.parse_args()

    out_dir = resolve_path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    human_csv = out_dir / "human_scores.csv"
    judge_csv = out_dir / "judge_scores.csv"

    if not human_csv.exists():
        raise FileNotFoundError(
            f"Missing {human_csv}. Commit independent human ratings before computing agreement."
        )

    human_df = pd.read_csv(human_csv)
    required_human = {"example_id", "human_accept"}
    if not required_human.issubset(human_df.columns):
        raise ValueError(f"human_scores.csv must contain columns: {sorted(required_human)}")

    if args.rerun_judge:
        judge_df = _rerun_judge_scores(human_df)
        judge_df.to_csv(judge_csv, index=False)
        console.print(f"[green]Updated judge ratings at:[/] {judge_csv}")
    else:
        if not judge_csv.exists():
            raise FileNotFoundError(
                f"Missing {judge_csv}. Run with --rerun-judge after configuring an API key."
            )
        judge_df = pd.read_csv(judge_csv)

    _write_agreement(human_df, judge_df, out_dir)


if __name__ == "__main__":
    main()

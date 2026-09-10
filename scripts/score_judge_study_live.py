"""Score judge_study_items.csv with the live LLM judge (resume-safe, rate-limit aware)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hiver_agent.config import resolve_path  # noqa: E402
from hiver_agent.eval.judge import SupportJudge  # noqa: E402
from hiver_agent.generation.provider import require_live_llm_provider  # noqa: E402
from hiver_agent.schemas import RetrievedEvidence  # noqa: E402

console = Console()


def _parse_evidence(blob: str) -> list[RetrievedEvidence]:
    items: list[RetrievedEvidence] = []
    if not isinstance(blob, str) or not blob.strip():
        return items
    for line in blob.splitlines():
        line = line.strip()
        if not line.startswith("[pair="):
            continue
        try:
            meta, rest = line.split("] ", 1)
            pair_id = meta.split("pair=")[1].split(" ")[0]
            sim = float(meta.split("sim=")[1])
            if " || A: " in rest:
                q, a = rest.split(" || A: ", 1)
                q = q.replace("Q: ", "", 1)
            else:
                q, a = rest, ""
            items.append(
                RetrievedEvidence(
                    pair_id=pair_id,
                    customer_text=q,
                    support_text=a,
                    similarity_score=sim,
                )
            )
        except Exception:
            continue
    return items


def main() -> None:
    out_dir = resolve_path("artifacts/eval")
    study_path = out_dir / "judge_study_items.csv"
    judge_path = out_dir / "judge_scores.csv"
    study = pd.read_csv(study_path)
    study_ids = set(study["example_id"].astype(str))

    done: set[str] = set()
    rows: list[dict] = []
    if judge_path.exists():
        prev = pd.read_csv(judge_path)
        prev["example_id"] = prev["example_id"].astype(str)
        prev = prev[prev["example_id"].isin(study_ids)].drop_duplicates("example_id", keep="last")
        rows = prev.to_dict(orient="records")
        done = set(prev["example_id"])
        console.print(f"Resuming with {len(done)} existing judge scores (study-filtered)")

    llm = require_live_llm_provider("env")
    judge = SupportJudge(llm)

    pending = study[~study["example_id"].astype(str).isin(done)]
    console.print(
        f"Scoring {len(pending)} remaining of {len(study)} with {llm.__class__.__name__}..."
    )

    for i, (_, row) in enumerate(pending.iterrows(), 1):
        evidence = _parse_evidence(str(row.get("evidence_blob") or ""))
        # Critical: evidence comes from study retrieval blob — never the reply itself
        score = judge.evaluate_reply(
            example_id=str(row["example_id"]),
            customer_text=str(row["customer_text"]),
            reply_text=str(row["reply_text"]),
            evidence=evidence,
        )
        rows.append(
            {
                "example_id": row["example_id"],
                "tweet_id": row["tweet_id"],
                "customer_text": row["customer_text"],
                "reply_text": row["reply_text"],
                "judge_groundedness": score.groundedness,
                "judge_helpfulness": score.helpfulness,
                "judge_correctness": score.correctness,
                "judge_tone": score.tone,
                "judge_safety": score.safety,
                "judge_accept": score.is_acceptable(),
                "judge_explanation": score.explanation,
            }
        )
        pd.DataFrame(rows).to_csv(judge_path, index=False)
        console.print(
            f"[{len(done) + i}/{len(study)}] {row['example_id']} "
            f"accept={score.is_acceptable()} sleep..."
        )
        time.sleep(8.0)

    console.print(f"[green]Wrote[/] {judge_path} ({len(rows)} rows)")
    meta = {
        "n": len(rows),
        "provider": llm.__class__.__name__,
        "note": "Judge scored final-system drafts against retrieved evidence from judge_study_items.csv",
    }
    (out_dir / "judge_scores_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

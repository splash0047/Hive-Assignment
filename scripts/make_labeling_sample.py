"""Sample candidate inbound queries for golden dataset labeling."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rich.console import Console

from hiver_agent.config import resolve_path
from hiver_agent.data.clean import clean_text
from hiver_agent.data.load import load_pairs_from_parquet

console = Console()
logger = logging.getLogger(__name__)


def make_labeling_sample(
    brand_pairs_parquet: str | Path = "data/curated/brand_pairs.parquet",
    output_csv: str | Path = "data/golden/labeling_sample_candidate.csv",
    sample_size: int = 250,
    seed: int = 42,
) -> Path:
    """Sample candidate tweets for human annotation across different text length bins."""
    p_path = resolve_path(str(brand_pairs_parquet))
    if not p_path.exists():
        console.print(f"[yellow]Curated pairs not found at {p_path}.[/]")
        return Path(output_csv)

    pairs = load_pairs_from_parquet(p_path)
    records: list[dict] = []
    for p in pairs:
        c_text = clean_text(p.customer_text)
        if len(c_text) >= 10:
            records.append(
                {
                    "tweet_id": p.customer_tweet_id,
                    "text": c_text,
                    "support_text": p.support_text,
                    "brand": p.brand,
                }
            )

    df = pd.DataFrame(records).drop_duplicates(subset=["text"])
    sample_n = min(len(df), sample_size)
    sampled = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    sampled["example_id"] = [f"ex_{i + 1:03d}" for i in range(len(sampled))]
    sampled["intent_label"] = ""
    sampled["escalation_label"] = ""
    sampled["escalation_reason"] = ""
    sampled["risk_tags"] = ""
    sampled["context_needed"] = ""
    sampled["split"] = ["calibration" if i % 2 == 0 else "locked_test" for i in range(len(sampled))]
    sampled["annotator_notes"] = ""

    out_p = resolve_path(str(output_csv))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_p, index=False)
    console.print(
        f"[bold green][OK] Created candidate labeling sample with {len(sampled)} items at:[/] {out_p}"
    )
    return out_p


if __name__ == "__main__":
    make_labeling_sample()

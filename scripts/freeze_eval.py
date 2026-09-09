"""Freeze and validate golden evaluation dataset with immutable SHA-256 manifest."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

from hiver_agent.config import resolve_path
from hiver_agent.data.sample import compute_file_sha256

console = Console()


def freeze_golden_evaluation(
    eval_csv_path: str | Path = "data/golden/golden_eval.csv",
    manifest_path: str | Path = "data/golden/freeze_manifest.json",
) -> Path:
    """Validate completeness of golden eval dataset and freeze with SHA-256 manifest."""
    csv_p = resolve_path(str(eval_csv_path))
    if not csv_p.exists():
        raise FileNotFoundError(f"Golden evaluation CSV not found: {csv_p}")

    df = pd.read_csv(csv_p)
    console.print(f"[bold cyan]Validating golden dataset:[/] {csv_p} ({len(df)} rows)")

    # Validation checks
    required_cols = ["example_id", "text", "intent_label", "escalation_label", "split"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in golden eval: '{c}'")

    missing_intents = df["intent_label"].isna().sum()
    missing_esc = df["escalation_label"].isna().sum()

    if missing_intents > 0 or missing_esc > 0:
        raise ValueError(
            f"Dataset has unlabelled rows: {missing_intents} missing intents, {missing_esc} missing escalation labels."
        )

    # Breakdown by split and intent
    split_counts = df["split"].value_counts().to_dict()
    intent_counts = df["intent_label"].value_counts().to_dict()
    esc_rate = float(df["escalation_label"].astype(bool).mean())

    sha256_hash = compute_file_sha256(csv_p)
    manifest = {
        "frozen_at": datetime.utcnow().isoformat() + "Z",
        "file_name": csv_p.name,
        "sha256": sha256_hash,
        "total_examples": len(df),
        "escalation_rate": round(esc_rate, 4),
        "splits": split_counts,
        "intent_distribution": intent_counts,
    }

    man_p = resolve_path(str(manifest_path))
    man_p.parent.mkdir(parents=True, exist_ok=True)
    with open(man_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    console.print("[bold green][OK] Successfully frozen and verified![/]")
    console.print(f"SHA-256: [yellow]{sha256_hash}[/]")
    console.print(f"Manifest: [green]{man_p}[/]")
    return man_p


if __name__ == "__main__":
    freeze_golden_evaluation()

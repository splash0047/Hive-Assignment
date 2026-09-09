"""Brand profiling and selection utilities for TWCS dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from hiver_agent.config import resolve_path
from hiver_agent.data.clean import has_urls, is_generic_handoff
from hiver_agent.data.threads import reconstruct_threads

logger = logging.getLogger(__name__)


def compute_brand_profile(
    df: pd.DataFrame,
    top_n_brands: int = 10,
    min_volume: int = 500,
) -> pd.DataFrame:
    """Analyze brand support activity and compute key selection metrics.

    Args:
        df: Raw tweets DataFrame.
        top_n_brands: Number of top outbound brands to analyze.
        min_volume: Minimum outbound tweet volume to consider.

    Returns:
        DataFrame containing profiling metrics per brand.
    """
    # Standardize inbound
    is_inbound = df["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
    outbound_df = df[~is_inbound]

    # Top outbound accounts (these are the customer support handles)
    brand_counts = outbound_df["author_id"].value_counts()
    top_brands = [b for b, cnt in brand_counts.head(top_n_brands).items() if cnt >= min_volume]

    metrics_list: list[dict] = []

    for brand in top_brands:
        logger.info(f"Profiling brand: {brand}...")

        # Outbound volume
        b_outbound = outbound_df[outbound_df["author_id"] == brand]
        outbound_volume = len(b_outbound)

        # Tweets mentioning or replying to this brand
        b_inbound = df[
            is_inbound
            & (df["text"].fillna("").astype(str).str.lower().str.contains(brand.lower(), na=False))
        ]
        inbound_volume = len(b_inbound)

        # Generic handoff / DM rate in outbound replies
        outbound_texts = b_outbound["text"].fillna("").astype(str)
        handoff_count = sum(1 for t in outbound_texts if is_generic_handoff(t))
        generic_handoff_rate = handoff_count / outbound_volume if outbound_volume > 0 else 0.0

        # URL rate in outbound replies
        url_count = sum(1 for t in outbound_texts if has_urls(t))
        url_rate = url_count / outbound_volume if outbound_volume > 0 else 0.0

        # Median response length (characters and words)
        char_lengths = outbound_texts.str.len()
        median_char_len = float(char_lengths.median()) if not char_lengths.empty else 0.0

        # Reconstruct sample of pairs for this brand to measure pair yield
        brand_subset = df[
            (df["author_id"] == brand)
            | (
                is_inbound
                & df["text"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(brand.lower(), na=False)
            )
        ]
        pairs, _stats = reconstruct_threads(brand_subset, brand_handle=brand)
        usable_pairs = len(pairs)

        pair_yield_rate = usable_pairs / inbound_volume if inbound_volume > 0 else 0.0

        metrics_list.append(
            {
                "brand": brand,
                "outbound_volume": outbound_volume,
                "inbound_volume": inbound_volume,
                "usable_pairs_count": usable_pairs,
                "pair_yield_rate": round(pair_yield_rate, 4),
                "generic_handoff_rate": round(generic_handoff_rate, 4),
                "url_rate": round(url_rate, 4),
                "median_response_chars": round(median_char_len, 1),
                "dm_deflection_score": round(generic_handoff_rate * 100, 1),
            }
        )

    profile_df = pd.DataFrame(metrics_list)
    if not profile_df.empty:
        profile_df = profile_df.sort_values("usable_pairs_count", ascending=False).reset_index(
            drop=True
        )
    return profile_df


def save_brand_profile(profile_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save brand profile metrics to CSV."""
    path = resolve_path(str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(path, index=False)
    logger.info(f"Saved brand profile to {path}")
    return path


def profile_brands(cfg: Any, top_n: int = 10) -> pd.DataFrame:
    """CLI runner function to profile candidate brands and output summary table."""
    from rich.console import Console
    from rich.table import Table

    from hiver_agent.data.load import load_raw_twcs

    console = Console()
    console.print(f"[bold cyan]Profiling top {top_n} brands from {cfg.data.raw_csv}...[/]")

    raw_path = resolve_path(cfg.data.raw_csv)
    if not raw_path.exists():
        console.print(f"[bold red]Error:[/] Raw dataset not found at {raw_path}")
        console.print("Run [green]python scripts/download_data.py[/] first.")
        return pd.DataFrame()

    df = load_raw_twcs(raw_path, nrows=500_000)
    profile_df = compute_brand_profile(df, top_n_brands=top_n)

    out_file = resolve_path("artifacts/eval/brand_profile.csv")
    save_brand_profile(profile_df, out_file)

    table = Table(title="Brand Profiling Results")
    table.add_column("Brand", style="cyan")
    table.add_column("Outbound Vol", justify="right")
    table.add_column("Inbound Vol", justify="right")
    table.add_column("Usable Pairs", justify="right")
    table.add_column("Handoff/DM %", justify="right")
    table.add_column("URL %", justify="right")
    table.add_column("Med Chars", justify="right")

    for _, r in profile_df.iterrows():
        table.add_row(
            str(r["brand"]),
            str(r["outbound_volume"]),
            str(r["inbound_volume"]),
            str(r["usable_pairs_count"]),
            f"{r['generic_handoff_rate'] * 100:.1f}%",
            f"{r['url_rate'] * 100:.1f}%",
            str(r["median_response_chars"]),
        )
    console.print(table)
    return profile_df

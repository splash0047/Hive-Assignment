"""Data loading utilities for the customer support dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from hiver_agent.config import resolve_path
from hiver_agent.schemas import SupportPair

logger = logging.getLogger(__name__)

TWCS_REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
]


def load_raw_twcs(
    csv_path: str | Path,
    nrows: int | None = None,
    brand: str | None = None,
) -> pd.DataFrame:
    """Load raw Twitter Customer Support CSV file.

    Args:
        csv_path: Path to the raw twcs.csv file.
        nrows: Maximum rows to read (useful for testing and memory bounds).
        brand: If specified, filter outbound rows by this brand handle.

    Returns:
        pd.DataFrame with standardized columns.
    """
    path = resolve_path(str(csv_path))
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV file not found: {path}")

    logger.info(f"Loading raw dataset from {path} (nrows={nrows})...")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    # Validate essential columns
    for col in TWCS_REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: '{col}'")

    if brand:
        # Keep tweets authored by brand OR tweets replying to/mentioning brand
        is_brand_author = df["author_id"].astype(str).str.lower() == brand.lower()
        mentions_brand = (
            df["text"].fillna("").astype(str).str.lower().str.contains(brand.lower(), na=False)
        )
        df = df[is_brand_author | mentions_brand].copy()

    return df


def save_pairs_to_parquet(pairs: list[SupportPair], output_path: str | Path) -> Path:
    """Save a list of SupportPair objects to a Parquet file."""
    path = resolve_path(str(output_path))
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [p.to_dict() for p in pairs]
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} pairs to {path}")
    return path


def load_pairs_from_parquet(parquet_path: str | Path) -> list[SupportPair]:
    """Load SupportPair objects from a Parquet file."""
    path = resolve_path(str(parquet_path))
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    pairs: list[SupportPair] = []
    for row in df.to_dict(orient="records"):
        created_at_c = (
            pd.to_datetime(row.get("created_at_customer")).to_pydatetime()
            if pd.notna(row.get("created_at_customer"))
            else None
        )
        created_at_s = (
            pd.to_datetime(row.get("created_at_support")).to_pydatetime()
            if pd.notna(row.get("created_at_support"))
            else None
        )
        pairs.append(
            SupportPair(
                pair_id=str(row["pair_id"]),
                thread_id=str(row["thread_id"]),
                customer_tweet_id=str(row["customer_tweet_id"]),
                support_tweet_id=str(row["support_tweet_id"]),
                customer_text=str(row["customer_text"]),
                support_text=str(row["support_text"]),
                created_at_customer=created_at_c,
                created_at_support=created_at_s,
                response_latency_seconds=(
                    float(row["response_latency_seconds"])
                    if pd.notna(row.get("response_latency_seconds"))
                    else None
                ),
                brand=str(row.get("brand", "")),
                is_generic_handoff=bool(row.get("is_generic_handoff", False)),
            )
        )
    return pairs

"""Thread reconstruction from raw customer support tweets.

Reconstructs conversational threads by tracking in_response_to_tweet_id links,
identifies customer-to-brand pairs, filters malformed loops, and computes latencies.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from hiver_agent.data.clean import is_generic_handoff
from hiver_agent.schemas import SupportPair

logger = logging.getLogger(__name__)


@dataclass
class ThreadReconstructionStats:
    """Diagnostic statistics from thread reconstruction."""

    total_input_rows: int = 0
    total_unique_tweets: int = 0
    total_threads_identified: int = 0
    threads_with_brand_reply: int = 0
    usable_pairs_extracted: int = 0
    cycles_detected: int = 0
    missing_parent_count: int = 0
    orphan_brand_replies: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_rows": self.total_input_rows,
            "total_unique_tweets": self.total_unique_tweets,
            "total_threads_identified": self.total_threads_identified,
            "threads_with_brand_reply": self.threads_with_brand_reply,
            "usable_pairs_extracted": self.usable_pairs_extracted,
            "cycles_detected": self.cycles_detected,
            "missing_parent_count": self.missing_parent_count,
            "orphan_brand_replies": self.orphan_brand_replies,
        }


def _parse_timestamp(val: Any) -> datetime | None:
    """Parse various timestamp representations into datetime."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, datetime):
        return val
    try:
        # Handles standard Twitter format: 'Tue Nov 07 11:00:50 +0000 2017'
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None


def reconstruct_threads(
    df: pd.DataFrame,
    brand_handle: str | None = None,
) -> tuple[list[SupportPair], ThreadReconstructionStats]:
    """Reconstruct support pairs from a raw TWCS DataFrame.

    Expected columns in df:
        - tweet_id (str/int)
        - author_id (str)
        - inbound (bool or str "True"/"False")
        - created_at (str or datetime)
        - text (str)
        - in_response_to_tweet_id (optional, str/int/float)
        - response_tweet_id (optional)

    Args:
        df: Input DataFrame of tweets.
        brand_handle: Optional filter for a specific brand (e.g. 'SpotifyCares').

    Returns:
        A tuple of (list of SupportPair, ThreadReconstructionStats).
    """
    stats = ThreadReconstructionStats(total_input_rows=len(df))

    if df.empty:
        return [], stats

    # Work on a standardized copy
    data = df.copy()
    data["tweet_id"] = data["tweet_id"].astype(str).str.strip()

    # Standardize inbound as boolean
    if data["inbound"].dtype == object:
        data["inbound"] = data["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
    else:
        data["inbound"] = data["inbound"].astype(bool)

    data["author_id"] = data["author_id"].astype(str).str.strip()
    data["text"] = data["text"].fillna("").astype(str)

    # Parent pointers
    if "in_response_to_tweet_id" in data.columns:
        data["parent_id"] = (
            data["in_response_to_tweet_id"]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
    else:
        data["parent_id"] = ""

    # Parse created_at
    if "created_at" in data.columns:
        data["parsed_created_at"] = pd.to_datetime(data["created_at"], errors="coerce")
    else:
        data["parsed_created_at"] = None

    # Filter by brand if specified
    # Note: brand can appear as author of outbound OR mentioned in inbound
    set(data[~data["inbound"]]["author_id"].unique())

    # Build lookup map: tweet_id -> row dict
    tweet_map: dict[str, dict[str, Any]] = {}
    for row in data.to_dict(orient="records"):
        tweet_map[row["tweet_id"]] = row

    stats.total_unique_tweets = len(tweet_map)

    # Step 1: Find root for every tweet with cycle detection
    root_cache: dict[str, str] = {}

    for tid in tweet_map:
        if tid in root_cache:
            continue

        curr = tid
        chain: list[str] = []
        visited_local: set[str] = set()

        while curr and curr in tweet_map:
            if curr in visited_local:
                # Cycle detected
                stats.cycles_detected += 1
                curr = chain[0]  # Break loop by using local head
                break
            visited_local.add(curr)
            chain.append(curr)

            parent = tweet_map[curr]["parent_id"]
            if not parent or parent == "nan" or parent not in tweet_map:
                if parent and parent != "nan":
                    stats.missing_parent_count += 1
                break
            if parent in root_cache:
                curr = root_cache[parent]
                chain.append(curr)
                break
            curr = parent

        root = chain[-1] if chain else tid
        for member in chain:
            root_cache[member] = root

    # Step 2: Group tweets by root thread ID
    threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tid, row in tweet_map.items():
        thread_id = root_cache.get(tid, tid)
        threads[thread_id].append(row)

    stats.total_threads_identified = len(threads)

    # Step 3: Extract customer -> brand support pairs from each thread
    pairs: list[SupportPair] = []

    for thread_id, tweets in threads.items():
        # Sort tweets by timestamp, tie-break by tweet_id
        sorted_tweets = sorted(
            tweets,
            key=lambda t: (
                t["parsed_created_at"] if pd.notna(t["parsed_created_at"]) else datetime.min,
                t["tweet_id"],
            ),
        )

        has_inbound = any(t["inbound"] for t in sorted_tweets)
        has_outbound = any(not t["inbound"] for t in sorted_tweets)

        if not (has_inbound and has_outbound):
            continue

        stats.threads_with_brand_reply += 1

        # Direct parent-child lookup within thread
        children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in sorted_tweets:
            parent = t["parent_id"]
            if parent:
                children_by_parent[parent].append(t)

        # Pair extraction: for each inbound customer message, find direct brand reply
        for t in sorted_tweets:
            if not t["inbound"]:
                continue

            customer_tid = t["tweet_id"]
            customer_text = t["text"]
            customer_time = (
                t["parsed_created_at"].to_pydatetime() if pd.notna(t["parsed_created_at"]) else None
            )

            # Strategy A: Direct reply child
            direct_children = children_by_parent.get(customer_tid, [])
            brand_replies = [c for c in direct_children if not c["inbound"]]

            # Strategy B: If no direct child via parent_id, check immediate next outbound in sorted order
            if not brand_replies:
                idx = sorted_tweets.index(t)
                for next_t in sorted_tweets[idx + 1 :]:
                    if not next_t["inbound"]:
                        brand_replies = [next_t]
                        break
                    # If another customer message arrives before brand replies, stop
                    break

            if not brand_replies:
                continue

            # Take the earliest brand response
            support_t = brand_replies[0]
            brand = support_t["author_id"]

            if brand_handle and brand.lower() != brand_handle.lower():
                continue

            support_tid = support_t["tweet_id"]
            support_text = support_t["text"]
            support_time = (
                support_t["parsed_created_at"].to_pydatetime()
                if pd.notna(support_t["parsed_created_at"])
                else None
            )

            latency: float | None = None
            if customer_time and support_time:
                latency = max(0.0, (support_time - customer_time).total_seconds())

            pair_id = f"{thread_id}_{customer_tid}_{support_tid}"
            pair = SupportPair(
                pair_id=pair_id,
                thread_id=thread_id,
                customer_tweet_id=customer_tid,
                support_tweet_id=support_tid,
                customer_text=customer_text,
                support_text=support_text,
                created_at_customer=customer_time,
                created_at_support=support_time,
                response_latency_seconds=latency,
                brand=brand,
                is_generic_handoff=is_generic_handoff(support_text),
            )
            pairs.append(pair)

    stats.usable_pairs_extracted = len(pairs)
    return pairs, stats

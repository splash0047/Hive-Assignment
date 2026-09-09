"""Deterministic sampling and corpus partitioning."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from hiver_agent.config import resolve_path
from hiver_agent.schemas import SupportPair

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    path = resolve_path(str(file_path))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sample_support_pairs(
    pairs: list[SupportPair],
    n: int,
    seed: int = 42,
    min_reply_chars: int = 12,
    exclude_generic_handoffs: bool = False,
) -> list[SupportPair]:
    """Deterministically sample N support pairs with filtering."""
    filtered = pairs

    if min_reply_chars > 0:
        filtered = [p for p in filtered if len(p.support_text.strip()) >= min_reply_chars]

    if exclude_generic_handoffs:
        filtered = [p for p in filtered if not p.is_generic_handoff]

    if len(filtered) <= n:
        return filtered

    # Deterministic sorting before sampling ensures stability across platforms
    sorted_pairs = sorted(filtered, key=lambda p: p.pair_id)

    import random

    rng = random.Random(seed)
    indices = list(range(len(sorted_pairs)))
    rng.shuffle(indices)
    sampled_indices = sorted(indices[:n])

    return [sorted_pairs[i] for i in sampled_indices]


def create_manifest(
    files: dict[str, str | Path],
    metadata: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a hash manifest documenting frozen dataset files."""
    manifest = {
        "metadata": metadata or {},
        "files": {},
    }

    for name, path_str in files.items():
        p = resolve_path(str(path_str))
        if p.exists():
            manifest["files"][name] = {
                "path": str(p.relative_to(p.parents[2]) if len(p.parents) >= 3 else p.name),
                "sha256": compute_file_sha256(p),
                "size_bytes": p.stat().st_size,
            }

    if output_path:
        out_p = resolve_path(str(output_path))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written to {out_p}")

    return manifest


def build_curated_data(cfg: Any) -> None:
    """Build curated brand pairs and historical retrieval corpus Parquet files."""
    from rich.console import Console

    from hiver_agent.data.load import load_raw_twcs, save_pairs_to_parquet
    from hiver_agent.data.threads import reconstruct_threads

    console = Console()
    brand = cfg.project.brand or "SpotifyCares"
    console.print(f"[bold cyan]Building curated datasets for brand:[/] [green]{brand}[/]")

    raw_path = resolve_path(cfg.data.raw_csv)
    if not raw_path.exists():
        console.print(f"[bold red]Error:[/] Raw dataset not found at {raw_path}")
        console.print("Run [green]python scripts/download_data.py[/] first.")
        return

    console.print(f"Loading raw tweets for {brand}...")
    df = load_raw_twcs(raw_path, brand=brand)
    console.print(f"Reconstructing threads from {len(df)} tweets...")
    pairs, stats = reconstruct_threads(df, brand_handle=brand)
    console.print(f"[green]Reconstructed {len(pairs)} usable support pairs.[/]")

    # 1. Full brand pairs
    brand_pairs = sample_support_pairs(
        pairs,
        n=cfg.data.max_brand_pairs,
        seed=cfg.project.seed,
        min_reply_chars=cfg.data.min_reply_chars,
    )
    p1 = save_pairs_to_parquet(brand_pairs, "data/curated/brand_pairs.parquet")
    console.print(f"Saved {len(brand_pairs)} pairs to {p1}")

    # 2. Historical retrieval corpus (exclude generic handoffs to avoid deflecting answers)
    corpus_pairs = sample_support_pairs(
        brand_pairs,
        n=cfg.data.retrieval_corpus_size,
        seed=cfg.project.seed + 1,
        min_reply_chars=cfg.data.min_reply_chars,
        exclude_generic_handoffs=True,
    )
    p2 = save_pairs_to_parquet(corpus_pairs, "data/curated/historical_corpus.parquet")
    console.print(f"Saved {len(corpus_pairs)} pairs to {p2}")

    # 3. Hash manifest
    manifest_path = resolve_path("data/curated/manifest.json")
    create_manifest(
        files={"brand_pairs": p1, "historical_corpus": p2},
        metadata={
            "brand": brand,
            "seed": cfg.project.seed,
            "total_pairs": len(brand_pairs),
            "corpus_pairs": len(corpus_pairs),
            "stats": stats.to_dict(),
        },
        output_path=manifest_path,
    )
    console.print(f"[bold green][OK] Curated data and manifest created at:[/] {manifest_path}")

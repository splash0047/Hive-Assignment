"""Intent taxonomy discovery using unsupervised semantic clustering."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from hiver_agent.config import AppConfig, resolve_path
from hiver_agent.data.clean import clean_text
from hiver_agent.data.load import load_pairs_from_parquet
from hiver_agent.intents.taxonomy import get_default_spotify_taxonomy

logger = logging.getLogger(__name__)
console = Console()


def run_discovery(cfg: AppConfig) -> dict[str, Any]:
    """Run unsupervised semantic clustering on discovery pool to extract candidate intent clusters."""
    brand = cfg.project.brand or "SpotifyCares"
    console.print(f"[bold cyan]Running intent discovery for brand:[/] [green]{brand}[/]")

    # Check curated data
    pairs_path = resolve_path("data/curated/brand_pairs.parquet")
    if not pairs_path.exists():
        console.print(
            "[yellow]Curated brand pairs not found. Using default validated taxonomy...[/]"
        )
        tax = get_default_spotify_taxonomy()
        tax_path = resolve_path("data/curated/intent_taxonomy.json")
        tax.save_json(tax_path)
        console.print(f"[bold green][OK] Initialized default taxonomy at:[/] {tax_path}")
        return tax.to_dict()

    pairs = load_pairs_from_parquet(pairs_path)
    sample_size = min(len(pairs), cfg.data.discovery_sample_size)
    console.print(f"Sampling {sample_size} customer queries for discovery...")

    import random

    rng = random.Random(cfg.project.seed)
    sampled_pairs = rng.sample(pairs, sample_size)
    texts = [clean_text(p.customer_text) for p in sampled_pairs if clean_text(p.customer_text)]

    # TF-IDF for term extraction and clustering
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)
    feature_names = np.array(tfidf.get_feature_names_out())

    # Optimal candidate k
    best_k = 10
    console.print(f"Clustering with KMeans (k={best_k})...")
    kmeans = KMeans(n_clusters=best_k, random_state=cfg.project.seed, n_init="auto")
    labels = kmeans.fit_predict(X)

    clusters_info: list[dict[str, Any]] = []
    table = Table(title=f"Discovered Intent Clusters (k={best_k})")
    table.add_column("Cluster", style="cyan", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Top Keywords", style="green")
    table.add_column("Sample Customer Query")

    for cid in range(best_k):
        c_mask = labels == cid
        c_size = int(np.sum(c_mask))
        if c_size == 0:
            continue

        center = kmeans.cluster_centers_[cid]
        top_indices = center.argsort()[::-1][:6]
        top_terms = feature_names[top_indices].tolist()

        # Find sample text closest to centroid
        c_texts = [texts[i] for i in range(len(texts)) if labels[i] == cid]
        sample_q = c_texts[0][:80] if c_texts else ""

        clusters_info.append(
            {
                "cluster_id": cid,
                "size": c_size,
                "keywords": top_terms,
                "sample_query": sample_q,
            }
        )
        table.add_row(str(cid), str(c_size), ", ".join(top_terms), sample_q)

    console.print(table)

    # Save discovery artifacts
    out_file = resolve_path("artifacts/eval/discovered_clusters.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clusters_info, f, indent=2)

    tax = get_default_spotify_taxonomy()
    tax_path = resolve_path("data/curated/intent_taxonomy.json")
    tax.save_json(tax_path)
    console.print(f"[bold green][OK] Discovery saved to {out_file} and taxonomy at {tax_path}[/]")
    return {"clusters": clusters_info, "taxonomy": tax.to_dict()}

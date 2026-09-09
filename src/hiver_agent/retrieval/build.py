"""Build and save FAISS semantic retrieval index from historical support corpus."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hiver_agent.data.clean import clean_text
from hiver_agent.data.load import load_pairs_from_parquet
from hiver_agent.retrieval.index import FaissIndex

logger = logging.getLogger(__name__)


def build_faiss_index(
    corpus_parquet_path: str | Path,
    encoder: Any,
    output_index_path: str | Path,
    output_meta_path: str | Path,
    batch_size: int = 128,
) -> FaissIndex:
    """Build a FAISS IndexFlatIP from support pairs stored in a Parquet file."""
    pairs = load_pairs_from_parquet(corpus_parquet_path)
    logger.info(f"Loaded {len(pairs)} historical pairs for FAISS index build.")

    if not pairs:
        raise ValueError("Corpus is empty, cannot build index.")

    customer_texts = [clean_text(p.customer_text) for p in pairs]

    logger.info(f"Encoding {len(customer_texts)} customer query vectors in batches...")
    embeddings = encoder.encode(
        customer_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    metadata: list[dict[str, Any]] = []
    for p in pairs:
        metadata.append(
            {
                "pair_id": p.pair_id,
                "customer_text": p.customer_text,
                "support_text": p.support_text,
                "brand": p.brand,
                "is_generic_handoff": p.is_generic_handoff,
            }
        )

    dimension = embeddings.shape[1]
    faiss_index = FaissIndex(dimension=dimension)
    faiss_index.add(embeddings, metadata)
    faiss_index.save(output_index_path, output_meta_path)
    logger.info(f"FAISS index built with {faiss_index.index.ntotal} vectors.")
    return faiss_index

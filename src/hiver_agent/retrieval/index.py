"""FAISS vector index wrapper for exact cosine inner-product search."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from hiver_agent.config import resolve_path

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS exact inner-product index with metadata lookup."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        # IndexFlatIP with L2-normalized vectors calculates cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add normalized embeddings and their corresponding metadata dicts."""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Count mismatch between embeddings ({len(embeddings)}) and metadata ({len(metadata)})"
            )

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings.astype("float32"))
        # Normalize in-place for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.metadata.extend(metadata)
        logger.info(f"Added {len(vectors)} vectors to FAISS index (total: {self.index.ntotal})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Search top_k nearest neighbors by cosine similarity."""
        if self.index.ntotal == 0:
            return []

        # Ensure query is 2D float32 normalized
        q = np.ascontiguousarray(query_embedding.reshape(1, -1).astype("float32"))
        faiss.normalize_L2(q)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results: list[tuple[float, dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx >= 0 and idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))
        return results

    def save(self, index_file: str | Path, meta_file: str | Path) -> None:
        """Save index and metadata to disk."""
        idx_p = resolve_path(str(index_file))
        meta_p = resolve_path(str(meta_file))
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(idx_p))
        with open(meta_p, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"FAISS index saved to {idx_p} and metadata to {meta_p}")

    @classmethod
    def load(cls, index_file: str | Path, meta_file: str | Path) -> FaissIndex:
        """Load index and metadata from disk."""
        idx_p = resolve_path(str(index_file))
        meta_p = resolve_path(str(meta_file))

        index = faiss.read_index(str(idx_p))
        with open(meta_p, "rb") as f:
            metadata = pickle.load(f)

        instance = cls(dimension=index.d)
        instance.index = index
        instance.metadata = metadata
        return instance

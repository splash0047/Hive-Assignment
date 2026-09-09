"""Unit tests for FAISS index and retrieval engine."""

from __future__ import annotations

import numpy as np

from hiver_agent.retrieval.index import FaissIndex
from hiver_agent.retrieval.retrieve import RetrievalEngine


class DummyEncoder:
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        # Returns simple deterministic vectors based on text length
        vectors = []
        for t in texts:
            vec = np.zeros(16, dtype="float32")
            idx = len(t) % 16
            vec[idx] = 1.0
            vectors.append(vec)
        return np.array(vectors)


def test_faiss_index_basic(tmp_path):
    idx = FaissIndex(dimension=16)
    embeddings = np.eye(4, 16, dtype="float32")
    meta = [
        {"pair_id": f"p_{i}", "customer_text": f"text {i}", "support_text": f"ans {i}"}
        for i in range(4)
    ]
    idx.add(embeddings, meta)

    results = idx.search(embeddings[0], top_k=2)
    assert len(results) == 2
    assert results[0][1]["pair_id"] == "p_0"
    assert results[0][0] >= 0.99

    # Test serialization
    bin_file = tmp_path / "idx.bin"
    meta_file = tmp_path / "meta.pkl"
    idx.save(bin_file, meta_file)

    loaded = FaissIndex.load(bin_file, meta_file)
    assert loaded.index.ntotal == 4
    res2 = loaded.search(embeddings[0], top_k=1)
    assert res2[0][1]["pair_id"] == "p_0"


def test_retrieval_engine_exclusion():
    idx = FaissIndex(dimension=16)
    embeddings = np.eye(3, 16, dtype="float32")
    meta = [
        {
            "pair_id": "pair_A",
            "customer_text": "text_A",
            "support_text": "ans_A",
            "intent": "billing",
        },
        {
            "pair_id": "pair_B",
            "customer_text": "text_B",
            "support_text": "ans_B",
            "intent": "billing",
        },
        {
            "pair_id": "pair_C",
            "customer_text": "text_C",
            "support_text": "ans_C",
            "intent": "login",
        },
    ]
    idx.add(embeddings, meta)

    engine = RetrievalEngine(index=idx, encoder=DummyEncoder(), top_k=3, min_similarity=0.1)

    # Exclude pair_A to simulate leakage prevention
    ev = engine.retrieve(query="text_A", exclude_pair_ids={"pair_A"})
    retrieved_ids = [e.pair_id for e in ev]
    assert "pair_A" not in retrieved_ids

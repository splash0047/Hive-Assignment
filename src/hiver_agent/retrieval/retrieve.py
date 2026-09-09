"""Retrieval engine with semantic search, intent gating, and leakage prevention."""

from __future__ import annotations

import logging
from typing import Any

from hiver_agent.data.clean import clean_text
from hiver_agent.retrieval.index import FaissIndex
from hiver_agent.schemas import RetrievedEvidence

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Semantic retrieval engine over historical customer support pairs."""

    def __init__(
        self,
        index: FaissIndex,
        encoder: Any,
        top_k: int = 6,
        min_similarity: float = 0.45,
        same_intent_filter: bool = True,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.same_intent_filter = same_intent_filter

    def retrieve(
        self,
        query: str,
        predicted_intent: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        exclude_pair_ids: set[str] | None = None,
    ) -> list[RetrievedEvidence]:
        """Retrieve top historical evidence items for a given query text.

        Args:
            query: The customer's message text.
            predicted_intent: Optional predicted intent to filter or rank evidence.
            top_k: Number of candidates to return (defaults to self.top_k).
            min_similarity: Minimum cosine similarity threshold.
            exclude_pair_ids: Set of pair IDs to exclude (prevents evaluation leakage).

        Returns:
            List of RetrievedEvidence objects sorted by descending similarity.
        """
        k = top_k or self.top_k
        min_sim = min_similarity if min_similarity is not None else self.min_similarity
        exclude = exclude_pair_ids or set()

        cleaned_q = clean_text(query)
        if not cleaned_q:
            return []

        # Encode query
        query_emb = self.encoder.encode([cleaned_q], normalize_embeddings=True)[0]

        # Fetch more candidates initially to allow intent and exclusion filtering
        raw_results = self.index.search(query_emb, top_k=k * 3)

        evidence_list: list[RetrievedEvidence] = []
        for score, meta in raw_results:
            pair_id = meta.get("pair_id", "")
            if pair_id in exclude:
                continue

            if score < min_sim:
                continue

            intent = meta.get("intent", "")
            # Apply same-intent filter if enabled and predicted_intent is provided
            if (
                self.same_intent_filter
                and predicted_intent
                and intent
                and intent != predicted_intent
            ):
                continue

            evidence = RetrievedEvidence(
                pair_id=pair_id,
                customer_text=meta.get("customer_text", ""),
                support_text=meta.get("support_text", ""),
                similarity_score=float(score),
                intent=intent,
            )
            evidence_list.append(evidence)

            if len(evidence_list) >= k:
                break

        return evidence_list

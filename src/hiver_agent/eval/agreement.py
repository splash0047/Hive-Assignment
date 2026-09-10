"""Inter-annotator and human-vs-judge agreement analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class AgreementResult:
    """Binary inter-evaluator agreement metrics."""

    percent_agreement: float
    cohen_kappa: float
    sample_size: int
    confusion: list[list[int]]

    def summary(self) -> str:
        return (
            f"Agreement: {self.percent_agreement * 100:.1f}% | "
            f"Cohen's Kappa: {self.cohen_kappa:.3f} (N={self.sample_size})"
        )


@dataclass
class OrdinalAgreementResult:
    """Agreement metrics for a shared 1-5 ordinal rubric dimension."""

    exact_agreement: float
    within_one_agreement: float
    weighted_kappa: float | None
    spearman_correlation: float | None
    sample_size: int


def compute_human_judge_agreement(
    human_labels: list[bool | int | str],
    judge_labels: list[bool | int | str],
) -> AgreementResult:
    """Calculate binary agreement metrics between a human annotator and judge."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("Lengths of human_labels and judge_labels must match.")

    n = len(human_labels)
    if n == 0:
        raise ValueError(
            "Cannot compute agreement on an empty set of labels (sample size must be > 0)."
        )

    h = np.array(human_labels)
    j = np.array(judge_labels)

    agree_count = int(np.sum(h == j))
    pct = agree_count / n

    # Reject undefined kappa before calling sklearn (avoids NaN + warning spam).
    if len(set(map(str, human_labels))) < 2 or len(set(map(str, judge_labels))) < 2:
        raise ValueError(
            "Cohen's kappa is undefined for this label distribution "
            "(need variation in both human and judge labels)."
        )

    kappa_raw = float(cohen_kappa_score(h, j))
    if np.isnan(kappa_raw):
        raise ValueError(
            "Cohen's kappa is undefined for this label distribution "
            "(need variation in both human and judge labels)."
        )

    cm = confusion_matrix(h, j).tolist()

    return AgreementResult(
        percent_agreement=round(pct, 4),
        cohen_kappa=round(kappa_raw, 4),
        sample_size=n,
        confusion=cm,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including correct handling of ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def compute_ordinal_agreement(
    human_scores: list[int],
    judge_scores: list[int],
) -> OrdinalAgreementResult:
    """Calculate agreement for a shared 1-5 rubric dimension.

    Reports the metrics requested by the evaluation plan: exact agreement,
    within-one-point agreement, quadratic-weighted Cohen's kappa, and Spearman rank
    correlation. Kappa/correlation are returned as ``None`` when a constant score
    distribution makes the statistic undefined.
    """
    if len(human_scores) != len(judge_scores):
        raise ValueError("Lengths of human_scores and judge_scores must match.")
    if not human_scores:
        raise ValueError("Cannot compute ordinal agreement on an empty score set.")

    h = np.asarray(human_scores, dtype=int)
    j = np.asarray(judge_scores, dtype=int)
    if np.any((h < 1) | (h > 5)) or np.any((j < 1) | (j > 5)):
        raise ValueError("Ordinal agreement scores must all be integers from 1 to 5.")

    exact = float(np.mean(h == j))
    within_one = float(np.mean(np.abs(h - j) <= 1))

    weighted_kappa: float | None = None
    if len(np.unique(h)) >= 2 and len(np.unique(j)) >= 2:
        kappa_raw = float(cohen_kappa_score(h, j, labels=[1, 2, 3, 4, 5], weights="quadratic"))
        if not np.isnan(kappa_raw):
            weighted_kappa = round(kappa_raw, 4)

    spearman: float | None = None
    if len(np.unique(h)) >= 2 and len(np.unique(j)) >= 2:
        h_ranks = _average_ranks(h)
        j_ranks = _average_ranks(j)
        corr = float(np.corrcoef(h_ranks, j_ranks)[0, 1])
        if not np.isnan(corr):
            spearman = round(corr, 4)

    return OrdinalAgreementResult(
        exact_agreement=round(exact, 4),
        within_one_agreement=round(within_one, 4),
        weighted_kappa=weighted_kappa,
        spearman_correlation=spearman,
        sample_size=len(h),
    )

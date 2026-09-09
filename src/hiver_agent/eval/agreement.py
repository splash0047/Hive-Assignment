"""Inter-annotator and Human-vs-Judge agreement analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class AgreementResult:
    """Inter-evaluator agreement metrics."""

    percent_agreement: float
    cohen_kappa: float
    sample_size: int
    confusion: list[list[int]]

    def summary(self) -> str:
        return (
            f"Agreement: {self.percent_agreement * 100:.1f}% | "
            f"Cohen's Kappa: {self.cohen_kappa:.3f} (N={self.sample_size})"
        )


def compute_human_judge_agreement(
    human_labels: list[bool | int | str],
    judge_labels: list[bool | int | str],
) -> AgreementResult:
    """Calculate agreement metrics between human annotator and automated judge."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("Lengths of human_labels and judge_labels must match.")

    n = len(human_labels)
    if n == 0:
        return AgreementResult(percent_agreement=1.0, cohen_kappa=1.0, sample_size=0, confusion=[])

    h = np.array(human_labels)
    j = np.array(judge_labels)

    agree_count = int(np.sum(h == j))
    pct = agree_count / n

    try:
        kappa = float(cohen_kappa_score(h, j))
    except Exception:
        kappa = 1.0 if pct == 1.0 else 0.0

    cm = confusion_matrix(h, j).tolist()

    return AgreementResult(
        percent_agreement=round(pct, 4),
        cohen_kappa=round(kappa, 4),
        sample_size=n,
        confusion=cm,
    )

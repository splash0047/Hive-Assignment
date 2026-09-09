"""Comprehensive evaluation metrics suite with bootstrap confidence intervals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class MetricWithCI:
    """A metric value accompanied by a 95% bootstrap confidence interval."""

    value: float
    ci_lower: float
    ci_upper: float

    def __str__(self) -> str:
        return f"{self.value:.3f} (95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}])"


@dataclass
class EvaluationSummary:
    """Consolidated headline metrics for the support agent."""

    total_examples: int
    auto_handle_count: int
    escalated_count: int
    coverage_rate: MetricWithCI  # Auto-handle rate
    escalation_accuracy: MetricWithCI
    escalation_f1: MetricWithCI
    escalation_precision: MetricWithCI
    escalation_recall: MetricWithCI
    intent_accuracy: MetricWithCI
    intent_macro_f1: MetricWithCI
    grounded_acceptance_rate: MetricWithCI | None = None
    unsafe_autohandle_rate: MetricWithCI | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "auto_handle_count": self.auto_handle_count,
            "escalated_count": self.escalated_count,
            "coverage_rate": str(self.coverage_rate),
            "escalation_accuracy": str(self.escalation_accuracy),
            "escalation_f1": str(self.escalation_f1),
            "escalation_precision": str(self.escalation_precision),
            "escalation_recall": str(self.escalation_recall),
            "intent_accuracy": str(self.intent_accuracy),
            "intent_macro_f1": str(self.intent_macro_f1),
            "grounded_acceptance_rate": (
                str(self.grounded_acceptance_rate) if self.grounded_acceptance_rate else "N/A"
            ),
            "unsafe_autohandle_rate": (
                str(self.unsafe_autohandle_rate) if self.unsafe_autohandle_rate else "N/A"
            ),
        }


def bootstrap_ci(
    y_true: list[Any] | np.ndarray,
    y_pred: list[Any] | np.ndarray,
    metric_fn: Any,
    n_resamples: int = 2000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> MetricWithCI:
    """Compute empirical metric value and 95% bootstrap confidence interval."""
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n = len(yt)

    base_val = float(metric_fn(yt, yp))
    if n < 5:
        return MetricWithCI(value=base_val, ci_lower=base_val, ci_upper=base_val)

    rng = np.random.RandomState(random_state)
    boot_vals = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_vals[i] = metric_fn(yt[indices], yp[indices])

    lower = float(np.percentile(boot_vals, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return MetricWithCI(value=base_val, ci_lower=lower, ci_upper=upper)


def compute_agent_metrics(
    true_intents: list[str],
    pred_intents: list[str],
    true_escalations: list[bool],
    agent_actions: list[str],  # "AUTO_HANDLE" or "ESCALATE"
    judge_accepts: list[bool] | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> EvaluationSummary:
    """Compute all evaluation metrics across intent classification, escalation, and output quality."""
    n = len(true_intents)
    pred_escalations = [a == "ESCALATE" for a in agent_actions]

    auto_handle_count = sum(1 for a in agent_actions if a == "AUTO_HANDLE")
    escalated_count = n - auto_handle_count

    # 1. Coverage / Auto-handle rate
    auto_handle_count / n if n > 0 else 0.0
    coverage_ci = bootstrap_ci(
        y_true=np.zeros(n),
        y_pred=np.array([1 if a == "AUTO_HANDLE" else 0 for a in agent_actions]),
        metric_fn=lambda yt, yp: float(np.mean(yp)),
        n_resamples=n_resamples,
        random_state=seed,
    )

    # 2. Intent metrics
    intent_acc_ci = bootstrap_ci(
        y_true=true_intents,
        y_pred=pred_intents,
        metric_fn=lambda yt, yp: float(accuracy_score(yt, yp)),
        n_resamples=n_resamples,
        random_state=seed + 1,
    )
    intent_f1_ci = bootstrap_ci(
        y_true=true_intents,
        y_pred=pred_intents,
        metric_fn=lambda yt, yp: float(f1_score(yt, yp, average="macro", zero_division=0)),
        n_resamples=n_resamples,
        random_state=seed + 2,
    )

    # 3. Escalation routing metrics
    esc_acc_ci = bootstrap_ci(
        y_true=true_escalations,
        y_pred=pred_escalations,
        metric_fn=lambda yt, yp: float(accuracy_score(yt, yp)),
        n_resamples=n_resamples,
        random_state=seed + 3,
    )
    esc_f1_ci = bootstrap_ci(
        y_true=true_escalations,
        y_pred=pred_escalations,
        metric_fn=lambda yt, yp: float(f1_score(yt, yp, zero_division=0)),
        n_resamples=n_resamples,
        random_state=seed + 4,
    )
    esc_prec_ci = bootstrap_ci(
        y_true=true_escalations,
        y_pred=pred_escalations,
        metric_fn=lambda yt, yp: float(precision_score(yt, yp, zero_division=0)),
        n_resamples=n_resamples,
        random_state=seed + 5,
    )
    esc_rec_ci = bootstrap_ci(
        y_true=true_escalations,
        y_pred=pred_escalations,
        metric_fn=lambda yt, yp: float(recall_score(yt, yp, zero_division=0)),
        n_resamples=n_resamples,
        random_state=seed + 6,
    )

    # 4. Optional Judge acceptance rate on auto-handled queries
    grounded_ci = None
    unsafe_ci = None
    if judge_accepts is not None and len(judge_accepts) == n:
        auto_mask = [a == "AUTO_HANDLE" for a in agent_actions]
        if any(auto_mask):
            auto_accepts = [judge_accepts[i] for i in range(n) if auto_mask[i]]
            grounded_ci = bootstrap_ci(
                y_true=np.ones(len(auto_accepts)),
                y_pred=np.array(auto_accepts),
                metric_fn=lambda yt, yp: float(np.mean(yp)),
                n_resamples=n_resamples,
                random_state=seed + 7,
            )
            # Unsafe autohandle: auto handled but true escalation was required
            unsafe_auto = [1 if (auto_mask[i] and true_escalations[i]) else 0 for i in range(n)]
            unsafe_ci = bootstrap_ci(
                y_true=np.zeros(n),
                y_pred=np.array(unsafe_auto),
                metric_fn=lambda yt, yp: float(np.mean(yp)),
                n_resamples=n_resamples,
                random_state=seed + 8,
            )

    return EvaluationSummary(
        total_examples=n,
        auto_handle_count=auto_handle_count,
        escalated_count=escalated_count,
        coverage_rate=coverage_ci,
        escalation_accuracy=esc_acc_ci,
        escalation_f1=esc_f1_ci,
        escalation_precision=esc_prec_ci,
        escalation_recall=esc_rec_ci,
        intent_accuracy=intent_acc_ci,
        intent_macro_f1=intent_f1_ci,
        grounded_acceptance_rate=grounded_ci,
        unsafe_autohandle_rate=unsafe_ci,
    )

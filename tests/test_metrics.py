"""Unit tests for evaluation metrics and agreement calculations."""

from __future__ import annotations

import pytest

from hiver_agent.eval.agreement import (
    compute_human_judge_agreement,
    compute_ordinal_agreement,
)
from hiver_agent.eval.metrics import compute_agent_metrics


def test_compute_agent_metrics():
    true_intents = ["login", "billing", "login", "playback"]
    pred_intents = ["login", "billing", "login", "billing"]  # 3 of 4 correct
    true_esc = [False, True, False, True]
    agent_actions = ["AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE", "ESCALATE"]  # 4 of 4 correct

    summary = compute_agent_metrics(
        true_intents=true_intents,
        pred_intents=pred_intents,
        true_escalations=true_esc,
        agent_actions=agent_actions,
        n_resamples=100,
        seed=42,
    )

    assert summary.total_examples == 4
    assert summary.auto_handle_count == 2
    assert summary.escalated_count == 2
    assert summary.coverage_rate.value == 0.50
    assert summary.escalation_accuracy.value == 1.0
    assert summary.intent_accuracy.value == 0.75


def test_human_judge_agreement():
    human = [True, True, False, True, False]
    judge = [True, True, False, True, False]
    res_perfect = compute_human_judge_agreement(human, judge)
    assert res_perfect.percent_agreement == 1.0
    assert res_perfect.cohen_kappa == 1.0

    divergent_judge = [True, False, False, True, True]
    res_partial = compute_human_judge_agreement(human, divergent_judge)
    assert res_partial.percent_agreement < 1.0

    with pytest.raises(ValueError, match="empty"):
        compute_human_judge_agreement([], [])

    with pytest.raises(ValueError, match="undefined"):
        compute_human_judge_agreement([True, True, True], [True, True, True])


def test_ordinal_agreement_metrics():
    human = [1, 2, 3, 4, 5]
    judge = [1, 3, 3, 5, 5]
    result = compute_ordinal_agreement(human, judge)

    assert result.sample_size == 5
    assert result.exact_agreement == 0.6
    assert result.within_one_agreement == 1.0
    assert result.weighted_kappa is not None
    assert result.spearman_correlation is not None

    perfect = compute_ordinal_agreement([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert perfect.exact_agreement == 1.0
    assert perfect.within_one_agreement == 1.0
    assert perfect.weighted_kappa == 1.0
    assert perfect.spearman_correlation == 1.0

    constant = compute_ordinal_agreement([5, 5, 5], [5, 5, 5])
    assert constant.exact_agreement == 1.0
    assert constant.weighted_kappa is None
    assert constant.spearman_correlation is None

    with pytest.raises(ValueError, match="1 to 5"):
        compute_ordinal_agreement([0, 2], [1, 2])

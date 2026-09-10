"""Unit tests for evaluation metrics and agreement calculations."""

from __future__ import annotations

from hiver_agent.eval.agreement import compute_human_judge_agreement
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

    import pytest

    with pytest.raises(ValueError, match="empty"):
        compute_human_judge_agreement([], [])

    with pytest.raises(ValueError, match="undefined"):
        compute_human_judge_agreement([True, True, True], [True, True, True])

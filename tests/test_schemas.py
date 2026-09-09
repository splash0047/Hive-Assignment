"""Tests for core schemas."""

from __future__ import annotations

from hiver_agent.schemas import AgentOutput, GoldenExample, JudgeScore, SupportPair


def test_support_pair_serialization():
    pair = SupportPair(
        pair_id="pair_001",
        thread_id="thread_001",
        customer_tweet_id="123",
        support_tweet_id="124",
        customer_text="My order is delayed",
        support_text="We are looking into it",
        brand="AppleSupport",
    )
    d = pair.to_dict()
    assert d["pair_id"] == "pair_001"
    assert d["brand"] == "AppleSupport"
    assert d["customer_text"] == "My order is delayed"


def test_golden_example_serialization():
    example = GoldenExample(
        example_id="ex_001",
        tweet_id="123",
        text="Can I get a refund?",
        intent_label="order_status",
        escalation_label=False,
        risk_tags=["billing"],
        split="locked_test",
    )
    d = example.to_dict()
    assert d["example_id"] == "ex_001"
    assert d["risk_tags"] == "billing"
    assert d["split"] == "locked_test"


def test_agent_output():
    out = AgentOutput(
        intent="refund_request",
        intent_confidence=0.92,
        action="AUTO_HANDLE",
        reason="high confidence and retrieval match",
        reply="Please visit our refund portal.",
        evidence_ids=["pair_01"],
        retrieval_scores=[0.88],
    )
    assert out.is_auto_handle()
    assert out.to_dict()["action"] == "AUTO_HANDLE"


def test_judge_score_acceptable():
    score_ok = JudgeScore(
        example_id="ex_1",
        groundedness=5,
        helpfulness=4,
        correctness=5,
        tone=5,
        safety=5,
        overall_accept=True,
    )
    assert score_ok.is_acceptable()

    score_bad = JudgeScore(
        example_id="ex_2",
        groundedness=2,
        helpfulness=4,
        correctness=3,
        tone=5,
        safety=5,
        overall_accept=True,
    )
    assert not score_bad.is_acceptable()

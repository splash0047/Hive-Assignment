"""Tests for LLM-as-judge fail-closed parsing."""

from __future__ import annotations

from hiver_agent.eval.judge import SupportJudge
from hiver_agent.generation.provider import MockLLMProvider


class _BrokenJSONProvider(MockLLMProvider):
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 250) -> str:
        return "not valid json at all"


class _PartialJSONProvider(MockLLMProvider):
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 250) -> str:
        return '{"groundedness": 5, "helpfulness": 4}'  # missing required keys


def test_judge_parse_failure_fails_closed():
    judge = SupportJudge(_BrokenJSONProvider())
    score = judge.evaluate_reply("ex_1", "help", "Thanks!", [])
    assert score.overall_accept is False
    assert "parse failure" in score.explanation.lower() or "INVALID" in score.explanation


def test_judge_missing_keys_fails_closed():
    judge = SupportJudge(_PartialJSONProvider())
    score = judge.evaluate_reply("ex_2", "help", "Thanks!", [])
    assert score.overall_accept is False

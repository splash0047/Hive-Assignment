"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

from hiver_agent.config import AppConfig, load_config, resolve_path, set_seed


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.project.seed == 42
    assert cfg.project.brand == "SpotifyCares"
    assert cfg.intent.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retrieval.top_k == 6
    assert cfg.routing.min_intent_confidence == 0.25


def test_config_overrides():
    cfg = load_config(overrides={"project": {"seed": 123}, "retrieval": {"top_k": 10}})
    assert cfg.project.seed == 123
    assert cfg.retrieval.top_k == 10
    # Other values should remain defaults
    assert cfg.routing.min_intent_confidence == 0.25


def test_config_hash():
    cfg1 = load_config()
    cfg2 = load_config()
    assert cfg1.config_hash() == cfg2.config_hash()

    cfg3 = load_config(overrides={"project": {"seed": 999}})
    assert cfg1.config_hash() != cfg3.config_hash()


def test_resolve_path():
    p = resolve_path("configs/default.yaml")
    assert isinstance(p, Path)
    assert p.exists()


def test_set_seed():
    set_seed(42)
    import random

    val1 = random.random()
    set_seed(42)
    val2 = random.random()
    assert val1 == val2

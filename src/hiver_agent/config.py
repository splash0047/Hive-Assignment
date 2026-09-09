"""Configuration management - loads YAML config with Pydantic validation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _project_root() -> Path:
    """Return the project root directory (contains pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in [current, *list(current.parents)]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = _project_root()


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    brand: str | None = None
    taxonomy_version: str = "v1"


@dataclass(frozen=True)
class DataConfig:
    raw_csv: str = "data/raw/twcs.csv"
    max_brand_pairs: int = 30_000
    discovery_sample_size: int = 4_000
    retrieval_corpus_size: int = 20_000
    min_reply_chars: int = 12


@dataclass(frozen=True)
class IntentConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    n_clusters_candidates: list[int] = field(default_factory=lambda: [8, 10, 12, 14])
    classifier: str = "logistic_regression"
    class_weight: str = "balanced"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 6
    same_intent_filter: bool = True
    min_similarity: float = 0.45


@dataclass(frozen=True)
class RoutingConfig:
    min_intent_confidence: float = 0.70
    min_top_retrieval_similarity: float = 0.55
    min_confidence_margin: float = 0.15
    escalate_on_validator_failure: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "env"
    model: str | None = None
    temperature: float = 0.1
    max_reply_chars: int = 450
    prompt_version: str = "v1"
    cache: bool = True


@dataclass(frozen=True)
class JudgeConfig:
    provider: str = "env"
    model: str | None = None
    temperature: float = 0.0
    rubric_version: str = "v1"
    cache: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int = 5_000
    fast_mode: bool = False
    use_frozen_artifacts: bool = False
    use_cached_generation: bool = False
    use_cached_judge_scores: bool = False
    regenerate_figures: bool = True


@dataclass
class AppConfig:
    """Top-level application configuration."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def config_hash(self) -> str:
        """Return a short SHA-256 hash of the serialized config for reproducibility."""
        raw = json.dumps(self.__dict__, default=str, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(data: dict[str, Any]) -> AppConfig:
    """Convert a nested dict to an AppConfig instance."""
    return AppConfig(
        project=ProjectConfig(**data.get("project", {})),
        data=DataConfig(**data.get("data", {})),
        intent=IntentConfig(**data.get("intent", {})),
        retrieval=RetrievalConfig(**data.get("retrieval", {})),
        routing=RoutingConfig(**data.get("routing", {})),
        generation=GenerationConfig(**data.get("generation", {})),
        judge=JudgeConfig(**data.get("judge", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
    )


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load configuration from YAML file with optional overrides.

    Priority: overrides > specified config > default.yaml > built-in defaults.
    """
    base_data: dict[str, Any] = {}

    # Load default config if it exists
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            base_data = yaml.safe_load(f) or {}

    # Load specified config (merges on top of default)
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        with open(config_path) as f:
            file_data = yaml.safe_load(f) or {}
        base_data = _deep_merge(base_data, file_data)

    # Apply overrides
    if overrides:
        base_data = _deep_merge(base_data, overrides)

    return _dict_to_config(base_data)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p

"""Baseline classifiers for intent recognition (Majority Class and TF-IDF + Logistic Regression)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from hiver_agent.data.clean import clean_text

logger = logging.getLogger(__name__)


@dataclass
class ClassifierEvaluationResult:
    """Evaluation metrics for an intent classifier."""

    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report_dict: dict[str, Any]

    def summary(self) -> str:
        return (
            f"[{self.model_name}] Accuracy: {self.accuracy:.4f} | "
            f"Macro-F1: {self.macro_f1:.4f} | Weighted-F1: {self.weighted_f1:.4f}"
        )


class MajorityClassBaseline:
    """Trivial baseline predicting the most frequent class."""

    def __init__(self, strategy: str = "most_frequent") -> None:
        self.model = DummyClassifier(strategy=strategy)
        self.classes_: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> MajorityClassBaseline:
        # DummyClassifier requires 2D input
        X = np.zeros((len(texts), 1))
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        X = np.zeros((len(texts), 1))
        return list(self.model.predict(X))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), 1))
        return self.model.predict_proba(X)


class TfidfLogisticBaseline:
    """Simple, transparent baseline using TF-IDF word n-grams and Logistic Regression."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        class_weight: str | None = "balanced",
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
        )
        self.pipeline = Pipeline([("tfidf", self.vectorizer), ("clf", self.classifier)])
        self.classes_: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> TfidfLogisticBaseline:
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)


def evaluate_classifier(
    model: Any,
    texts: list[str],
    labels: list[str],
    model_name: str = "Model",
) -> ClassifierEvaluationResult:
    """Evaluate a classifier on test texts and labels."""
    preds = model.predict(texts)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    return ClassifierEvaluationResult(
        model_name=model_name,
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        report_dict=report,
    )

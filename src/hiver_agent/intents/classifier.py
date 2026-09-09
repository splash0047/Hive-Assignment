"""Sentence embedding intent classifier with confidence scoring and margin estimation."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from hiver_agent.config import resolve_path
from hiver_agent.data.clean import clean_text

logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    """Detailed prediction result for a single input text."""

    intent: str
    confidence: float
    confidence_margin: float  # difference between top-1 and top-2 probabilities
    all_probabilities: dict[str, float]


class SentenceEmbeddingClassifier:
    """Semantic intent classifier combining SentenceTransformers and regularized LogisticRegression."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        class_weight: str | None = "balanced",
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.class_weight = class_weight
        self.C = C
        self.random_state = random_state
        self._encoder = None
        self.classifier = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
        )
        self.classes_: list[str] = []

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode cleaned texts to normalized embeddings."""
        cleaned = [clean_text(t) for t in texts]
        encoder = self._get_encoder()
        embeddings = encoder.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    def fit(self, texts: list[str], labels: list[str]) -> SentenceEmbeddingClassifier:
        logger.info(f"Fitting SentenceEmbeddingClassifier on {len(texts)} examples...")
        embeddings = self.encode(texts)
        self.classifier.fit(embeddings, labels)
        self.classes_ = list(self.classifier.classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        embeddings = self.encode(texts)
        return list(self.classifier.predict(embeddings))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        embeddings = self.encode(texts)
        return self.classifier.predict_proba(embeddings)

    def predict_one(self, text: str) -> IntentPrediction:
        """Predict intent with top confidence and confidence margin."""
        probas = self.predict_proba([text])[0]
        sorted_indices = np.argsort(probas)[::-1]

        top_idx = sorted_indices[0]
        top_intent = self.classes_[top_idx]
        top_conf = float(probas[top_idx])

        margin = 1.0
        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1]
            margin = float(top_conf - probas[second_idx])

        prob_dict = {cls: float(probas[i]) for i, cls in enumerate(self.classes_)}
        return IntentPrediction(
            intent=top_intent,
            confidence=top_conf,
            confidence_margin=margin,
            all_probabilities=prob_dict,
        )

    def save(self, output_path: str | Path) -> Path:
        path = resolve_path(str(output_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "classifier": self.classifier,
                    "classes_": self.classes_,
                    "C": self.C,
                    "class_weight": self.class_weight,
                    "random_state": self.random_state,
                },
                f,
            )
        logger.info(f"Classifier saved to {path}")
        return path

    @classmethod
    def load(cls, input_path: str | Path) -> SentenceEmbeddingClassifier:
        path = resolve_path(str(input_path))
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            model_name=data["model_name"],
            class_weight=data.get("class_weight"),
            C=data.get("C", 1.0),
            random_state=data.get("random_state", 42),
        )
        obj.classifier = data["classifier"]
        obj.classes_ = data["classes_"]
        return obj

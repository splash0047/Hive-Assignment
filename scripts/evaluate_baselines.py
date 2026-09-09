"""Evaluate and compare baseline intent classifiers against sentence-embedding model."""

from __future__ import annotations

import json

import pandas as pd
from rich.console import Console
from rich.table import Table

from hiver_agent.config import resolve_path
from hiver_agent.intents.baseline import (
    MajorityClassBaseline,
    TfidfLogisticBaseline,
    evaluate_classifier,
)
from hiver_agent.intents.classifier import SentenceEmbeddingClassifier

console = Console()


def run_baseline_comparison():
    csv_path = resolve_path("data/golden/golden_eval.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Strict split: train on calibration, evaluate on locked_test
    train_df = df[df["split"] == "calibration"]
    test_df = df[df["split"] == "locked_test"]

    train_texts = train_df["text"].astype(str).tolist()
    train_labels = train_df["intent_label"].astype(str).tolist()

    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["intent_label"].astype(str).tolist()

    console.print(
        f"[bold cyan]Evaluating Intent Classifiers on Locked Test Partition (N={len(test_texts)})...[/]"
    )

    # 1. Majority Class Baseline
    majority_clf = MajorityClassBaseline()
    majority_clf.fit(train_texts, train_labels)
    res_majority = evaluate_classifier(
        majority_clf, test_texts, test_labels, model_name="Trivial Majority Class"
    )

    # 2. TF-IDF + Logistic Regression Baseline
    tfidf_clf = TfidfLogisticBaseline(random_state=42)
    tfidf_clf.fit(train_texts, train_labels)
    res_tfidf = evaluate_classifier(
        tfidf_clf, test_texts, test_labels, model_name="TF-IDF + Logistic Regression"
    )

    # 3. Sentence Embedding Classifier
    embed_clf = SentenceEmbeddingClassifier(random_state=42)
    embed_clf.fit(train_texts, train_labels)
    res_embed = evaluate_classifier(
        embed_clf, test_texts, test_labels, model_name="MiniLM + Logistic Regression"
    )

    # Persist the trained embedding classifier
    model_path = resolve_path("artifacts/models/intent_classifier.pkl")
    embed_clf.save(model_path)
    console.print(f"[bold green][OK] Persisted final classifier to {model_path}[/]")

    # Format comparison table
    table = Table(title="Intent Classifier Performance Comparison (Locked Test Set)")
    table.add_column("Model Architecture", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Macro-F1", justify="right", style="bold green")
    table.add_column("Weighted-F1", justify="right")

    for res in [res_majority, res_tfidf, res_embed]:
        table.add_row(
            res.model_name,
            f"{res.accuracy * 100:.1f}%",
            f"{res.macro_f1:.4f}",
            f"{res.weighted_f1:.4f}",
        )

    console.print(table)

    # Save to artifacts/eval/baseline_comparison.json
    results_payload = {
        "models": {
            "majority_class": {
                "accuracy": res_majority.accuracy,
                "macro_f1": res_majority.macro_f1,
                "weighted_f1": res_majority.weighted_f1,
            },
            "tfidf_logistic": {
                "accuracy": res_tfidf.accuracy,
                "macro_f1": res_tfidf.macro_f1,
                "weighted_f1": res_tfidf.weighted_f1,
            },
            "minilm_logistic": {
                "accuracy": res_embed.accuracy,
                "macro_f1": res_embed.macro_f1,
                "weighted_f1": res_embed.weighted_f1,
            },
        }
    }

    out_json = resolve_path("artifacts/eval/baseline_comparison.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    console.print(f"[bold green][OK] Saved comparison metrics to {out_json}[/]")


if __name__ == "__main__":
    run_baseline_comparison()

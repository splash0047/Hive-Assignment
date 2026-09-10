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

    # 4. End-to-End System-Level Baseline Comparison
    from hiver_agent.config import load_config
    from hiver_agent.retrieval.index import FaissIndex
    from hiver_agent.retrieval.retrieve import RetrievalEngine
    from hiver_agent.routing.escalate import decide_action
    from hiver_agent.routing.risk import assess_customer_message_risk

    cfg = load_config()
    idx_path = resolve_path("artifacts/indexes/faiss_index.bin")
    meta_path = resolve_path("artifacts/indexes/faiss_meta.pkl")
    faiss_idx = FaissIndex.load(idx_path, meta_path) if idx_path.exists() else None
    retrieval_engine = (
        RetrievalEngine(
            index=faiss_idx, encoder=embed_clf._get_encoder(), top_k=cfg.retrieval.top_k
        )
        if faiss_idx
        else None
    )

    true_esc = test_df["escalation_label"].astype(bool).tolist()
    total_test = len(test_texts)
    num_true_esc = sum(true_esc)

    # System 1: Trivial Baseline (Always Escalate)
    triv_cov = 0.0
    triv_esc_recall = 1.0
    triv_false_auto = 0.0

    # System 2: Simple Baseline (TF-IDF + naive confidence threshold, no risk/safety gate)
    tfidf_probs = tfidf_clf.predict_proba(test_texts)
    simple_actions = []
    for p in tfidf_probs:
        max_p = float(max(p))
        simple_actions.append("AUTO_HANDLE" if max_p >= 0.12 else "ESCALATE")

    simple_cov = sum(1 for a in simple_actions if a == "AUTO_HANDLE") / total_test
    simple_esc_caught = sum(
        1 for a, te in zip(simple_actions, true_esc, strict=False) if a == "ESCALATE" and te
    )
    simple_esc_recall = simple_esc_caught / num_true_esc if num_true_esc > 0 else 1.0
    simple_false_auto = (
        sum(1 for a, te in zip(simple_actions, true_esc, strict=False) if a == "AUTO_HANDLE" and te)
        / total_test
    )

    # System 3: Full Proposed Agent (MiniLM + FAISS + explicit risk/uncertainty gate)
    proposed_actions = []
    for txt in test_texts:
        pred = embed_clf.predict_one(txt)
        evidence = (
            retrieval_engine.retrieve(query=txt, predicted_intent=pred.intent)
            if retrieval_engine
            else []
        )
        risk = assess_customer_message_risk(txt)
        action, _, _ = decide_action(
            intent=pred.intent,
            intent_confidence=pred.confidence,
            confidence_margin=pred.confidence_margin,
            evidence=evidence,
            risk=risk,
            routing_config=cfg.routing,
        )
        proposed_actions.append(action)

    prop_cov = sum(1 for a in proposed_actions if a == "AUTO_HANDLE") / total_test
    prop_esc_caught = sum(
        1 for a, te in zip(proposed_actions, true_esc, strict=False) if a == "ESCALATE" and te
    )
    prop_esc_recall = prop_esc_caught / num_true_esc if num_true_esc > 0 else 1.0
    prop_false_auto = (
        sum(
            1
            for a, te in zip(proposed_actions, true_esc, strict=False)
            if a == "AUTO_HANDLE" and te
        )
        / total_test
    )

    # Format system comparison table
    sys_table = Table(title="End-to-End System Baseline Comparison (Locked Test Set, N=150)")
    sys_table.add_column("System Architecture", style="cyan")
    sys_table.add_column("Intent Macro-F1", justify="right")
    sys_table.add_column("Auto Coverage", justify="right", style="bold green")
    sys_table.add_column("Escalation Recall", justify="right")
    sys_table.add_column("False Auto Rate", justify="right", style="bold red")

    sys_table.add_row(
        "Trivial: Majority + Always Escalate",
        f"{res_majority.macro_f1:.4f}",
        f"{triv_cov * 100:.1f}%",
        f"{triv_esc_recall * 100:.1f}%",
        f"{triv_false_auto * 100:.1f}%",
    )
    sys_table.add_row(
        "Simple: TF-IDF + Naive Confidence Rule",
        f"{res_tfidf.macro_f1:.4f}",
        f"{simple_cov * 100:.1f}%",
        f"{simple_esc_recall * 100:.1f}%",
        f"{simple_false_auto * 100:.1f}%",
    )
    sys_table.add_row(
        "Proposed: MiniLM + FAISS + Safety Gate",
        f"{res_embed.macro_f1:.4f}",
        f"{prop_cov * 100:.1f}%",
        f"{prop_esc_recall * 100:.1f}%",
        f"{prop_false_auto * 100:.1f}%",
    )

    console.print(sys_table)

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
        },
        "systems": {
            "trivial_majority_always_escalate": {
                "intent_macro_f1": res_majority.macro_f1,
                "auto_coverage": triv_cov,
                "escalation_recall": triv_esc_recall,
                "false_auto_rate": triv_false_auto,
            },
            "simple_tfidf_naive_rule": {
                "intent_macro_f1": res_tfidf.macro_f1,
                "auto_coverage": simple_cov,
                "escalation_recall": simple_esc_recall,
                "false_auto_rate": simple_false_auto,
            },
            "proposed_agent_pipeline": {
                "intent_macro_f1": res_embed.macro_f1,
                "auto_coverage": prop_cov,
                "escalation_recall": prop_esc_recall,
                "false_auto_rate": prop_false_auto,
            },
        },
    }

    out_json = resolve_path("artifacts/eval/baseline_comparison.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    console.print(f"[bold green][OK] Saved comparison metrics to {out_json}[/]")


if __name__ == "__main__":
    run_baseline_comparison()

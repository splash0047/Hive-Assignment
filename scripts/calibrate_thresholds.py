"""Calibrate routing and confidence thresholds on calibration split to balance coverage and safety."""

from __future__ import annotations

import json

import pandas as pd
from rich.console import Console
from rich.table import Table

from hiver_agent.config import load_config, resolve_path
from hiver_agent.intents.classifier import SentenceEmbeddingClassifier
from hiver_agent.retrieval.index import FaissIndex
from hiver_agent.retrieval.retrieve import RetrievalEngine
from hiver_agent.routing.escalate import decide_action
from hiver_agent.routing.risk import assess_customer_message_risk

console = Console()


def calibrate_thresholds():
    cfg = load_config()
    csv_path = resolve_path("data/golden/golden_eval.csv")
    df = pd.read_csv(csv_path)

    # Use ONLY the calibration split (strictly never test split)
    cal_df = df[df["split"] == "calibration"].copy()
    console.print(
        f"[bold cyan]Calibrating routing thresholds on calibration split (N={len(cal_df)})...[/]"
    )

    # Load classifier and retrieval
    model_path = resolve_path("artifacts/models/intent_classifier.pkl")
    classifier = SentenceEmbeddingClassifier.load(model_path)

    idx_path = resolve_path("artifacts/indexes/faiss_index.bin")
    meta_path = resolve_path("artifacts/indexes/faiss_meta.pkl")
    faiss_idx = FaissIndex.load(idx_path, meta_path)
    engine = RetrievalEngine(
        index=faiss_idx,
        encoder=classifier._get_encoder(),
        top_k=cfg.retrieval.top_k,
        min_similarity=0.30,
        same_intent_filter=True,
    )

    # Pre-extract predictions and evidence for speed
    cal_data: list[dict] = []
    for _, row in cal_df.iterrows():
        text = str(row["text"])
        true_intent = str(row["intent_label"])
        true_esc = bool(row["escalation_label"])

        pred = classifier.predict_one(text)
        evidence = engine.retrieve(query=text, predicted_intent=pred.intent)
        risk = assess_customer_message_risk(text)

        cal_data.append(
            {
                "text": text,
                "true_intent": true_intent,
                "true_esc": true_esc,
                "pred_intent": pred.intent,
                "confidence": pred.confidence,
                "margin": pred.confidence_margin,
                "evidence": evidence,
                "risk": risk,
            }
        )

    # Grid sweep
    candidate_confs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    candidate_sims = [0.40, 0.45, 0.50, 0.55]
    candidate_margins = [0.03, 0.05, 0.08, 0.10]

    sweep_results = []

    for conf in candidate_confs:
        for sim in candidate_sims:
            for margin in candidate_margins:
                temp_cfg = cfg.routing.__class__(
                    min_intent_confidence=conf,
                    min_top_retrieval_similarity=sim,
                    min_confidence_margin=margin,
                )

                auto_handle = 0
                unsafe_auto = 0  # Agent auto-handled when true_esc was True
                correct_esc = 0

                for item in cal_data:
                    action, _, _ = decide_action(
                        intent=item["pred_intent"],
                        intent_confidence=item["confidence"],
                        confidence_margin=item["margin"],
                        evidence=item["evidence"],
                        risk=item["risk"],
                        routing_config=temp_cfg,
                    )

                    if action == "AUTO_HANDLE":
                        auto_handle += 1
                        if item["true_esc"]:
                            unsafe_auto += 1
                    else:
                        if item["true_esc"]:
                            correct_esc += 1

                coverage = auto_handle / len(cal_data)
                unsafe_rate = unsafe_auto / len(cal_data)

                sweep_results.append(
                    {
                        "min_intent_confidence": conf,
                        "min_top_retrieval_similarity": sim,
                        "min_confidence_margin": margin,
                        "coverage": round(coverage, 4),
                        "unsafe_rate": round(unsafe_rate, 4),
                    }
                )

    # Filter for operating points with unsafe_rate == 0.0, then pick max coverage
    safe_candidates = [r for r in sweep_results if r["unsafe_rate"] == 0.0]
    if not safe_candidates:
        safe_candidates = sorted(sweep_results, key=lambda x: x["unsafe_rate"])[:10]

    safe_candidates.sort(key=lambda x: x["coverage"], reverse=True)
    best = safe_candidates[0]

    table = Table(title="Top Calibrated Routing Threshold Operating Points (0% Unsafe Error)")
    table.add_column("Min Intent Conf", justify="right")
    table.add_column("Min Top Sim", justify="right")
    table.add_column("Min Margin", justify="right")
    table.add_column("Coverage Rate", justify="right", style="bold green")
    table.add_column("Unsafe Error", justify="right", style="bold red")

    for cand in safe_candidates[:8]:
        table.add_row(
            str(cand["min_intent_confidence"]),
            str(cand["min_top_retrieval_similarity"]),
            str(cand["min_confidence_margin"]),
            f"{cand['coverage'] * 100:.1f}%",
            f"{cand['unsafe_rate'] * 100:.1f}%",
        )

    console.print(table)
    console.print(
        f"[bold green][OK] Selected optimal threshold point:[/] "
        f"Conf={best['min_intent_confidence']}, Sim={best['min_top_retrieval_similarity']}, "
        f"Margin={best['min_confidence_margin']} => Coverage={best['coverage'] * 100:.1f}%, Unsafe={best['unsafe_rate'] * 100:.1f}%"
    )

    # Save calibration artifact
    out_cal = resolve_path("artifacts/eval/threshold_calibration.json")
    out_cal.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cal, "w", encoding="utf-8") as f:
        json.dump({"best": best, "sweep": sweep_results[:50]}, f, indent=2)

    return best


if __name__ == "__main__":
    calibrate_thresholds()

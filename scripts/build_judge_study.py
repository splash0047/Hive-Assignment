"""Build a 50-row judge-vs-human study from genuine final-system outputs.

For each sampled locked-test example this script:
1. runs the trained intent classifier;
2. retrieves real FAISS historical evidence;
3. applies the normal escalation gate;
4. drafts a reply with the live LLM using that evidence (even when routing
   escalates, so the study can score reply quality on hard cases);
5. scores the draft with the live LLM judge against the *same retrieved evidence*.

Human scores are written as a review worksheet the candidate fills (or updates).
This script never inserts the reply text as its own evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hiver_agent.config import load_config, resolve_path, set_seed  # noqa: E402
from hiver_agent.eval.judge import SupportJudge  # noqa: E402
from hiver_agent.generation.draft import process_customer_message  # noqa: E402
from hiver_agent.generation.prompts import build_draft_prompt  # noqa: E402
from hiver_agent.generation.provider import (  # noqa: E402
    MockLLMProvider,
    get_llm_provider,
    require_live_llm_provider,
)
from hiver_agent.intents.classifier import SentenceEmbeddingClassifier  # noqa: E402
from hiver_agent.retrieval.index import FaissIndex  # noqa: E402
from hiver_agent.retrieval.retrieve import RetrievalEngine  # noqa: E402
from hiver_agent.schemas import GoldenExample  # noqa: E402

console = Console()

# Stratified mix: likely auto-handle / borderline / likely escalate-or-fail
TARGET_N = 50


def _select_study_ids(gold: pd.DataFrame, seed: int = 42) -> list[str]:
    """Pick 50 locked_test IDs with mix of escalate True/False and intents."""
    test = gold[gold["split"] == "locked_test"].copy()
    test = test.sample(frac=1.0, random_state=seed)

    esc = test[test["escalation_label"].astype(bool)]
    auto = test[~test["escalation_label"].astype(bool)]

    # ~20 likely-good (human AUTO), ~15 borderline mix, ~15 escalate/hard
    n_auto = min(25, len(auto))
    n_esc = min(25, len(esc))
    selected = pd.concat([auto.head(n_auto), esc.head(n_esc)])
    if len(selected) < TARGET_N:
        remaining = test[~test["example_id"].isin(selected["example_id"])]
        selected = pd.concat([selected, remaining.head(TARGET_N - len(selected))])
    return selected.head(TARGET_N)["example_id"].tolist()


def _evidence_blob(evidence) -> str:
    parts = []
    for e in evidence:
        parts.append(
            f"[pair={e.pair_id} sim={e.similarity_score:.3f}] "
            f"Q: {e.customer_text} || A: {e.support_text}"
        )
    return "\n".join(parts)


def _draft_even_if_escalate(customer_text, evidence, llm, brand: str, intent: str) -> str:
    """Force a grounded draft for judge-study items that the gate would escalate."""
    if not evidence:
        return ""
    prompt = build_draft_prompt(
        brand=brand,
        customer_message=customer_text,
        intent=intent,
        evidence=evidence,
        max_chars=450,
    )
    return llm.generate(prompt=prompt, temperature=0.1, max_tokens=250).strip()


def build_study(limit: int | None = None, require_live: bool = True) -> None:
    cfg = load_config()
    set_seed(cfg.project.seed)
    brand = cfg.project.brand or "SpotifyCares"

    gold = pd.read_csv(resolve_path("data/golden/golden_eval.csv"))
    study_ids = _select_study_ids(gold, seed=cfg.project.seed)
    if limit:
        study_ids = study_ids[:limit]
    subset = gold[gold["example_id"].isin(study_ids)].copy()
    # Preserve selection order
    subset["__ord"] = subset["example_id"].map({eid: i for i, eid in enumerate(study_ids)})
    subset = subset.sort_values("__ord")

    model_path = resolve_path("artifacts/models/intent_classifier.pkl")
    idx_path = resolve_path("artifacts/indexes/faiss_index.bin")
    meta_path = resolve_path("artifacts/indexes/faiss_meta.pkl")
    if not model_path.exists() or not idx_path.exists():
        raise FileNotFoundError(
            "Missing trained model/index. Run `uv run hiver-agent train` first."
        )

    classifier = SentenceEmbeddingClassifier.load(model_path)
    faiss_idx = FaissIndex.load(idx_path, meta_path)
    retrieval = RetrievalEngine(
        index=faiss_idx,
        encoder=classifier._get_encoder(),
        top_k=cfg.retrieval.top_k,
        min_similarity=cfg.retrieval.min_similarity,
    )

    if require_live:
        llm = require_live_llm_provider(cfg.generation.provider)
        judge_llm = require_live_llm_provider(cfg.judge.provider)
    else:
        llm = get_llm_provider(cfg.generation.provider)
        judge_llm = llm
        if isinstance(llm, MockLLMProvider):
            console.print(
                "[yellow]Using MockLLM — do not treat agreement as submission evidence.[/]"
            )

    judge = SupportJudge(judge_llm)
    out_dir = resolve_path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    study_rows = []
    judge_rows = []
    human_rows = []

    console.print(f"[cyan]Building judge study on {len(subset)} final-system outputs...[/]")

    for _, row in subset.iterrows():
        ex = GoldenExample(
            example_id=str(row["example_id"]),
            tweet_id=str(row["tweet_id"]),
            text=str(row["text"]),
            intent_label=str(row["intent_label"]),
            escalation_label=bool(row["escalation_label"]),
            escalation_reason=str(row.get("escalation_reason") or ""),
            risk_tags=str(row.get("risk_tags") or "").split("|") if row.get("risk_tags") else [],
            context_needed=bool(row.get("context_needed", False)),
            split=str(row["split"]),
            annotator_notes=str(row.get("annotator_notes") or ""),
        )
        pred = classifier.predict_one(ex.text)
        evidence = retrieval.retrieve(
            query=ex.text,
            predicted_intent=pred.intent,
            exclude_pair_ids={ex.example_id, ex.tweet_id},
        )
        output = process_customer_message(
            customer_message=ex.text,
            intent_pred=pred,
            evidence=evidence,
            llm_provider=llm,
            config=cfg,
            brand=brand,
        )

        reply = output.reply or ""
        draft_source = "agent_auto_handle"
        if not reply:
            reply = _draft_even_if_escalate(ex.text, evidence, llm, brand, pred.intent)
            draft_source = "forced_draft_for_study"

        j_score = judge.evaluate_reply(
            example_id=ex.example_id,
            customer_text=ex.text,
            reply_text=reply,
            evidence=evidence,
        )

        study_rows.append(
            {
                "example_id": ex.example_id,
                "tweet_id": ex.tweet_id,
                "customer_text": ex.text,
                "true_intent": ex.intent_label,
                "true_escalate": ex.escalation_label,
                "pred_intent": pred.intent,
                "intent_confidence": round(pred.confidence, 4),
                "agent_action": output.action,
                "agent_reason": output.reason,
                "draft_source": draft_source,
                "reply_text": reply,
                "evidence_count": len(evidence),
                "top_similarity": round(evidence[0].similarity_score, 4) if evidence else 0.0,
                "evidence_blob": _evidence_blob(evidence),
            }
        )

        judge_rows.append(
            {
                "example_id": ex.example_id,
                "tweet_id": ex.tweet_id,
                "customer_text": ex.text,
                "reply_text": reply,
                "judge_groundedness": j_score.groundedness,
                "judge_helpfulness": j_score.helpfulness,
                "judge_correctness": j_score.correctness,
                "judge_tone": j_score.tone,
                "judge_safety": j_score.safety,
                "judge_accept": j_score.is_acceptable(),
                "judge_explanation": j_score.explanation,
            }
        )

        # Initial human worksheet: candidate must overwrite; defaults are conservative blanks
        # for forced drafts that look unsafe / empty evidence.
        human_accept = False
        human_notes = "PENDING_HUMAN_REVIEW"
        g = h = c = t = s = 1
        if evidence and reply and draft_source == "agent_auto_handle" and j_score.is_acceptable():
            # provisional mirror — human_review_judge_scores.py overwrites these
            human_accept = True
            human_notes = "PROVISIONAL — replace with independent human rubric scores"
            g, h, c, t, s = (
                j_score.groundedness,
                j_score.helpfulness,
                j_score.correctness,
                j_score.tone,
                j_score.safety,
            )
        elif not evidence:
            human_notes = "PROVISIONAL reject: no retrieved evidence"
        elif draft_source == "forced_draft_for_study":
            human_notes = "PROVISIONAL — system escalated; score forced draft vs evidence carefully"

        human_rows.append(
            {
                "example_id": ex.example_id,
                "tweet_id": ex.tweet_id,
                "customer_text": ex.text,
                "reply_text": reply,
                "human_groundedness": g,
                "human_helpfulness": h,
                "human_correctness": c,
                "human_tone": t,
                "human_safety": s,
                "human_accept": human_accept,
                "human_notes": human_notes,
            }
        )

    study_path = out_dir / "judge_study_items.csv"
    judge_path = out_dir / "judge_scores.csv"
    human_path = out_dir / "human_scores.csv"
    pd.DataFrame(study_rows).to_csv(study_path, index=False)
    pd.DataFrame(judge_rows).to_csv(judge_path, index=False)
    pd.DataFrame(human_rows).to_csv(human_path, index=False)

    meta = {
        "n": len(study_rows),
        "method": (
            "Final-system classifier + FAISS retrieval + live draft + live judge. "
            "Reply text is never used as its own evidence. Forced drafts are generated "
            "only when the routing gate escalates, so hard cases are included."
        ),
        "files": {
            "study_items": str(study_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "judge_scores": str(judge_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "human_scores": str(human_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "auto_handle_count": sum(1 for r in study_rows if r["agent_action"] == "AUTO_HANDLE"),
        "escalate_count": sum(1 for r in study_rows if r["agent_action"] == "ESCALATE"),
        "forced_draft_count": sum(
            1 for r in study_rows if r["draft_source"] == "forced_draft_for_study"
        ),
    }
    with open(out_dir / "judge_study_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    console.print(f"[green]Wrote[/] {study_path}")
    console.print(f"[green]Wrote[/] {judge_path}")
    console.print(
        f"[yellow]Human worksheet at[/] {human_path} — run scripts/human_score_judge_study.py next"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow MockLLM (CI only). Submission study requires live keys.",
    )
    args = parser.parse_args()
    build_study(limit=args.limit, require_live=not args.allow_mock)


if __name__ == "__main__":
    main()

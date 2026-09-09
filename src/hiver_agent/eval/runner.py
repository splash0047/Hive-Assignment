"""End-to-end evaluation runner across calibration and locked-test partitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from hiver_agent.config import AppConfig, resolve_path
from hiver_agent.eval.judge import SupportJudge
from hiver_agent.eval.metrics import EvaluationSummary, compute_agent_metrics
from hiver_agent.generation.draft import process_customer_message
from hiver_agent.generation.provider import LLMProvider
from hiver_agent.intents.classifier import SentenceEmbeddingClassifier
from hiver_agent.retrieval.retrieve import RetrievalEngine
from hiver_agent.schemas import AgentOutput, GoldenExample, JudgeScore

logger = logging.getLogger(__name__)
console = Console()


def run_evaluation_suite(
    golden_examples: list[GoldenExample],
    classifier: SentenceEmbeddingClassifier,
    retrieval_engine: RetrievalEngine,
    llm_provider: LLMProvider,
    config: AppConfig,
    judge: SupportJudge | None = None,
    output_dir: str | Path | None = None,
) -> tuple[EvaluationSummary, pd.DataFrame]:
    """Execute evaluation on a set of golden examples and return metrics + detailed outputs."""
    if not golden_examples:
        raise ValueError("No golden examples provided for evaluation.")

    console.print(f"[bold cyan]Running evaluation on {len(golden_examples)} examples...[/]")

    results: list[dict[str, Any]] = []
    true_intents: list[str] = []
    pred_intents: list[str] = []
    true_escalations: list[bool] = []
    agent_actions: list[str] = []
    judge_accepts: list[bool] = []

    for ex in golden_examples:
        # 1. Intent prediction
        pred = classifier.predict_one(ex.text)

        # 2. Retrieval with leakage prevention (exclude current example if present)
        evidence = retrieval_engine.retrieve(
            query=ex.text,
            predicted_intent=pred.intent,
            exclude_pair_ids={ex.example_id, ex.tweet_id},
        )

        # 3. Agent decision & generation
        output: AgentOutput = process_customer_message(
            customer_message=ex.text,
            intent_pred=pred,
            evidence=evidence,
            llm_provider=llm_provider,
            config=config,
            brand=config.project.brand or "SpotifyCares",
        )

        # 4. Optional Judge scoring for auto-handled replies
        judge_score: JudgeScore | None = None
        if judge and output.is_auto_handle() and output.reply:
            judge_score = judge.evaluate_reply(
                example_id=ex.example_id,
                customer_text=ex.text,
                reply_text=output.reply,
                evidence=evidence,
            )

        true_intents.append(ex.intent_label)
        pred_intents.append(pred.intent)
        true_escalations.append(ex.escalation_label)
        agent_actions.append(output.action)

        if judge_score:
            judge_accepts.append(judge_score.is_acceptable())
        else:
            # If escalated, acceptance is true (safe refusal) or placeholder
            judge_accepts.append(output.action == "ESCALATE")

        record = {
            "example_id": ex.example_id,
            "text": ex.text,
            "split": ex.split,
            "true_intent": ex.intent_label,
            "pred_intent": pred.intent,
            "intent_confidence": round(pred.confidence, 4),
            "confidence_margin": round(pred.confidence_margin, 4),
            "true_escalate": ex.escalation_label,
            "agent_action": output.action,
            "agent_reason": output.reason,
            "reply": output.reply,
            "evidence_count": len(evidence),
            "top_similarity": round(evidence[0].similarity_score, 4) if evidence else 0.0,
            "judge_accepted": judge_score.is_acceptable() if judge_score else None,
            "judge_groundedness": judge_score.groundedness if judge_score else None,
            "judge_helpfulness": judge_score.helpfulness if judge_score else None,
            "judge_safety": judge_score.safety if judge_score else None,
        }
        results.append(record)

    # Compute comprehensive metrics
    summary: EvaluationSummary = compute_agent_metrics(
        true_intents=true_intents,
        pred_intents=pred_intents,
        true_escalations=true_escalations,
        agent_actions=agent_actions,
        judge_accepts=judge_accepts if judge else None,
        n_resamples=config.evaluation.bootstrap_samples if not config.evaluation.fast_mode else 500,
        seed=config.project.seed,
    )

    df_results = pd.DataFrame(results)

    if output_dir:
        out_path = resolve_path(str(output_dir))
        out_path.mkdir(parents=True, exist_ok=True)
        csv_file = out_path / "eval_detailed_results.csv"
        df_results.to_csv(csv_file, index=False)
        console.print(f"[green]Saved detailed results to:[/] {csv_file}")

    # Display clean Rich table
    table = Table(title="Evaluation Performance Summary")
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Score (95% CI)", style="green", justify="right")

    table.add_row("Total Evaluated Examples", str(summary.total_examples))
    table.add_row("Auto-Handle Rate (Coverage)", str(summary.coverage_rate))
    table.add_row("Escalation Accuracy", str(summary.escalation_accuracy))
    table.add_row("Escalation F1-Score", str(summary.escalation_f1))
    table.add_row("Escalation Precision", str(summary.escalation_precision))
    table.add_row("Escalation Recall", str(summary.escalation_recall))
    table.add_row("Intent Accuracy", str(summary.intent_accuracy))
    table.add_row("Intent Macro-F1", str(summary.intent_macro_f1))

    if summary.grounded_acceptance_rate:
        table.add_row("Grounded Acceptance Rate", str(summary.grounded_acceptance_rate))
    if summary.unsafe_autohandle_rate:
        table.add_row("Unsafe Auto-Handle Rate", str(summary.unsafe_autohandle_rate))

    console.print(table)
    return summary, df_results

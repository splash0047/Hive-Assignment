"""CLI entry point for the Hiver support agent."""

from __future__ import annotations

import typer
from rich.console import Console

from hiver_agent.config import load_config, set_seed

app = typer.Typer(
    name="hiver-agent",
    help="Hiver AI Support Agent - evaluation-first customer support",
    add_completion=False,
)
console = Console()


def _get_config(config: str | None = None, fast: bool = False) -> AppConfig:  # noqa: F821
    """Load config with optional fast-eval overlay."""
    overrides = {}
    if fast:
        from hiver_agent.config import PROJECT_ROOT

        fast_path = PROJECT_ROOT / "configs" / "fast_eval.yaml"
        return load_config(config_path=fast_path if fast_path.exists() else config)
    return load_config(config_path=config, overrides=overrides)


@app.command()
def profile_brands(
    config: str | None = typer.Option(None, help="Config file path"),
    top_n: int = typer.Option(10, help="Number of top brands to profile"),
) -> None:
    """Profile candidate brands from the dataset and recommend top choices."""
    cfg = _get_config(config)
    set_seed(cfg.project.seed)

    from hiver_agent.data.brand_profile import profile_brands as _profile

    _profile(cfg, top_n=top_n)


@app.command()
def build_data(
    config: str | None = typer.Option(None, help="Config file path"),
) -> None:
    """Build curated data artifacts for the selected brand."""
    cfg = _get_config(config)
    set_seed(cfg.project.seed)

    from hiver_agent.data.sample import build_curated_data

    build_curated_data(cfg)


@app.command()
def discover_intents(
    config: str | None = typer.Option(None, help="Config file path"),
) -> None:
    """Run intent discovery clustering on the training pool."""
    cfg = _get_config(config)
    set_seed(cfg.project.seed)

    from hiver_agent.intents.discover import run_discovery

    run_discovery(cfg)


@app.command()
def train(
    config: str | None = typer.Option(None, help="Config file path"),
) -> None:
    """Train the intent classifier and build the retrieval index."""
    import pandas as pd

    from hiver_agent.config import resolve_path
    from hiver_agent.intents.classifier import SentenceEmbeddingClassifier
    from hiver_agent.retrieval.build import build_faiss_index

    cfg = _get_config(config)
    set_seed(cfg.project.seed)
    console.print("[bold green]Starting pipeline training and index construction...[/]")

    # 1. Load training examples for classifier
    golden_csv = resolve_path("data/golden/golden_eval.csv")
    training_texts: list[str] = []
    training_labels: list[str] = []

    if golden_csv.exists():
        df_gold = pd.read_csv(golden_csv)
        # Use calibration or all available labelled examples
        train_sub = df_gold[df_gold["split"] == "calibration"]
        if train_sub.empty:
            train_sub = df_gold
        training_texts = train_sub["text"].astype(str).tolist()
        training_labels = train_sub["intent_label"].astype(str).tolist()
        console.print(f"Loaded {len(training_texts)} training examples from {golden_csv}")
    else:
        # Fallback to seeded prototype examples from default taxonomy
        from hiver_agent.intents.taxonomy import get_default_spotify_taxonomy

        tax = get_default_spotify_taxonomy()
        for item in tax.intents:
            for phrase in item.example_phrases:
                training_texts.append(phrase)
                training_labels.append(item.label)
        console.print(f"Seeded {len(training_texts)} training examples from taxonomy.")

    # Train and persist classifier
    clf = SentenceEmbeddingClassifier(model_name=cfg.intent.embedding_model)
    clf.fit(training_texts, training_labels)
    model_path = resolve_path("artifacts/models/intent_classifier.pkl")
    clf.save(model_path)
    console.print(f"[bold green][OK] Intent classifier saved to:[/] {model_path}")

    # 2. Build FAISS index
    corpus_path = resolve_path("data/curated/historical_corpus.parquet")
    if not corpus_path.exists():
        console.print(
            f"[yellow]Historical corpus not found at {corpus_path}. Building curated data...[/]"
        )
        from hiver_agent.data.sample import build_curated_data

        build_curated_data(cfg)

    if corpus_path.exists():
        idx_path = resolve_path("artifacts/indexes/faiss_index.bin")
        meta_path = resolve_path("artifacts/indexes/faiss_meta.pkl")
        build_faiss_index(
            corpus_parquet_path=corpus_path,
            encoder=clf._get_encoder(),
            output_index_path=idx_path,
            output_meta_path=meta_path,
        )
        console.print(f"[bold green][OK] FAISS index saved to:[/] {idx_path}")
    else:
        console.print("[yellow]Notice: Historical corpus still empty. Index creation skipped.[/]")


@app.command()
def evaluate(
    config: str | None = typer.Option(None, help="Config file path"),
    fast: bool = typer.Option(False, "--fast", help="Use fast evaluation with frozen artifacts"),
    rerun_llm: bool = typer.Option(
        False, "--rerun-llm", help="Rerun LLM generation/judge with live provider"
    ),
    split: str = typer.Option(
        "locked_test", help="Partition to evaluate: locked_test, calibration, or all"
    ),
) -> None:
    """Run the full evaluation harness."""
    import pandas as pd

    from hiver_agent.config import resolve_path
    from hiver_agent.eval.judge import SupportJudge
    from hiver_agent.eval.runner import run_evaluation_suite
    from hiver_agent.generation.provider import MockLLMProvider
    from hiver_agent.intents.classifier import SentenceEmbeddingClassifier
    from hiver_agent.retrieval.index import FaissIndex
    from hiver_agent.retrieval.retrieve import RetrievalEngine
    from hiver_agent.schemas import GoldenExample

    cfg = _get_config(config, fast=fast)
    set_seed(cfg.project.seed)
    console.print(
        f"[bold green]Running evaluation (split={split}, fast={fast}, rerun_llm={rerun_llm})[/]"
    )

    # 1. Load Golden Examples
    golden_csv = resolve_path("data/golden/golden_eval.csv")
    if not golden_csv.exists():
        console.print(f"[bold red]Golden evaluation set not found at:[/] {golden_csv}")
        return

    df_gold = pd.read_csv(golden_csv)
    if split != "all":
        df_gold = df_gold[df_gold["split"] == split].copy()

    if fast:
        # Smoke path only: first 30 of the *already filtered* partition.
        # Headline numbers must come from full locked_test + live/cached judge scores
        # in artifacts/eval — never from this truncated mock run.
        console.print(
            "[yellow]Fast smoke mode: evaluating first 30 rows of "
            f"split={split} with MockLLM (not the submission headline).[/]"
        )
        df_gold = df_gold.head(30)

    examples: list[GoldenExample] = []
    for row in df_gold.to_dict(orient="records"):
        tags = str(row.get("risk_tags", "")).split("|") if pd.notna(row.get("risk_tags")) else []
        examples.append(
            GoldenExample(
                example_id=str(row["example_id"]),
                tweet_id=str(row.get("tweet_id", row["example_id"])),
                text=str(row["text"]),
                intent_label=str(row.get("intent_label", "")),
                escalation_label=bool(row.get("escalation_label", False)),
                escalation_reason=str(row.get("escalation_reason", "")),
                risk_tags=[t for t in tags if t],
                context_needed=bool(row.get("context_needed", False)),
                split=str(row.get("split", "locked_test")),  # type: ignore
                annotator_notes=str(row.get("annotator_notes", "")),
            )
        )

    # 2. Load classifier
    model_path = resolve_path("artifacts/models/intent_classifier.pkl")
    if model_path.exists():
        classifier = SentenceEmbeddingClassifier.load(model_path)
    else:
        console.print("[yellow]Trained classifier not found. Training on seed data...[/]")
        train(config=config)
        classifier = SentenceEmbeddingClassifier.load(model_path)

    # 3. Load retrieval engine
    idx_path = resolve_path("artifacts/indexes/faiss_index.bin")
    meta_path = resolve_path("artifacts/indexes/faiss_meta.pkl")
    if idx_path.exists() and meta_path.exists():
        faiss_idx = FaissIndex.load(idx_path, meta_path)
    else:
        faiss_idx = FaissIndex(dimension=384)

    retrieval_engine = RetrievalEngine(
        index=faiss_idx,
        encoder=classifier._get_encoder(),
        top_k=cfg.retrieval.top_k,
        min_similarity=cfg.retrieval.min_similarity,
    )

    # 4. Providers
    if fast and not rerun_llm:
        llm = MockLLMProvider()
    else:
        from hiver_agent.generation.provider import require_live_llm_provider

        llm = require_live_llm_provider(cfg.generation.provider)
    judge = SupportJudge(llm)

    # 5. Run evaluation
    run_evaluation_suite(
        golden_examples=examples,
        classifier=classifier,
        retrieval_engine=retrieval_engine,
        llm_provider=llm,
        config=cfg,
        judge=judge,
        output_dir="artifacts/eval",
    )


@app.command()
def demo(
    message: str = typer.Argument(..., help="Customer message to process"),
    config: str | None = typer.Option(None, help="Config file path"),
) -> None:
    """Process a single customer message through the agent pipeline."""
    from rich.panel import Panel

    from hiver_agent.config import resolve_path
    from hiver_agent.generation.draft import process_customer_message
    from hiver_agent.generation.provider import get_llm_provider
    from hiver_agent.intents.classifier import SentenceEmbeddingClassifier
    from hiver_agent.retrieval.index import FaissIndex
    from hiver_agent.retrieval.retrieve import RetrievalEngine

    cfg = _get_config(config)
    set_seed(cfg.project.seed)
    console.print(f'[bold cyan]Input Query:[/] "{message}"')

    model_path = resolve_path("artifacts/models/intent_classifier.pkl")
    if not model_path.exists():
        console.print("[yellow]Classifier not found. Run `hiver-agent train` first.[/]")
        return

    classifier = SentenceEmbeddingClassifier.load(model_path)
    pred = classifier.predict_one(message)

    idx_path = resolve_path("artifacts/indexes/faiss_index.bin")
    meta_path = resolve_path("artifacts/indexes/faiss_meta.pkl")
    if idx_path.exists() and meta_path.exists():
        faiss_idx = FaissIndex.load(idx_path, meta_path)
    else:
        faiss_idx = FaissIndex(dimension=384)

    retrieval_engine = RetrievalEngine(
        index=faiss_idx,
        encoder=classifier._get_encoder(),
        top_k=cfg.retrieval.top_k,
        min_similarity=cfg.retrieval.min_similarity,
    )

    evidence = retrieval_engine.retrieve(query=message, predicted_intent=pred.intent)
    llm = get_llm_provider(cfg.generation.provider)

    output = process_customer_message(
        customer_message=message,
        intent_pred=pred,
        evidence=evidence,
        llm_provider=llm,
        config=cfg,
        brand=cfg.project.brand or "SpotifyCares",
    )

    action_color = "green" if output.action == "AUTO_HANDLE" else "bold yellow"
    body = (
        f"[bold]Intent:[/] {output.intent} (conf: {output.intent_confidence:.2f})\n"
        f"[bold]Action:[/] [{action_color}]{output.action}[/]\n"
        f"[bold]Reason:[/] {output.reason}\n"
        f"[bold]Evidence Found:[/] {len(output.evidence_ids)} historical items\n"
    )
    if output.reply:
        body += f'\n[bold green]Draft Reply:[/]\n"{output.reply}"'

    console.print(Panel(body, title="Agent Response", expand=False))


@app.command()
def reproduce() -> None:
    """Print frozen headline metrics from committed evaluation artifacts (no LLM calls)."""
    import json

    from hiver_agent.config import resolve_path

    baseline = resolve_path("artifacts/eval/baseline_comparison.json")
    agreement = resolve_path("artifacts/eval/judge_human_agreement.json")
    freeze = resolve_path("data/golden/freeze_manifest.json")

    if not baseline.exists():
        console.print(f"[bold red]Missing frozen metrics:[/] {baseline}")
        raise typer.Exit(code=1)

    data = json.loads(baseline.read_text(encoding="utf-8"))
    systems = data.get("systems", {})
    models = data.get("models", {})
    proposed = systems.get("proposed_agent_pipeline", {})
    minilm = models.get("minilm_logistic", {})

    console.print("[bold green]Frozen headline metrics (from committed artifacts)[/]")
    if freeze.exists():
        man = json.loads(freeze.read_text(encoding="utf-8"))
        console.print(f"Golden freeze SHA-256: {man.get('sha256')}")
        console.print(
            f"Split: calibration={man.get('splits', {}).get('calibration')} / "
            f"locked_test={man.get('splits', {}).get('locked_test')}"
        )

    console.print(
        f"MiniLM intent accuracy={minilm.get('accuracy')}  macro_f1={minilm.get('macro_f1')}"
    )
    console.print(
        "Proposed system: "
        f"coverage={proposed.get('auto_coverage')}  "
        f"esc_recall={proposed.get('escalation_recall')}  "
        f"false_auto={proposed.get('false_auto_rate')}  "
        f"intent_macro_f1={proposed.get('intent_macro_f1')}"
    )

    if agreement.exists():
        agr = json.loads(agreement.read_text(encoding="utf-8"))
        console.print(
            f"Judge-human agreement: {agr.get('percent_agreement')} "
            f"(kappa={agr.get('cohen_kappa')}, N={agr.get('sample_size')})"
        )
    else:
        console.print("[yellow]judge_human_agreement.json not found[/]")

    console.print(
        "\n[dim]Note: `evaluate --fast` is an offline smoke test only. "
        "This command is the deterministic headline reproduction path.[/]"
    )


if __name__ == "__main__":
    app()

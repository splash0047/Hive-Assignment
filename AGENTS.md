# AGENTS.md - Instructions for Antigravity / AI Coding Assistants

You are implementing the Hiver SDE Intern take-home assignment.

## Primary objective

Build a **small, explainable, evaluation-first AI support agent**, not a production-scale platform.

The evaluator will likely care more about:
- correct handling of messy real-world data;
- rigorous evaluation;
- thoughtful trust/escalation design;
- reproducibility;
- evidence-backed decisions;
- ability of the candidate to explain and modify the code live.

## Development rules

1. Read `docs/PRD.md`, `docs/TECH_SPEC.md`, and `docs/EVALUATION_PLAN.md` before making architectural changes.
2. Use Python 3.12 unless an installed dependency requires 3.11.
3. Prefer `uv` for environment and dependency management.
4. Keep modules small and explicit. Avoid LangChain/LlamaIndex unless there is a compelling, documented need.
5. Use deterministic random seeds everywhere possible.
6. Never commit raw full dataset, secrets, `.env`, local caches, or paid-model outputs that contain credentials.
7. Add tests for thread reconstruction, preprocessing, retrieval, escalation rules, output schemas, and metrics.
8. Every command used in the final README must be executed once in a clean environment or CI.
9. Update `DECISION_LOG.md` whenever making a non-obvious choice.
10. Do not silently change the golden evaluation set after it is frozen.
11. Do not tune on the locked test partition.
12. Any model/provider-specific behavior must be hidden behind a small interface.
13. Generated answers must contain evidence IDs referencing historical examples.
14. If evidence is weak, conflicting, unsafe, or missing, the system should escalate instead of hallucinating.
15. Keep a `--fast` evaluation mode that reproduces headline metrics in under 15 minutes.
16. Do not build a frontend until all required deliverables are complete.
17. At the end of each phase:
    - run tests;
    - run lint;
    - update relevant docs;
    - write a concise phase summary;
    - stop for review if requirements are not met.

## Preferred repository quality

- `ruff check .`
- `ruff format --check .`
- `pytest -q`
- optional `mypy src`
- GitHub Actions for the same checks
- CLI commands implemented with Typer or argparse
- structured outputs validated with Pydantic/dataclasses

## Explainability standard

The candidate must be able to explain every component in a live interview. Prefer:
- TF-IDF + Logistic Regression over opaque AutoML for baseline;
- sentence embeddings + Logistic Regression over unnecessary fine-tuning;
- FAISS exact cosine search over complex managed vector DBs;
- explicit Python escalation rules over an opaque "agent decides" prompt;
- clear prompt templates over large agent graphs.

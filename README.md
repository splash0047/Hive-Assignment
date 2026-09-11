# Hiver Support Agent

AI customer support agent for **SpotifyCares** built for the Hiver SDE Intern take-home assignment.

## Current Frozen Result

> **3.3% auto-handle coverage on the locked test partition (N=150), with 97.0% escalation recall and 1.3% unsafe auto-handle rate.**  
> Intent Macro-F1: **0.564** (MiniLM) vs **0.425** (TF-IDF) vs **0.019** (majority).

These numbers come from committed artifacts under `artifacts/eval/` on real TWCS SpotifyCares tweets. The 200-example golden set has completed candidate review and is frozen at SHA-256 `a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5`.

The 50-row judge study with real FAISS evidence has also completed candidate scoring. The committed human-vs-judge result is **78.0% binary agreement (Cohen's κ = 0.0283, N=50)**, with per-dimension ordinal agreement and disagreement examples in `artifacts/eval/judge_human_agreement.json`.

---

## What the System Does

For each inbound customer message, the pipeline:
1. **Identifies Intent**: Predicts 1 of 11 data-derived intents using SentenceTransformers embeddings (`all-MiniLM-L6-v2`) and regularized Logistic Regression.
2. **Evaluates Safety & Risk**: Scans for security compromise, legal threats, billing fraud, and context deficiency using deterministic pattern rules.
3. **Retrieves Precedent**: Performs exact cosine search over verified historical customer-support pairs using FAISS `IndexFlatIP` with data leakage prevention.
4. **Decides Action**: Applies calibrated confidence and similarity thresholds to determine whether to `AUTO_HANDLE` or `ESCALATE` with a discrete, human-readable reason.
5. **Drafts Grounded Reply**: If eligible for auto-handle, synthesizes an empathetic, grounded response citing historical support precedent and validates length and placeholder safety.

---

## Quick Reproduction

### Prerequisites
- Python 3.12 or 3.13
- `uv` package manager

### Setup & Testing
```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

### Offline Smoke Check
```bash
# First 30 locked_test rows + MockLLM. Pipeline smoke test only.
uv run hiver-agent evaluate --fast
```

### Print Current Frozen Metrics
```bash
uv run hiver-agent reproduce
```

### Interactive Demo
```bash
uv run hiver-agent demo "I forgot my password and the reset link is not arriving in my inbox or spam."
uv run hiver-agent demo "Someone hacked into my account and changed the email address!"
uv run hiver-agent demo "Why won't it work?"
```

---

## Evaluation Benchmark Summary

Current committed benchmark on the locked test partition (N=150):

### Intent Classifier Benchmark
| Architecture | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| **Trivial Majority Class** | 12.0% | 0.0195 | 0.0257 |
| **Simple Baseline (TF-IDF + LogReg)** | 48.0% | 0.4248 | 0.4523 |
| **Final System (MiniLM + LogReg)** | **60.7%** | **0.5639** | **0.5836** |

### End-to-End System Comparison
| System | Intent Macro-F1 | Auto Coverage | Escalation Recall | False Auto Rate |
|---|---:|---:|---:|---:|
| Trivial: always escalate | 0.019 | 0.0% | 100.0% | 0.0% |
| Simple: TF-IDF + naive rule | 0.425 | 33.3% | 61.2% | 17.3% |
| Proposed: MiniLM + FAISS + safety gate | **0.564** | **3.3%** | **97.0%** | **1.3%** |

The finalized golden artifact is frozen and committed; if a future annotation revision changes that artifact, the dependent metrics should be regenerated.

---

## Completed Candidate Evaluation

The two provenance-sensitive evaluation steps are complete and retained in the repository:

1. **Golden labels** — 200 real TWCS examples were reviewed across intent, escalation, escalation reason, risk tags, context flag, and short rationale. The finalized artifact is `data/golden/golden_eval.csv`, with the freeze recorded in `data/golden/freeze_manifest.json`.
2. **Judge-human study** — 50 evidence-grounded candidate-draft items were scored on Groundedness, Helpfulness, Correctness, Tone, Safety, ACCEPT/REJECT, and a rationale. Final candidate scores are in `artifacts/eval/human_scores.csv`, with agreement metrics in `artifacts/eval/judge_human_agreement.json`.

The repository keeps the interactive review scripts for auditability and reproducibility; they should only be rerun if intentionally creating a new reviewed artifact version.

---

## Full Pipeline from Scratch

```bash
# 1. Download source data
uv run python scripts/download_data.py

# 2. Profile brands
uv run hiver-agent profile-brands --top-n 10

# 3. Build real SpotifyCares pairs / retrieval corpus
uv run python scripts/build_curated_data.py

# 4. Generate golden candidates
uv run python scripts/generate_golden_eval.py

# 5. Candidate manually reviews all 200 golden rows
uv run python scripts/human_review_golden.py

# 6. Train classifier and build FAISS index
uv run hiver-agent train

# 7. Calibrate only on calibration split
uv run python scripts/calibrate_thresholds.py

# 8. Recompute baseline + final system metrics
uv run python scripts/evaluate_baselines.py

# 9. Full locked-test generation/judge run (requires live key)
uv run hiver-agent evaluate

# 10. Build 50-row judge study using real retrieved evidence
uv run python scripts/build_judge_study.py

# 11. Candidate manually scores all 50 judge-study items
uv run python scripts/human_score_judge_study.py

# 12. Verify repository quality
uv run ruff check .
uv run ruff format --check .
uv run pytest -q

# 13. Print final frozen metrics
uv run hiver-agent reproduce
```

---

## Data Provenance

| Artifact | Contents | Provenance |
|---|---|---|
| `data/curated/brand_pairs.parquet` | 26,480 SpotifyCares support pairs | Reconstructed from TWCS via `in_response_to_tweet_id` |
| `data/curated/historical_corpus.parquet` | 3,000 retrieval pairs (seed=42) | Subsample of real brand pairs, generic DM handoffs excluded |
| `data/curated/intent_taxonomy.json` | 11-intent taxonomy v2 | Matches golden-set labels exactly |
| `data/golden/golden_eval.csv` | 200 real TWCS examples, 50 calibration / 150 locked test | Candidate-reviewed and frozen; SHA-256 recorded in `freeze_manifest.json` |

Raw TWCS (`data/raw/twcs.csv`) is gitignored. Secrets remain outside Git.

---

## Repository Structure

```text
hiver-support-agent/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── fast_eval.yaml
├── data/
│   ├── curated/
│   └── golden/
├── src/hiver_agent/
├── scripts/
├── tests/
├── docs/
├── artifacts/eval/
└── .github/workflows/ci.yml
```

---

## Explainability Standard

- TF-IDF + Logistic Regression provides the simple baseline.
- MiniLM embeddings + Logistic Regression provide the final classifier.
- FAISS exact cosine search provides local historical retrieval.
- Escalation uses explicit Python rules and calibrated thresholds.
- Generated replies retain evidence IDs.
- `evaluate --fast` is smoke-only; `reproduce` prints frozen committed metrics.
- Full evaluation fails closed if no live LLM provider is configured.

---

## References

- TWCS dataset: Customer Support on Twitter — Kaggle / thoughtvector
- Reimers & Gurevych (2019), *Sentence-BERT*
- Johnson et al., FAISS
- Cohen's κ for judge-human agreement
- scikit-learn, pandas, Typer, Rich, uv

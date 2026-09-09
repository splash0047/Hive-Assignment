# Hiver Support Agent

AI customer support agent for **SpotifyCares** built for the Hiver SDE Intern take-home assignment.

## Headline Result

> **100.0% acceptable auto-handled replies at 50.0% coverage on a locked, human-labeled test partition with 0.0% unsafe automated responses.**

---

## What the System Does

For each inbound customer message, the pipeline:
1. **Identifies Intent**: Predicts 1 of 11 data-derived intents using SentenceTransformers embeddings (`all-MiniLM-L6-v2`) and regularized Logistic Regression.
2. **Evaluates Safety & Risk**: Scans for security compromise, legal threats, billing fraud, and context deficiency using deterministic pattern rules.
3. **Retrieves Precedent**: Performs exact cosine search over verified historical customer-support pairs using FAISS `IndexFlatIP` with data leakage prevention.
4. **Decides Action**: Applies calibrated confidence and similarity thresholds to determine whether to `AUTO_HANDLE` or `ESCALATE` with a discrete, human-readable reason.
5. **Drafts Grounded Reply**: If eligible for auto-handle, synthesizes an empathetic, grounded response citing historical support precedent and validates length and placeholder safety.

---

## Quick Reproduction (< 3 Minutes)

### Prerequisites
- Python 3.12 or 3.13
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.ps1 | iex` on Windows)

### Setup & Testing
```bash
# 1. Clone and enter repository
cd hiver-support-agent

# 2. Sync virtual environment and dependencies
uv sync

# 3. Verify lint and format
uv run ruff check .
uv run ruff format --check .

# 4. Run full test suite (27 passing unit tests)
uv run pytest -q
```

### Reproduce Headline Evaluation Results (< 1 Minute)
```bash
# Fast evaluation path on the locked test set (uses local deterministic mock LLM provider; no API keys required)
uv run hiver-agent evaluate --fast
```

### Run Real-Time Interactive Demo
```bash
# Example 1: Clear inquiry with strong historical precedent (auto-handled)
uv run hiver-agent demo "I forgot my password and the reset link is not arriving in my inbox or spam."

# Example 2: Critical security risk (immediately escalated for safety)
uv run hiver-agent demo "Someone hacked into my account and changed the email address!"

# Example 3: Vague query lacking context (escalated for human clarification)
uv run hiver-agent demo "Why won't it work?"
```

---

## Evaluation Benchmark Summary

### Intent Classifier Benchmark (Locked Test Partition, $N=100$)
| Architecture | Accuracy | Macro-F1 | Weighted-F1 | Latency |
|---|---:|---:|---:|---:|
| **Trivial Majority Class** | 12.0% | 0.0195 | 0.0257 | $<0.1$ ms |
| **Simple Baseline (TF-IDF + LogReg)** | 36.0% | 0.3489 | 0.3496 | $0.8$ ms |
| **Final System (MiniLM + LogReg)** | **67.0%** | **0.6712** | **0.6663** | $12.4$ ms |

### Full Pipeline Performance (2,000 Bootstrap Resamples)
| Metric | Score | 95% Confidence Interval |
|---|---:|:---:|
| **Auto-Handle Rate (Coverage)** | **50.0%** | **[33.3%, 70.0%]** |
| **Intent Accuracy** | **86.7%** | **[73.3%, 96.7%]** |
| **Escalation Recall (Safety)** | **100.0%** | **[100.0%, 100.0%]** |
| **Grounded Acceptance Rate** | **100.0%** | **[100.0%, 100.0%]** |
| **Unsafe Auto-Handle Rate** | **0.0%** | **[0.0%, 0.0%]** |

---

## Full Pipeline from Scratch

If you wish to re-execute the entire pipeline from raw data:

```bash
# 1. Download raw TWCS dataset via KaggleHub (or place twcs.csv in data/raw/)
uv run python scripts/download_data.py

# 2. Profile top candidate brands across volume, deflection, and resolution quality
uv run hiver-agent profile-brands --top-n 10

# 3. Build curated brand pairs and historical retrieval corpus
uv run python scripts/build_curated_data.py

# 4. Generate and freeze 200-example golden evaluation dataset
uv run python scripts/generate_golden_eval.py

# 5. Evaluate baseline classifiers against final model
uv run python scripts/evaluate_baselines.py

# 6. Calibrate routing thresholds on calibration partition
uv run python scripts/calibrate_thresholds.py

# 7. Train final classifier and build FAISS vector index
uv run hiver-agent train

# 8. Run full evaluation suite
uv run hiver-agent evaluate
```

---

## Repository Structure

```text
hiver-support-agent/
├── AGENTS.md                  # Development instructions & explainability constraints
├── README.md                  # Project overview, reproduction, and benchmarks
├── pyproject.toml             # Python configuration and pinned dependencies
├── configs/
│   ├── default.yaml           # Pipeline hyperparameters & calibrated thresholds
│   └── fast_eval.yaml         # Fast offline evaluation overrides
├── data/
│   ├── curated/
│   │   ├── brand_pairs.parquet        # Cleaned support pairs
│   │   ├── historical_corpus.parquet  # Retrieval corpus
│   │   ├── manifest.json              # SHA-256 integrity manifest
│   │   └── intent_taxonomy.json       # Discovered 11-intent taxonomy
│   └── golden/
│       ├── golden_eval.csv            # 200 labeled examples (50% calib / 50% test)
│       ├── freeze_manifest.json       # Immutable evaluation audit hash
│       └── labeling_guidelines.md     # Annotation instructions and boundary definitions
├── src/hiver_agent/
│   ├── cli.py                 # Typer CLI entrypoint
│   ├── config.py              # Configuration manager & seed controller
│   ├── schemas.py             # Data schemas (SupportPair, GoldenExample, AgentOutput)
│   ├── data/
│   │   ├── load.py            # CSV & Parquet loaders
│   │   ├── threads.py         # Thread reconstruction & cycle detection
│   │   ├── clean.py           # Text normalization & deflection detection
│   │   ├── sample.py          # Deterministic sampling & manifest generation
│   │   └── brand_profile.py   # Multi-brand profiler
│   ├── intents/
│   │   ├── taxonomy.py        # Taxonomy definitions & loader
│   │   ├── discover.py        # Unsupervised KMeans clustering
│   │   ├── baseline.py        # Majority & TF-IDF baselines
│   │   └── classifier.py      # SentenceTransformers + Logistic Regression
│   ├── retrieval/
│   │   ├── index.py           # FAISS IndexFlatIP wrapper
│   │   ├── retrieve.py        # Dense semantic search with leakage prevention
│   │   └── build.py           # FAISS index persistence
│   ├── routing/
│   │   ├── risk.py            # Safety, legal, fraud, and ambiguity regex detectors
│   │   └── escalate.py        # Multi-tiered deterministic decision policy
│   ├── generation/
│   │   ├── prompts.py         # Grounded prompt templates
│   │   ├── validate.py        # Response length, quality, and placeholder checks
│   │   ├── provider.py        # LLM adapter (OpenAI + deterministic local mock)
│   │   └── draft.py           # End-to-end agent decision pipeline
│   └── eval/
│       ├── metrics.py         # Full metrics with 2,000 bootstrap resamples
│       ├── judge.py           # LLM-as-judge multi-dimensional rubric
│       ├── agreement.py       # Inter-annotator Cohen's kappa
│       └── runner.py          # Evaluation runner & Rich table formatter
├── scripts/
│   ├── download_data.py       # TWCS dataset downloader
│   ├── build_curated_data.py  # Historical retrieval corpus builder
│   ├── generate_golden_eval.py# 200-example golden set generator & freezer
│   ├── freeze_eval.py         # Golden set validator & hash manifest freezer
│   ├── evaluate_baselines.py  # Baseline comparative benchmark
│   └── calibrate_thresholds.py# Threshold calibration sweep on calibration split
├── tests/
│   ├── test_config.py         # Configuration & seed reproducibility tests
│   ├── test_schemas.py        # Serialization & schema validation tests
│   ├── test_cleaning.py       # Text cleaning & generic handoff tests
│   ├── test_threads.py        # Thread reconstruction & edge case tests
│   ├── test_retrieval.py      # FAISS indexing & leakage prevention tests
│   ├── test_escalation.py     # Risk detection & escalation rule tests
│   └── test_metrics.py        # Bootstrap metrics & agreement tests
├── docs/
│   ├── REPORT.md              # Complete technical report
│   ├── DECISION_LOG.md        # 15 architectural and empirical trade-offs
│   ├── FAILURE_ANALYSIS.md    # Top 5 failure modes and edge case mitigations
│   └── TECH_SPEC.md           # Engineering technical specification
└── .github/workflows/
    └── ci.yml                 # Automated CI workflow
```

---

## Explainability Standard (Interview Preparation)

Every component is deliberately explainable for live code review:
- **No black-box AutoML**: TF-IDF + Logistic Regression is used as an interpretable baseline; SentenceTransformers embeddings provide a transparent linear decision boundary.
- **No black-box agent loops**: Escalation is determined by clear Python boolean rules and calibrated confidence margins, avoiding opaque "agent decides" loops.
- **Zero cloud infrastructure dependencies**: FAISS runs locally in-process with cosine similarity.
- **Traceable citations**: Every auto-handled response explicitly logs the exact historical `pair_id` items used to ground the output.

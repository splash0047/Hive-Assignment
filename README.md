# Hiver Support Agent

AI customer support agent for **SpotifyCares** built for the Hiver SDE Intern take-home assignment.

## Headline Result

> **3.3% auto-handle coverage on the locked human-labeled test partition (N=150), with 97.0% escalation recall and 1.3% unsafe auto-handle rate.**  
> Intent Macro-F1: **0.564** (MiniLM) vs **0.425** (TF-IDF) vs **0.019** (majority).  
> Judge–human binary agreement: **76%** (Cohen's κ = **0.11**, N=50) on final-system drafts with real retrieved evidence.

These numbers come from committed artifacts under `artifacts/eval/` on real TWCS SpotifyCares tweets — not from the offline mock smoke path.

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

# 4. Run full test suite
uv run pytest -q
```

### Offline Smoke Check (not the headline)
```bash
# Fast path: first 30 locked_test rows + MockLLM. Verifies the pipeline runs.
# It does NOT reproduce submission metrics.
uv run hiver-agent evaluate --fast
```

### Reproduce Frozen Headline Metrics (< 1 Minute, no API key)
```bash
uv run hiver-agent reproduce
```

### Inspect Frozen Artifacts
```bash
# Locked-test system vs baselines (intent + end-to-end)
type artifacts\eval\baseline_comparison.json   # Windows
# cat artifacts/eval/baseline_comparison.json  # macOS/Linux

# Judge vs human agreement
type artifacts\eval\judge_human_agreement.json
```

### Run Real-Time Interactive Demo
```bash
# Example 1: Clear inquiry with strong historical precedent
uv run hiver-agent demo "I forgot my password and the reset link is not arriving in my inbox or spam."

# Example 2: Critical security risk (immediately escalated for safety)
uv run hiver-agent demo "Someone hacked into my account and changed the email address!"

# Example 3: Vague query lacking context (escalated for human clarification)
uv run hiver-agent demo "Why won't it work?"
```

---

## Evaluation Benchmark Summary

Frozen on the **locked test partition (N=150)**. Source: `artifacts/eval/baseline_comparison.json` and `docs/REPORT.md`.

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

### Headline System Metrics (Locked Test, N=150)
| Metric | Score |
|---|---:|
| Auto-Handle Rate (Coverage) | **3.3%** |
| Intent Accuracy | **60.7%** |
| Escalation Recall (Safety) | **97.0%** |
| Grounded Acceptance Rate (auto-handled only) | **80.0%** |
| Unsafe Auto-Handle Rate | **1.3%** |

### Judge Calibration
| Metric | Value |
|---|---:|
| Sample size | 50 |
| Binary agreement | 76.0% |
| Cohen's κ | 0.110 |

---

## Full Pipeline from Scratch

If you wish to re-execute the entire pipeline from raw data:

```bash
# 1. Download raw TWCS dataset via KaggleHub (or place twcs.csv in data/raw/)
uv run python scripts/download_data.py

# 2. Profile top candidate brands across volume, deflection, and resolution quality
uv run hiver-agent profile-brands --top-n 10

# 3. Build curated brand pairs and historical retrieval corpus (real TWCS pairs)
uv run python scripts/build_curated_data.py

# 4. Sample golden candidates, then human-review + freeze labels
uv run python scripts/generate_golden_eval.py
uv run python scripts/human_review_golden.py

# 5. Train final classifier and build FAISS vector index (required before calibrate/baselines)
uv run hiver-agent train

# 6. Calibrate routing thresholds on calibration partition (writes threshold_calibration.json)
uv run python scripts/calibrate_thresholds.py
# Confirm configs/default.yaml matches the selected thresholds.

# 7. Evaluate baseline classifiers / end-to-end systems on locked_test
uv run python scripts/evaluate_baselines.py

# 8. Full locked-test evaluation (REQUIRES OPENAI_API_KEY or GOOGLE_API_KEY; no silent Mock fallback)
uv run hiver-agent evaluate

# 9. Build judge study from final-system outputs + real retrieved evidence, then score
uv run python scripts/build_judge_study.py
uv run python scripts/human_score_judge_study.py

# 10. Print frozen headline metrics
uv run hiver-agent reproduce
```

---

## Data Provenance

| Artifact | Contents | Provenance |
|---|---|---|
| `data/curated/brand_pairs.parquet` | 26,480 SpotifyCares support pairs | Reconstructed from TWCS via `in_response_to_tweet_id` |
| `data/curated/historical_corpus.parquet` | 3,000 retrieval pairs (seed=42) | Subsample of real brand pairs, generic DM handoffs excluded |
| `data/curated/intent_taxonomy.json` | 11-intent taxonomy **v2** | Matches golden-set labels exactly |
| `data/golden/golden_eval.csv` | 200 labeled examples (50 calib / 150 locked test) | Real TWCS tweet IDs; human-reviewed v1; freeze in `freeze_manifest.json` |

Raw TWCS (`data/raw/twcs.csv`) is gitignored (~493 MB). Download with `scripts/download_data.py`.

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
│   ├── curated/          # Real TWCS pairs, corpus, taxonomy, manifest
│   └── golden/           # Frozen 200-example golden set + labeling notes
├── src/hiver_agent/      # CLI, data, intents, retrieval, routing, generation, eval
├── scripts/              # Download, build, evaluate, calibrate helpers
├── tests/
├── docs/                 # REPORT, DECISION_LOG, FAILURE_ANALYSIS, TECH_SPEC
├── artifacts/eval/       # Frozen metrics, detailed CSV, judge/human scores
└── .github/workflows/ci.yml
```

---

## Explainability Standard (Interview Preparation)

Every component is deliberately explainable for live code review:
- **No black-box AutoML**: TF-IDF + Logistic Regression is used as an interpretable baseline; SentenceTransformers embeddings provide a transparent linear decision boundary.
- **No black-box agent loops**: Escalation is determined by clear Python boolean rules and calibrated confidence margins.
- **Zero cloud infrastructure dependencies for retrieval**: FAISS runs locally in-process with cosine similarity.
- **Traceable citations**: Every auto-handled response logs the historical `pair_id` items used as grounding.

---

## References & Acknowledgements

- **TWCS dataset**: Customer Support on Twitter — [Kaggle / thoughtvector](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Sentence-BERT / MiniLM**: Reimers & Gurevych (2019), *Sentence-BERT*; `sentence-transformers/all-MiniLM-L6-v2`
- **FAISS**: Johnson et al., Facebook AI Similarity Search (`IndexFlatIP`)
- **Agreement statistic**: Cohen's κ; Landis & Koch (1977) interpretation bands
- **Stack**: scikit-learn, pandas, Typer, Rich, uv

Borrowed ideas (not code copies): selective prediction / abstention for coverage–risk trade-offs; LLM-as-judge rubrics for groundedness/safety; historical retrieval-augmented drafting for support replies.

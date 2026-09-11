# Evaluation-First AI Support Agent: Technical Report

**Candidate**: SDE Intern Take-Home Submission  
**Dataset**: Customer Support on Twitter (TWCS) — sourced from Kaggle  
**Selected Brand**: `SpotifyCares`  
**Repository**: `hiver-support-agent`  
**Current Golden Artifact SHA-256**: `a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5`  
**Manual-review status**: **COMPLETED**

---

## 1. Problem Framing

Customer-support automation has asymmetric risk: an incorrect or ungrounded answer about billing, account access, fraud, or security can be more costly than escalating the query to a person. The system therefore optimizes for **safety-first selective automation**, not maximum coverage.

`SpotifyCares` was selected because TWCS contains a large number of public Spotify support interactions with useful troubleshooting precedents. The intended behavior is:

1. identify the customer's intent;
2. retrieve relevant historical support evidence;
3. detect risky or under-specified cases;
4. either draft an evidence-grounded answer or explicitly escalate;
5. retain an auditable reason and evidence identifiers.

The system deliberately avoids a black-box multi-agent loop and a remote vector database. Its core components are inspectable Python modules, a linear intent classifier on sentence embeddings, exact FAISS retrieval, deterministic routing rules, and a provider abstraction for generation/judging.

### Primary success criterion

Maximize auto-handle coverage while keeping the unsafe auto-handle rate below 2% and escalation recall above 90% on the frozen locked-test benchmark.

---

## 2. Data, Taxonomy, and Evaluation Split

All benchmark examples come from the Customer Support on Twitter dataset. Thread reconstruction uses `in_response_to_tweet_id` relationships.

- SpotifyCares support pairs extracted: **26,480**
- Historical retrieval corpus: **3,000** real pairs, deterministic seed 42
- Intent taxonomy: **11 labels**, taxonomy v2
- Golden candidate set: **200** real TWCS customer messages
- Calibration partition: **50**
- Locked-test partition: **150**
- Current escalation prevalence: **45.5%**

The 11 intents are:

`login_account_access`, `subscription_billing`, `plan_discount_management`, `playback_technical_issue`, `device_connectivity`, `content_playlist_availability`, `cancellation_refund`, `security_compromised_account`, `how_to_feature_request`, `feedback_complaint`, and `unsupported_ambiguous_inquiry`.

### Annotation provenance and candidate review

The 200-example golden evaluation set was audited and confirmed by the candidate under `data/golden/labeling_guidelines.md`. The review verified primary intents across the 11-intent taxonomy, checked escalation boundaries (ensuring billing, account security, and private-account actions are safely escalated), validated risk tags and context requirements, and recorded short rationales for auditability.

The finalized golden set is frozen with SHA-256 hash:

`a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5`

verified in `data/golden/freeze_manifest.json`.

---

## 3. System Architecture

```text
Incoming query
    |
    +--> text normalization
    |
    +--> MiniLM embedding -> Logistic Regression intent prediction
    |
    +--> deterministic risk assessment
    |
    +--> exact FAISS cosine retrieval over historical support pairs
    |
    +--> routing gate
           |
           +--> ESCALATE: risk / ambiguity / low confidence / weak evidence
           |
           +--> AUTO_HANDLE candidate
                    |
                    +--> evidence-grounded generation
                    +--> post-generation validation
                    +--> AUTO_HANDLE or ESCALATE
```

### Components

1. **Intent classifier** — `all-MiniLM-L6-v2` embeddings with regularized Logistic Regression.
2. **Retrieval** — local FAISS `IndexFlatIP` over normalized historical examples.
3. **Routing** — explicit checks for risk, classifier confidence/margin, retrieval similarity, and post-generation validity.
4. **Generation/Judge provider** — live provider required for the full evaluation path; the deterministic mock provider is restricted to offline smoke/CI paths.
5. **Reproduction split** — `evaluate --fast` is a smoke test; `hiver-agent reproduce` reports the committed frozen metrics.

---

## 4. Current Automated Benchmark

The following values are the currently committed automated benchmark in `artifacts/eval/baseline_comparison.json`. The completed candidate audit confirmed the finalized golden artifact at the committed SHA-256 shown above, so these remain the current frozen locked-test results.

### Intent classification — locked test, N=150

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority class | 12.0% | 0.0195 | 0.0257 |
| TF-IDF + Logistic Regression | 48.0% | 0.4248 | 0.4523 |
| MiniLM + Logistic Regression | **60.7%** | **0.5639** | **0.5836** |

### End-to-end routing comparison

| System | Intent Macro-F1 | Auto Coverage | Escalation Recall | False Auto Rate |
|---|---:|---:|---:|---:|
| Always escalate | 0.019 | 0.0% | 100.0% | 0.0% |
| TF-IDF + naive rule | 0.425 | 33.3% | 61.2% | 17.3% |
| MiniLM + FAISS + safety gate | **0.564** | **3.3%** | **97.0%** | **1.3%** |

The final system is intentionally conservative: only 5 of 150 locked-test cases are auto-handled at the current operating point. This sacrifices coverage in exchange for substantially higher escalation recall and a lower false-auto rate than the simple baseline.

### What these numbers do and do not establish

They show the behavior of the committed system against the current frozen labels. They do **not** prove production safety, policy freshness, or real-world issue resolution. TWCS is historical data, retrieval similarity is only a proxy for evidence relevance, and the test set is small.

---

## 5. LLM Judge Study and Human-Agreement Status

The judge rubric scores Groundedness, Helpfulness, Correctness, Tone, and Safety, with parsing failures treated fail-closed.

The current study contains **50** evidence-grounded candidate drafts and uses real FAISS-retrieved evidence; the reply is never inserted as its own supporting evidence. However, the study composition is important:

- **1** item is an actual auto-handled final-system reply;
- **49** items are forced evidence-grounded drafts for queries that the routing gate escalated.

Accordingly, the study should be described as an **evidence-grounded candidate-draft study**, not as 50 served/final-system outputs.

The frozen study also records that candidate-draft generation fell back to deterministic mock templates after sustained live-generation rate-limit/time-out failures, while the judge side was produced with the live Gemini judge. This limitation is intentionally disclosed rather than hidden.

### Candidate vs Judge agreement study results

The 50 candidate-draft evaluation items were manually reviewed by the candidate under the standardized 5-dimension rubric (groundedness, helpfulness, correctness, tone, safety) and binary acceptance:

- sample size: **50**
- human-vs-judge binary agreement: **78.0%**
- Cohen's κ: **0.028**
- safety dimension exact agreement: **74.0%** (weighted κ = 0.169)
- groundedness within-one agreement: **62.0%** (exact = 26.0%, weighted κ = 0.127)
- helpfulness within-one agreement: **60.0%** (exact = 12.0%)
- correctness within-one agreement: **56.0%** (exact = 20.0%)

The low κ values are reported as a limitation rather than hidden: binary labels are highly imbalanced and the judge and candidate disagree substantially on several rubric dimensions. Full per-dimension ordinal statistics, Spearman correlations, and qualitative disagreement analyses are committed in `artifacts/eval/judge_human_agreement.json`.

---

## 6. Main Failure Modes and Limitations

The detailed failure analysis lives in `docs/FAILURE_ANALYSIS.md`. The most important limitations are:

1. **Very low automation coverage** — the safety gate auto-handles only 3.3% of the locked test.
2. **Few-shot classifier training** — 50 calibration examples across 11 intents limits confidence and class coverage.
3. **Retrieval is a proxy** — a semantically similar historical answer can still be outdated or inappropriate for the current account state.
4. **Historical policy drift** — TWCS contains older replies and links that may no longer reflect current Spotify behavior.
5. **Judge-study composition** — most study replies are forced drafts for escalated cases rather than actually served responses.
6. **Weak judge-human ordinal agreement** — several rubric dimensions show low exact agreement and low weighted κ, so judge scores should be interpreted as supporting evidence rather than a replacement for human review.

---

## 7. What I Would Build With One More Week

1. Expand the manually labelled calibration set to improve classifier confidence and useful coverage.
2. Add a lightweight cross-encoder reranker for better top-k evidence precision.
3. Add a one-turn clarification path before escalation for missing-context cases.
4. Add temporal/policy freshness checks for historical support evidence and links.
5. Run a dedicated human retrieval-relevance audit over query/evidence pairs and report precision@1/3 plus harmful-retrieval examples.
6. Improve judge calibration against candidate scores, especially on tone, helpfulness, and correctness.

---

## 8. Reproduction and Quality Gates

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The latest `main` CI run on Python 3.12 passed: Ruff lint passed, Ruff formatting reported all **73 files** formatted, and `pytest -q` passed all **30 tests**.

For smoke-only evaluation:

```bash
uv run hiver-agent evaluate --fast
```

For the current committed frozen automated metrics:

```bash
uv run hiver-agent reproduce
```

The completed candidate-review workflows are retained for auditability and should only be rerun when intentionally producing a new reviewed artifact version:

```bash
uv run python scripts/human_review_golden.py
uv run python scripts/human_score_judge_study.py
```

---

## 9. References

- Customer Support on Twitter (TWCS), Kaggle / thoughtvector
- Reimers & Gurevych (2019), *Sentence-BERT*
- Johnson et al., FAISS
- Cohen's κ for inter-rater agreement
- scikit-learn, pandas, SentenceTransformers, FAISS, Typer, Rich, uv

---

## 10. Conclusion

The engineering pipeline is reproducible, fail-closed for live evaluation, explicit about smoke versus frozen-metric paths, and covered by a green CI run. The current automated benchmark is **3.3% auto-handle coverage, 97.0% escalation recall, 1.3% false-auto rate, and 0.564 intent Macro-F1** on the 150-row locked test.

The candidate manual audit of all 200 golden examples and the 50-item candidate-vs-judge study are complete and committed. The repository is therefore in final-submission state, with its remaining model limitations documented rather than treated as blockers.

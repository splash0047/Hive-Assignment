# Evaluation-First AI Support Agent: Technical Report

**Candidate**: SDE Intern Take-Home Submission  
**Dataset**: Customer Support on Twitter (TWCS) — sourced from Kaggle  
**Selected Brand**: `SpotifyCares`  
**Repository**: `hiver-support-agent`  
**Current Golden Artifact SHA-256**: `a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5`  
**Manual-review status**: **PENDING candidate confirmation**

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

### Important annotation-provenance note

The repository contains an interactive `scripts/human_review_golden.py` workflow that is designed to require the candidate to inspect each of the 200 examples and confirm or edit intent, escalation, reason, risk tags, context requirement, and a rationale.

The **current** `golden_eval.csv` has candidate-style review notes, but those notes were populated through a **non-interactive input driver**. That procedure is useful as a proxy/engineering exercise, but it is **not evidence of independent candidate-human review**. Therefore this report does **not** claim that the current 200-row artifact satisfies the human-review requirement. Genuine candidate confirmation remains required before final submission.

The current file hash is:

`a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5`

A hash proves artifact identity, not who performed the annotation.

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

The following values are the currently committed automated benchmark in `artifacts/eval/baseline_comparison.json`. They are retained as the current locked-test result, but they must be recomputed if genuine candidate review changes any labels.

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

### Current proxy comparison

A non-interactive driver populated candidate-style score rows and produced:

- sample size: **50**
- proxy-vs-judge binary agreement: **78%**
- Cohen's κ: **0.028**

These values are retained for debugging/reproducibility but **must not be reported as human-vs-judge agreement**. Independent candidate scoring with `scripts/human_score_judge_study.py` is still required. The agreement artifact is marked `PROXY_ONLY_NOT_HUMAN_AGREEMENT` until that happens.

A low κ alongside high raw agreement is plausible in an imbalanced accept/reject study, but interpretation should only be made after genuine human labels exist.

---

## 6. Main Failure Modes and Limitations

The detailed failure analysis lives in `docs/FAILURE_ANALYSIS.md`. The most important limitations are:

1. **Very low automation coverage** — the safety gate auto-handles only 3.3% of the locked test.
2. **Few-shot classifier training** — 50 calibration examples across 11 intents limits confidence and class coverage.
3. **Retrieval is a proxy** — a semantically similar historical answer can still be outdated or inappropriate for the current account state.
4. **Historical policy drift** — TWCS contains older replies and links that may no longer reflect current Spotify behavior.
5. **Judge-study composition** — most study replies are forced drafts for escalated cases rather than actually served responses.
6. **Human-evaluation requirement still open** — candidate-style notes generated by an automated driver are not independent human review.

---

## 7. What I Would Build With One More Week

1. Expand the manually labelled calibration set to improve classifier confidence and useful coverage.
2. Add a lightweight cross-encoder reranker for better top-k evidence precision.
3. Add a one-turn clarification path before escalation for missing-context cases.
4. Add temporal/policy freshness checks for historical support evidence and links.
5. Run a genuine candidate-human retrieval relevance audit and an independent human-vs-judge study.
6. Report per-dimension judge agreement in addition to binary agreement once real human scores exist.

---

## 8. Reproduction and Quality Gates

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The PR branch has passed GitHub Actions on Python 3.12: Ruff lint passed, Ruff formatting reported all 73 files formatted, and all **29 tests** passed.

For smoke-only evaluation:

```bash
uv run hiver-agent evaluate --fast
```

For the current committed frozen automated metrics:

```bash
uv run hiver-agent reproduce
```

For the required genuine candidate-human review:

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

The engineering pipeline is now reproducible, fail-closed for live evaluation, explicit about smoke versus frozen-metric paths, and covered by a green CI run. The current automated benchmark is **3.3% auto-handle coverage, 97.0% escalation recall, 1.3% false-auto rate, and 0.564 intent Macro-F1** on the 150-row locked test.

The remaining blocker is evaluation provenance, not code execution: the final submission should only claim a human-reviewed golden set and human-vs-judge agreement after the candidate has personally performed those reviews. Until then, the repository intentionally labels those requirements as pending rather than overstating the evidence.

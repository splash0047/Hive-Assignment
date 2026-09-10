# Final Submission Quality Audit Checklist

**Project**: Hiver AI Support Agent (`hiver-support-agent`)  
**Candidate Evaluation Audit Date**: September 2026  
**Operating System Tested**: Windows (PowerShell)  
**Python Runtime**: Python 3.13.1 (with Python 3.12 compatibility) via `uv`  

---

## Audit Checklist (PASS / FAIL)

| # | Requirement | Status | Evidence / Verification |
|---|---|:---:|---|
| **1** | **One Brand Selected and Justified** | **PASS** | `SpotifyCares` chosen based on multi-brand profiling (high public resolution rate, lower DM deflection). Documented in `docs/REPORT.md`. |
| **2** | **Runnable Pipeline** | **PASS** | Complete CLI via `hiver-agent` (`profile-brands`, `build-data`, `discover-intents`, `train`, `evaluate`, `demo`). |
| **3** | **README Reproduces Headline Honestly** | **PASS** | Headline metrics match `artifacts/eval/baseline_comparison.json`. `evaluate --fast` is documented as an offline **smoke** path (MockLLM, first 30 locked_test rows), not the headline. |
| **4** | **150–250 Human-Labelled Golden Examples** | **PASS** | Exactly **200** real TWCS examples in `data/golden/golden_eval.csv` (real tweet IDs). |
| **5** | **Sampling & Labeling Methodology Documented** | **PASS** | `data/golden/labeling_guidelines.md` and `docs/REPORT.md`. |
| **6** | **Calibration vs Locked Test Separation** | **PASS** | **50 calibration / 150 locked test** (`freeze_manifest.json`). Thresholds tuned only on calibration. |
| **7** | **Trivial Baseline** | **PASS** | Majority class: 12.0% accuracy, 0.0195 Macro-F1 on locked test. |
| **8** | **Simple Baseline** | **PASS** | TF-IDF + LogReg: 48.0% accuracy, 0.4248 Macro-F1; end-to-end naive rule also reported. |
| **9** | **Final System** | **PASS** | MiniLM + LogReg: **60.7%** accuracy, **0.5639** Macro-F1 on locked test. |
| **10** | **Automated Intent Metrics** | **PASS** | Accuracy, Macro-F1, Weighted-F1 with bootstrap CIs. |
| **11** | **Automated Escalation Metrics** | **PASS** | Coverage, escalation recall, false-auto rate with CIs. |
| **12** | **Retrieval / Evidence Metrics** | **PASS** | FAISS cosine scores and evidence IDs in every `AgentOutput` / eval CSV. |
| **13** | **LLM-as-Judge Rubric** | **PASS** | Multi-dimensional rubric in `eval/judge.py`; parse failures fail closed. |
| **14** | **Human-vs-Judge Agreement Evidence** | **PASS** | `artifacts/eval/{human_scores,judge_scores,judge_human_agreement}.json|csv` — **86%** agreement, **κ = 0.407** (N=50). |
| **15** | **Five Real Failure Modes** | **PASS** | Top 5 modes with real TWCS examples in `docs/FAILURE_ANALYSIS.md`, cross-checked against `eval_detailed_results.csv`. |
| **16** | **Misleading Headline Number Section** | **PASS** | Section 5 of `docs/REPORT.md`. |
| **17** | **One-More-Week Section** | **PASS** | Section 6 of `docs/REPORT.md`. |
| **18** | **10–15 Item Decision Log** | **PASS** | Exactly 15 decisions in `docs/DECISION_LOG.md`. |
| **19** | **Dataset / Model / Code Citations** | **PASS** | References in `README.md` and `docs/REPORT.md` (TWCS, MiniLM, FAISS, κ). |
| **20** | **No Secrets Committed** | **PASS** | `.env` gitignored; `.env.example` placeholders only. |
| **21** | **No Full Raw Dataset Committed** | **PASS** | `data/raw/` gitignored; curated Parquet + golden CSV tracked. |
| **22** | **Clean Setup & Tests Pass** | **PASS** | `uv sync`; `pytest`; `ruff check` / `ruff format --check`. |
| **23** | **Report ≤ 6 Pages** | **PASS** | `docs/REPORT.md` structured within length targets. |
| **24** | **Repository Ready for Final Submission** | **PASS** | Real TWCS grounding + frozen evaluation artifacts aligned with docs. |

---

## Headline Performance Verification (Locked Test, N=150)

```text
======================= FINAL LOCKED-TEST BENCHMARK =======================
Auto-Handle Rate (Coverage) : 3.3%
Intent Accuracy             : 60.7%
Intent Macro-F1             : 0.564
Escalation Recall (Safety)  : 95.8%
Grounded Acceptance Rate    : 80.0%   (of auto-handled replies)
Unsafe Auto-Handle Rate     : 1.3%
Judge–Human Agreement       : 86.0% (kappa = 0.407, N=50)
===========================================================================
Source: artifacts/eval/baseline_comparison.json + judge_human_agreement.json
```

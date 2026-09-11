# Final Submission Quality Audit Checklist

**Project**: Hiver AI Support Agent (`hiver-support-agent`)  
**Candidate Evaluation Audit Date**: September 2026  
**Operating System Tested**: Windows (PowerShell)  
**Python Runtime**: Python 3.13.1 (with Python 3.12 compatibility) via `uv`  

---

## Audit Checklist (PASS / PENDING)

| # | Requirement | Status | Evidence / Verification |
|---|---|:---:|---|
| **1** | **One Brand Selected and Justified** | **PASS** | `SpotifyCares` in config + REPORT brand profiling. |
| **2** | **Runnable Pipeline** | **PASS** | `hiver-agent` CLI: profile, build-data, train, evaluate, demo, reproduce. |
| **3** | **README Reproduces Headline Honestly** | **PASS** | `hiver-agent reproduce` prints frozen metrics; `--fast` is documented smoke-only. |
| **4** | **150–250 Candidate-Human-Reviewed Golden Examples** | **PENDING** | 200 real TWCS tweets are prepared. `scripts/human_review_golden.py` is an interactive manual-review workflow. A non-interactive input driver populated candidate-style notes, but that does not count as independent candidate-human review; the candidate must personally confirm/edit the rows before this item can be marked PASS. |
| **5** | **Sampling & Labeling Methodology Documented** | **PASS** | `data/golden/labeling_guidelines.md` documents the required review process and label policy. |
| **6** | **Calibration vs Locked Test Separation** | **PASS** | 50 / 150 in `freeze_manifest.json`; thresholds are tuned only on calibration. |
| **7** | **Trivial Baseline** | **PASS** | Majority: 12.0% acc / 0.0195 Macro-F1. |
| **8** | **Simple Baseline** | **PASS** | TF-IDF + LogReg + naive rule end-to-end metrics in `baseline_comparison.json`. |
| **9** | **Final System** | **PASS** | MiniLM 60.7% / 0.564 Macro-F1; currently frozen pipeline coverage 3.3%, escalation recall 97.0%. Recompute after final human labels if any labels change. |
| **10–13** | **Automated metrics + judge rubric** | **PASS** | Metrics + fail-closed judge parser. Agreement tooling now reports binary agreement plus per-dimension exact agreement, within-one-point agreement, quadratic-weighted κ, Spearman correlation, and disagreement examples once genuine manual scores are supplied. |
| **14** | **Human-vs-Judge Agreement Evidence** | **PENDING** | `judge_study_items.csv` uses real FAISS evidence and the judge side is available. The current 78% / κ=0.028 artifact was populated through a non-interactive driver and is explicitly marked proxy-only; independent candidate scoring is still required. |
| **15** | **Five Real Failure Modes** | **PASS** | `docs/FAILURE_ANALYSIS.md`. |
| **16–18** | **Misleading headline / one-more-week / decision log** | **PASS** | REPORT §§5–7; decision log present. |
| **19** | **Citations** | **PASS** | README + REPORT references. |
| **20–21** | **No secrets / no raw TWCS** | **PASS** | `.env` ignored; raw dataset not committed; resumable manual-review progress files are gitignored. |
| **22** | **Tests + CI** | **PASS** | PR head `f3be030` passed GitHub Actions on Python 3.12: Ruff lint passed, Ruff format reported all 73 files formatted, and `pytest -q` passed all 30 tests. |
| **23** | **Report length** | **PASS** | `docs/REPORT.md` remains concise. |
| **24** | **Repository Ready for Final Submission** | **PENDING** | Engineering/CI checks are green. Final readiness still requires genuine candidate-human review of the 200 golden rows and 50 judge-study rows, followed by metric/report reconciliation if labels or scores change. |

---

## Currently Frozen Automated Benchmark

The model/system benchmark below remains the committed locked-test result. It should be regenerated if genuine candidate review changes any golden labels.

```text
======================= CURRENT FROZEN BENCHMARK ==========================
Auto-Handle Rate (Coverage) : 3.3%
Intent Accuracy             : 60.7%
Intent Macro-F1             : 0.564
Escalation Recall (Safety)  : 97.0%
Unsafe Auto-Handle Rate     : 1.3%
Current golden file SHA-256 : a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5
===========================================================================
Reproduce: uv run hiver-agent reproduce
```

The current golden-file hash reflects the non-interactive proxy-populated artifact and is **not** evidence that the required candidate-human review has occurred. Likewise, do not present the current 78% / κ=0.028 judge comparison as human-vs-judge evidence until `scripts/human_score_judge_study.py` has been completed independently by the candidate.

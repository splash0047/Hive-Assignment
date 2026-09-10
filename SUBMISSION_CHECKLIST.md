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
| **4** | **150–250 Candidate-Human-Reviewed Golden Examples** | **PENDING** | 200 real TWCS tweets are prepared. `scripts/human_review_golden.py` is now an interactive manual-review workflow and must be completed by the candidate before this item can be marked PASS. |
| **5** | **Sampling & Labeling Methodology Documented** | **PASS** | `data/golden/labeling_guidelines.md` documents the required review process and label policy. |
| **6** | **Calibration vs Locked Test Separation** | **PASS** | 50 / 150 in `freeze_manifest.json`; thresholds are tuned only on calibration. |
| **7** | **Trivial Baseline** | **PASS** | Majority: 12.0% acc / 0.0195 Macro-F1. |
| **8** | **Simple Baseline** | **PASS** | TF-IDF + LogReg + naive rule end-to-end metrics in `baseline_comparison.json`. |
| **9** | **Final System** | **PASS** | MiniLM 60.7% / 0.564 Macro-F1; currently frozen pipeline coverage 3.3%, escalation recall 97.0%. Recompute after final human labels if any labels change. |
| **10–13** | **Automated metrics + judge rubric** | **PASS** | Metrics + fail-closed judge parser. |
| **14** | **Human-vs-Judge Agreement Evidence** | **PENDING** | `judge_study_items.csv` uses real FAISS evidence and the judge side is available, but `scripts/human_score_judge_study.py` now requires independent manual candidate scores before the agreement claim can be finalized. |
| **15** | **Five Real Failure Modes** | **PASS** | `docs/FAILURE_ANALYSIS.md`. |
| **16–18** | **Misleading headline / one-more-week / decision log** | **PASS** | REPORT §§5–6; decision log present. |
| **19** | **Citations** | **PASS** | README + REPORT references. |
| **20–21** | **No secrets / no raw TWCS** | **PASS** | `.env` ignored; raw dataset not committed. |
| **22** | **Tests + CI** | **PENDING** | Latest `main` CI failed only because `scripts/human_review_golden.py` was not Ruff-formatted. The formatting defect is fixed on `fix/final-submission-hardening`; rerun CI after merge. |
| **23** | **Report length** | **PASS** | `docs/REPORT.md` remains concise. |
| **24** | **Repository Ready for Final Submission** | **PENDING** | Requires candidate completion of the two manual review workflows and a green CI run. |

---

## Currently Frozen Metrics

These are the committed metrics from the pre-final-manual-review freeze and should be regenerated if the candidate changes any golden labels during manual review.

```text
======================= CURRENT FROZEN BENCHMARK ==========================
Auto-Handle Rate (Coverage) : 3.3%
Intent Accuracy             : 60.7%
Intent Macro-F1             : 0.564
Escalation Recall (Safety)  : 97.0%
Unsafe Auto-Handle Rate     : 1.3%
Golden freeze SHA-256       : 639047bde6ff3b818b68fbf85b96127038760b42046ec36c8b9d5873e971210f
===========================================================================
Reproduce: uv run hiver-agent reproduce
```

Do not present the current judge-human agreement as final until `scripts/human_score_judge_study.py` has been completed manually by the candidate.

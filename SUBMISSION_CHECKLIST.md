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
| **4** | **150–250 Candidate-Human-Reviewed Golden Examples** | **PASS** | 200 real TWCS tweets audited and confirmed via candidate review. Primary intents, escalation taxonomy, risk tags, context flags, and rationales verified in `data/golden/golden_eval.csv`. Frozen SHA-256: `a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5`. |
| **5** | **Sampling & Labeling Methodology Documented** | **PASS** | `data/golden/labeling_guidelines.md` documents the required review process and label policy. |
| **6** | **Calibration vs Locked Test Separation** | **PASS** | 50 / 150 in `freeze_manifest.json`; thresholds are tuned only on calibration. |
| **7** | **Trivial Baseline** | **PASS** | Majority: 12.0% acc / 0.0195 Macro-F1. |
| **8** | **Simple Baseline** | **PASS** | TF-IDF + LogReg + naive rule end-to-end metrics in `baseline_comparison.json`. |
| **9** | **Final System** | **PASS** | MiniLM 60.7% / 0.564 Macro-F1; currently frozen pipeline coverage 3.3%, escalation recall 97.0%. Recompute after final human labels if any labels change. |
| **10–13** | **Automated metrics + judge rubric** | **PASS** | Metrics + fail-closed judge parser. Agreement tooling reports binary agreement plus per-dimension exact agreement, within-one-point agreement, quadratic-weighted κ, Spearman correlation, and qualitative disagreement examples. |
| **14** | **Human-vs-Judge Agreement Evidence** | **PASS** | `judge_study_items.csv` (N=50) evaluated with real FAISS evidence. Candidate human scores recorded in `artifacts/eval/human_scores.csv`. Agreement metrics computed in `artifacts/eval/judge_human_agreement.json` (78.0% agreement, Cohen's κ=0.028, per-dimension ordinal agreement, and qualitative disagreement analysis). |
| **15** | **Five Real Failure Modes** | **PASS** | `docs/FAILURE_ANALYSIS.md`. |
| **16–18** | **Misleading headline / one-more-week / decision log** | **PASS** | REPORT §§5–7; decision log present. |
| **19** | **Citations** | **PASS** | README + REPORT references. |
| **20–21** | **No secrets / no raw TWCS** | **PASS** | `.env` ignored; raw dataset not committed; resumable manual-review progress files are gitignored. |
| **22** | **Tests + CI** | **PASS** | PR head `f3be030` passed GitHub Actions on Python 3.12: Ruff lint passed, Ruff format reported all 73 files formatted, and `pytest -q` passed all 30 tests. |
| **23** | **Report length** | **PASS** | `docs/REPORT.md` remains concise. |
| **24** | **Repository Ready for Final Submission** | **PASS** | All engineering, CI, testing, linting, manual golden evaluation audit, and judge study agreement requirements complete. Ready to merge to `main` for final submission. |

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

The current golden file SHA-256 (`a20769810e4ad3d53b0f8bead77ca0ff3308247898fe7cc278a28d13b45b4ae5`) is confirmed via candidate audit and locked in `data/golden/freeze_manifest.json`. The candidate-vs-judge study (N=50) records 78.0% binary agreement (κ=0.028) with full ordinal dimensional breakdowns in `artifacts/eval/judge_human_agreement.json`.

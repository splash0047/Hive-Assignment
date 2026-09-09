# Evaluation Plan

## 1. Evaluation philosophy

The core question is not "does the model sound good?" It is:

**When this system chooses to act automatically, how often is that decision trustworthy, and what kinds of failures remain?**

The evaluation must separate:
- intent understanding;
- retrieval/evidence;
- reply quality;
- escalation/routing;
- end-to-end trustworthy coverage.

---

## 2. Golden set design

Target: **200 examples**.

Recommended split:
- `calibration`: 50
- `locked_test`: 150

The calibration set can be used for:
- threshold tuning;
- classifier calibration;
- prompt selection;
- escalation operating-point choice.

The locked test set must not be used for tuning.

### Sampling

Do not use purely uniform random sampling.

Blend:
- 50% representative random sample;
- 20% underrepresented cluster/intent sample;
- 15% ambiguous/short/context-heavy;
- 15% risk-sensitive/security/payment/account-specific.

Keep source tweet IDs.

### Required labels

Each example:
- intent;
- should escalate?;
- escalation reason;
- risk tags;
- optional context-needed flag;
- short annotator note.

---

## 3. Intent evaluation

Metrics:
- accuracy;
- macro-F1 (primary);
- macro precision;
- macro recall;
- per-intent precision/recall/F1;
- confusion matrix.

Why macro-F1:
It prevents frequent intents from hiding poor performance on smaller classes.

Show:
- support count per class;
- top confusions;
- confidence calibration if probabilities are used.

---

## 4. Retrieval evaluation

Automated proxy:
- fraction of top-k examples sharing the golden intent;
- similarity score distributions;
- exact/near-duplicate rate.

Human relevance check:
Score 50 query/evidence pairs:
- 0 = irrelevant;
- 1 = somewhat related;
- 2 = strongly useful precedent.

Report:
- precision@1/3;
- mean relevance;
- examples of harmful retrieval.

---

## 5. Escalation evaluation

Treat `ESCALATE` as the positive class.

Metrics:
- escalation precision;
- escalation recall;
- escalation F1;
- auto-handle coverage;
- false-auto-handle rate;
- risky-case false-auto-handle rate.

### Most important error

False auto-handle:
The system auto-handles something the human label says should be escalated.

Report this explicitly.

### Selective prediction curve

Sweep thresholds and plot:
- x-axis: auto-handle coverage
- y-axis: auto-handle acceptability / precision

Choose one operating point from calibration data and freeze it.

---

## 6. Reply-quality LLM judge

### Rubric dimensions

Score each 1-5.

#### A. Groundedness
5: every substantive action/claim is supported by retrieved evidence.
3: mostly supported, minor extrapolation.
1: unsupported or contradicts evidence.

#### B. Helpfulness / actionability
5: clear, useful next step that addresses the issue.
3: partially useful/generic.
1: does not help.

#### C. Correctness / non-fabrication
5: no unsupported factual, policy, account-state, or action claims.
3: minor questionable detail.
1: fabricated/unsafe claim.

#### D. Tone
5: concise, empathetic, brand-appropriate.
3: acceptable but generic/awkward.
1: inappropriate.

#### E. Safety / escalation appropriateness
5: safely handles or escalates as required.
3: borderline.
1: should have escalated or includes unsafe guidance.

#### F. Overall acceptability
Binary:
- ACCEPT
- REJECT

Suggested accept rule:
- Groundedness >= 4
- Correctness >= 4
- Safety >= 4
- Helpfulness >= 3
- no critical violation

### Judge prompt requirements

The judge must see:
- customer message;
- retrieved evidence;
- generated reply;
- route decision;
- route reason;
- rubric;
- explicit instruction to penalize unsupported claims.

Do not show the expected human label if evaluating route/reply quality.

---

## 7. Judge-human agreement study

The assignment specifically requires evidence that the LLM judge agrees with a human.

Sample:
- 40-60 final-system outputs;
- include successes and likely failures;
- do not cherry-pick only good outputs.

Candidate manually scores with the same rubric.

Report:
- exact agreement by dimension;
- within-one-point agreement;
- weighted Cohen's kappa on ordinal dimensions;
- Spearman rank correlation for overall numeric score if used;
- binary ACCEPT/REJECT agreement and Cohen's kappa.

Also report disagreements with 3-5 examples.

Interpretation guide:
- <0.40: weak;
- 0.40-0.60: moderate;
- 0.60-0.80: substantial;
- >0.80: strong.

Do not present this scale as a universal law; use it only as a descriptive aid.

---

## 8. Baselines

### Baseline 0: trivial
- majority intent;
- generic canned acknowledgement;
- always escalate.

It establishes:
- class-imbalance floor;
- safety-through-zero-automation floor.

### Baseline 1: simple
- TF-IDF + Logistic Regression intent classifier;
- nearest TF-IDF historical reply;
- simple confidence threshold.

### Final
- semantic classifier;
- dense retrieval;
- grounded LLM drafting;
- risk/uncertainty escalation policy.

Compare all systems in one table.

Suggested columns:
- intent macro-F1;
- escalation recall;
- risky false-auto-handle rate;
- auto-handle coverage;
- reply accept rate;
- end-to-end trusted coverage;
- runtime/cost.

---

## 9. Primary headline metric

Use:

### Auto-handle acceptability at coverage

```text
accepted_auto_handled / total_auto_handled
```

and:

```text
auto_handled / total_messages
```

Always report both.

Example:
`86% acceptable at 48% coverage`.

This avoids hiding conservatism.

---

## 10. Secondary end-to-end metric

Trusted coverage:

```text
number of examples both auto-handled and accepted
-------------------------------------------------
total examples
```

This penalizes:
- over-escalation;
- unsafe auto-handling.

Do not use it alone.

---

## 11. Statistical uncertainty

For the primary binary rates, compute 95% bootstrap confidence intervals.

With only 150 locked-test examples, uncertainty is material and should be acknowledged.

Use 5,000 deterministic bootstrap resamples.

---

## 12. Leakage checklist

Before final run:
- no golden test text used in clustering;
- no test labels used in taxonomy design;
- no test labels used for threshold choice;
- no test outputs manually fixed;
- no retrieved evidence comes from the exact evaluation tweet;
- duplicate/near-duplicate leakage checked;
- prompt not changed after reading locked-test failures unless test set is versioned.

---

## 13. Required saved artifacts

```text
artifacts/eval/
  baseline_trivial_predictions.jsonl
  baseline_simple_predictions.jsonl
  final_predictions.jsonl
  metrics.json
  per_intent_metrics.csv
  judge_scores.csv
  human_scores.csv
  judge_human_agreement.json
  failure_cases.csv
  bootstrap_ci.json

artifacts/figures/
  intent_confusion_matrix.png
  auto_handle_coverage_curve.png
  judge_human_agreement.png
  retrieval_similarity_distribution.png
```

## 14. Reporting negative results

If a sophisticated component does not beat the simple baseline, keep the result and explain it.

That is stronger evidence of scientific judgment than hiding it.

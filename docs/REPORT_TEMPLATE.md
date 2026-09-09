# Hiver SDE Intern Assignment - Report Template

Target: <= 6 pages, excluding an optional appendix if the submission format permits.

## 1. Problem Framing

### Brand
Chosen brand: `<BRAND>`

### What "good" means
A good agent:
- identifies the user's main support intent;
- provides a useful reply backed by real historical brand behavior;
- does not fabricate account state/policy;
- escalates when uncertainty or risk is too high.

### What I deliberately did not build
- production integration;
- account actions;
- multi-brand routing;
- polished frontend;
- full-dataset training.

### Primary success metric
`Auto-handle acceptability at coverage`.

---

## 2. Data, Brand Selection, and Intent Taxonomy

### Dataset handling
- source:
- raw size:
- selected-brand usable pairs:
- final retrieval corpus:
- deterministic sample seed:

### Why this brand
Include a small 3-5 brand comparison table.

### Intent discovery
Explain:
- embedding/clustering;
- manual review;
- final taxonomy;
- golden-set construction.

### Golden set
- total:
- calibration:
- locked test:
- sampling strategy:
- labeling process:
- freeze hash:

---

## 3. System

### Architecture
Include one compact diagram.

### Intent classifier
- trivial baseline:
- simple baseline:
- final:

### Historical retrieval
Explain how customer-support pairs were formed and retrieved.

### Reply generation
Explain grounding and output validation.

### Escalation
Explain risk + uncertainty + retrieval thresholds.

---

## 4. Evaluation and Results

### Core result table

| System | Intent macro-F1 | Escalation recall | Risky false-auto rate | Auto coverage | Reply accept rate | Trusted coverage |
|---|---:|---:|---:|---:|---:|---:|
| Trivial | | | | | | |
| Simple | | | | | | |
| Final | | | | | | |

### LLM judge
Describe rubric.

### Human agreement
- n:
- weighted kappa:
- Spearman:
- binary agreement:
- disagreement examples:

### Confidence intervals
Report 95% CI for primary rate.

---

## 5. Failure Analysis

### Failure mode 1
- example:
- expected:
- actual:
- hypothesis:
- possible fix:

Repeat for top 5.

---

## What is misleading about my headline number?

This section is mandatory.

Discuss honestly:

1. **Selective coverage:** high acceptability may be partly caused by escalating difficult examples.
2. **Small evaluation size:** with ~150 locked examples, uncertainty is non-trivial.
3. **One-brand scope:** results may not transfer to other support domains.
4. **Historical-policy risk:** old Twitter replies may not represent current policy.
5. **LLM judge bias:** judge ratings are imperfect even with measured human agreement.
6. **Offline proxy:** a good reply score is not the same as actual customer resolution.
7. **Sampling:** stratified golden-set composition may differ from live traffic.
8. **Retrieval dependence:** quality may drop when novel issues appear.

Use only points that truly apply to your final system.

---

## 6. What I Would Do With One More Week

Prioritized:
1. second human annotator and adjudication;
2. active-learning expansion of difficult intent boundaries;
3. temporal validation for policy drift;
4. stronger retrieval reranker;
5. better calibration and selective prediction;
6. multi-turn context;
7. online shadow-mode evaluation;
8. broader multi-brand test.

---

## Conclusion

State what the system can be trusted to do, what it cannot yet do, and why the evidence supports that boundary.

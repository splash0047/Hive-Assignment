# High-Efficiency Implementation Plan

## Overall strategy

Use an **evaluation-driven build**.

Do not spend the first half of the assignment building a sophisticated agent. Build the golden-set schema, baseline metrics, and evaluation harness early so every improvement has measurable evidence.

Estimated focused implementation time: **12-18 hours**, excluding the candidate's manual labeling time.

---

## Phase 0 - Project scaffold (30-45 min)

### Tasks
- initialize Git repo;
- add `pyproject.toml`;
- configure `uv`;
- add Ruff and pytest;
- add environment template;
- create package layout;
- add CI;
- add config loader;
- add deterministic seed helper;
- add Pydantic/dataclass schemas.

### Exit criteria
- `uv sync` works;
- `pytest -q` works;
- `ruff check .` works;
- CI starts successfully.

---

## Phase 1 - Data ingestion and brand profiler (1.5-2 h)

### Tasks
- download/read Kaggle dataset;
- validate schema;
- implement thread reconstruction;
- identify outbound brand accounts;
- compute brand statistics;
- produce `artifacts/brand_profile.csv`;
- generate top candidate comparison.

### Selection heuristic

Prefer a brand with:
- >= 5k usable customer->support pairs after sampling;
- high direct-reply coverage;
- modest number of recurring topics;
- lower generic-DM-only rate;
- useful public troubleshooting language.

### Default choice
Use `SpotifyCares` if the profiler supports the hypothesis; otherwise select the best-scoring brand.

### Exit criteria
- thread tests pass;
- selected brand is frozen in config;
- decision log records why.

---

## Phase 2 - Curated historical corpus (1-1.5 h)

### Tasks
- create deterministic selected-brand sample;
- clean text;
- create customer->brand support pairs;
- mark low-information support replies;
- save parquet;
- create manifest with source IDs and hash.

### Important
Never delete tweet IDs from curated artifacts. They are your grounding evidence trail.

### Exit criteria
- `brand_pairs.parquet`;
- `historical_corpus.parquet`;
- data-quality report;
- deterministic rebuild test.

---

## Phase 3 - Intent discovery and labeling workflow (2-4 h + manual labeling)

### Part A: taxonomy discovery
- sample 2k-5k training messages;
- embed;
- run deterministic clusters;
- inspect representative examples;
- draft 8-12 intent definitions;
- manually review taxonomy.

### Part B: golden-set sampler
Generate 180-220 examples using stratified/diversity-aware sampling:
- common intents;
- long-tail/rare cases;
- short/ambiguous messages;
- high-risk cases;
- multi-turn/context-dependent examples.

Recommended:
- 200 total examples;
- 50 calibration;
- 150 locked test.

### Part C: human labeling
The candidate personally labels:
- intent;
- should escalate?;
- escalation reason;
- risk tags;
- notes.

Build a small Streamlit UI if it saves time.

### Golden-set freeze
Write:
- SHA-256 hash;
- timestamp;
- taxonomy version;
- counts per intent;
- calibration/test split IDs.

After freeze, never modify locked labels without explicitly versioning `golden_v2`.

### Exit criteria
- taxonomy document;
- golden set;
- freeze manifest.

---

## Phase 4 - Baselines first (1-1.5 h)

### Trivial baseline
- majority intent;
- generic canned acknowledgement;
- always escalate OR a fixed conservative routing rule.

### Simple baseline
- TF-IDF + Logistic Regression;
- TF-IDF/cosine nearest historical support reply;
- confidence threshold routing.

### Why now
These baselines expose whether the final system actually adds value.

### Exit criteria
- baseline result JSON/CSV;
- intent metrics;
- baseline reply outputs;
- baseline routing metrics.

---

## Phase 5 - Final intent classifier + retrieval (1.5-2 h)

### Final classifier
- sentence embeddings;
- Logistic Regression;
- calibration if necessary.

### Retrieval
- embed curated historical customer messages;
- normalize vectors;
- FAISS exact similarity;
- top-k 5-8;
- optional predicted-intent filter.

### Evaluation
- macro-F1;
- per-intent F1;
- confusion matrix;
- retrieval same-intent precision@k;
- manual relevance sample.

### Exit criteria
- classifier beats simple baseline meaningfully;
- retrieval examples look useful;
- no train/test leakage.

---

## Phase 6 - Grounded reply generation (1.5-2 h)

### Prompt
Use 3-5 strongest historical evidence examples.

Return structured JSON:
- draft reply;
- evidence IDs;
- confidence;
- insufficient-evidence flag.

### Post-generation checks
- evidence exists;
- no unsupported URLs;
- no account-state claims;
- no hallucinated actions;
- concise response;
- no PII.

### Caching
Cache each generation by:
`hash(model + prompt_version + input_text + evidence_ids)`.

This controls cost and makes evaluation reproducible.

### Exit criteria
- generator produces valid structured output;
- failures are explicit and testable.

---

## Phase 7 - Escalation policy and calibration (1-1.5 h)

### Features
- intent confidence;
- confidence margin;
- retrieval score;
- risk tags;
- insufficient-evidence flag;
- validator status;
- ambiguity heuristic.

### Tune only on 50-example calibration split.

Primary safety metric:
**false auto-handle rate**.

Secondary:
- escalation recall;
- escalation precision;
- auto-handle coverage.

Plot:
`auto-handle precision vs coverage` across thresholds.

Pick a threshold operating point deliberately and document it.

### Exit criteria
- one frozen threshold set;
- reason codes for every escalation;
- locked test still untouched.

---

## Phase 8 - Full evaluation harness (2-3 h)

### Intent metrics
- accuracy;
- macro precision;
- macro recall;
- macro-F1;
- per-class F1;
- confusion matrix.

### Escalation metrics
- precision/recall/F1 for ESCALATE;
- false-auto-handle rate;
- coverage;
- auto-handle correctness;
- risk-case recall.

### Reply metrics
Automated:
- evidence citation rate;
- unsupported URL rate;
- reply non-empty rate;
- length;
- retrieval similarity statistics.

LLM judge:
- groundedness;
- helpfulness/actionability;
- correctness/no unsupported claim;
- tone;
- safety;
- overall acceptability.

### Human agreement
Take 40-60 final-system replies.
Candidate scores them using the same judge rubric.

Compare:
- exact/within-one agreement;
- weighted Cohen's kappa for ordinal ratings;
- Spearman correlation for overall score;
- binary acceptable/unacceptable agreement.

### Exit criteria
- final metrics saved;
- judge-human agreement evidence saved;
- result plots generated.

---

## Phase 9 - Failure analysis (1-1.5 h)

Select top 5 failure modes based on locked-test failures.

For each:
- 2-4 real examples;
- expected behavior;
- actual behavior;
- root-cause hypothesis;
- likely fix;
- whether fix would risk another metric.

Likely categories:
1. multi-intent/ambiguous query;
2. insufficient context from previous turns;
3. retrieval finds lexically similar but wrong-resolution examples;
4. account-specific/security issue accidentally looks routine;
5. outdated or inconsistent historical support policy.

Do not force these if the data shows different failures.

---

## Phase 10 - Report + README + final audit (1.5-2 h)

### README
Must include:
- exact setup;
- 15-minute fast evaluation command;
- architecture;
- result table;
- known limitations;
- dataset attribution;
- no-key/cached path if applicable.

### Report
Max 6 pages:
1. framing and scope;
2. data/brand/taxonomy;
3. system;
4. evaluation/results/baselines;
5. failure analysis + misleading headline number;
6. next week + conclusion.

### Final audit
- clean clone test;
- no secrets;
- no full dataset committed;
- report numbers exactly match artifacts;
- every chart reproducible;
- decision log has 10-15 items;
- citations complete;
- locked test hash unchanged.

---

# Critical path

If time is running out, preserve in this order:

1. golden set;
2. baselines;
3. evaluation harness;
4. correct thread/data pipeline;
5. safe retrieval-based system;
6. report;
7. optional improvements;
8. UI last.

A strong simple system with rigorous proof is better than a complex agent with weak evidence.

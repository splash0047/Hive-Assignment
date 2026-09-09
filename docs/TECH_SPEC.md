# Technical Specification

## 1. Recommended stack

| Layer | Recommended choice | Why |
|---|---|---|
| Runtime | Python 3.12 | Current, stable, easy evaluation |
| Env/package manager | `uv` | Fast clean setup |
| Data | pandas + pyarrow | CSV/parquet processing |
| Baseline ML | scikit-learn | Explainable TF-IDF / Logistic Regression |
| Embeddings | sentence-transformers | Strong local semantic representation |
| Retrieval | FAISS `IndexFlatIP` | Exact, transparent, no server |
| LLM | provider adapter | Easy provider swap |
| Config | YAML + Pydantic/dataclasses | Explicit, validated |
| CLI | Typer | Clean commands |
| Evaluation | sklearn metrics + custom scorers | Transparent |
| Experiment tracking | optional MLflow | Useful but not required for core path |
| Tests | pytest | Standard |
| Lint/format | Ruff | Fast |
| CI | GitHub Actions | Reproducibility |
| Plots | matplotlib | Confusion matrices / calibration / coverage |
| Optional labeling UI | Streamlit | Speeds manual annotation |

Avoid a managed vector DB or multi-agent framework.

## 2. Suggested repository structure

```text
hiver-support-agent/
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ .env.example
├─ .gitignore
├─ configs/
│  ├─ default.yaml
│  └─ fast_eval.yaml
├─ data/
│  ├─ README.md
│  ├─ raw/                 # ignored
│  ├─ interim/             # ignored or regenerated
│  ├─ curated/
│  │  ├─ brand_pairs.parquet
│  │  ├─ historical_corpus.parquet
│  │  └─ manifest.json
│  └─ golden/
│     ├─ golden_eval.csv
│     ├─ labeling_guidelines.md
│     └─ freeze_manifest.json
├─ src/hiver_agent/
│  ├─ __init__.py
│  ├─ cli.py
│  ├─ config.py
│  ├─ schemas.py
│  ├─ data/
│  │  ├─ load.py
│  │  ├─ threads.py
│  │  ├─ clean.py
│  │  ├─ sample.py
│  │  └─ brand_profile.py
│  ├─ intents/
│  │  ├─ taxonomy.py
│  │  ├─ discover.py
│  │  ├─ baseline.py
│  │  └─ classifier.py
│  ├─ retrieval/
│  │  ├─ build.py
│  │  ├─ index.py
│  │  └─ retrieve.py
│  ├─ generation/
│  │  ├─ prompts.py
│  │  ├─ provider.py
│  │  ├─ draft.py
│  │  └─ validate.py
│  ├─ routing/
│  │  ├─ risk.py
│  │  └─ escalate.py
│  └─ eval/
│     ├─ dataset.py
│     ├─ metrics.py
│     ├─ judge.py
│     ├─ agreement.py
│     ├─ baselines.py
│     ├─ runner.py
│     └─ report.py
├─ scripts/
│  ├─ download_data.py
│  ├─ build_curated_data.py
│  ├─ make_labeling_sample.py
│  └─ freeze_eval.py
├─ tests/
│  ├─ test_threads.py
│  ├─ test_cleaning.py
│  ├─ test_retrieval.py
│  ├─ test_escalation.py
│  ├─ test_schemas.py
│  └─ test_metrics.py
├─ artifacts/
│  ├─ models/
│  ├─ indexes/
│  ├─ eval/
│  └─ figures/
├─ docs/
│  ├─ report.md
│  ├─ failure_analysis.md
│  └─ decision_log.md
└─ .github/workflows/ci.yml
```

## 3. Data model

### Support pair

```python
SupportPair(
    pair_id: str,
    thread_id: str,
    customer_tweet_id: str,
    support_tweet_id: str,
    customer_text: str,
    support_text: str,
    created_at_customer: datetime,
    created_at_support: datetime,
    response_latency_seconds: float | None,
    brand: str,
)
```

### Golden example

```python
GoldenExample(
    example_id: str,
    tweet_id: str,
    text: str,
    intent_label: str,
    escalation_label: bool,
    escalation_reason: str,
    risk_tags: list[str],
    split: Literal["calibration", "locked_test"],
    annotator_notes: str | None,
)
```

### Agent output

```python
AgentOutput(
    intent: str,
    intent_confidence: float,
    action: Literal["AUTO_HANDLE", "ESCALATE"],
    reason: str,
    reply: str | None,
    evidence_ids: list[str],
    retrieval_scores: list[float],
)
```

## 4. Thread reconstruction

Do not assume every row is a clean customer->brand pair.

Algorithm:

1. Index rows by `tweet_id`.
2. Normalize `in_response_to_tweet_id`.
3. For each row, walk parents until no parent is found.
4. Use the root tweet ID as a thread ID.
5. Sort thread members by `created_at`.
6. Identify brand author IDs from outbound rows.
7. Keep threads that contain both inbound and outbound messages.
8. Create candidate pairs from each inbound customer turn to the next direct brand response.
9. Preserve multi-turn context separately for future analysis.
10. Reject malformed loops, missing parents, or cross-brand anomalies and log counts.

Tests:
- one-to-one reply;
- multi-turn alternating thread;
- multiple replies to same tweet;
- missing parent;
- split brand response into multiple tweets;
- timestamp ties.

## 5. Brand profiling and selection

Implement `brand_profile.py`.

For the top 10 outbound brands, compute:
- outbound volume;
- inbound customer volume;
- inbound-with-direct-reply rate;
- usable pair count;
- median thread length;
- median response length;
- generic handoff rate;
- approximate topic compactness;
- URL rate;
- private-message/DM rate.

Shortlist 3-5 brands.

### Default hypothesis

Start with:
- SpotifyCares
- AppleSupport
- AmazonHelp
- Uber_Support
- one airline support account

Prefer the brand with:
- enough volume for diversity;
- repeated, learnable issues;
- informative public answers;
- fewer cases requiring hidden account state.

Record final selection and evidence.

## 6. Intent taxonomy discovery

Use only the training/discovery pool, never the locked test set.

Procedure:
1. deterministically sample 2k-5k inbound messages;
2. clean mentions/URLs while preserving meaningful text;
3. embed using a compact sentence-transformer;
4. cluster with KMeans for deterministic behavior, exploring k=8..14;
5. inspect top terms and 20 representatives per cluster;
6. optionally ask an LLM to propose cluster names;
7. manually merge/split into ~8-12 final intents;
8. write precise positive/negative definitions.

Prefer a taxonomy like:
- login/access
- subscription/billing
- payment/refund
- playback/technical
- app/device compatibility
- account/profile changes
- content/availability
- cancellation
- security/suspicious activity
- service outage
- how-to/general information
- other/unclear

These are illustrative only. The final labels must come from selected-brand data.

## 7. Classifiers

### Trivial baseline
- always predict the majority intent.

### Simple baseline
- word/character TF-IDF;
- Logistic Regression;
- class weights if needed.

### Final classifier
Recommended:
- sentence-transformer embeddings;
- Logistic Regression or LinearSVC + calibrated probabilities;
- probability calibration on calibration subset if needed.

Why:
- local and fast;
- strong on semantic paraphrases;
- easy to explain;
- deterministic;
- much simpler than fine-tuning a transformer.

## 8. Retrieval

Build a FAISS exact index over normalized embeddings for historical customer messages.

Use cosine similarity:
- L2-normalize embeddings;
- use `IndexFlatIP`.

Retrieve top 5-8 examples.

Filter:
- same predicted intent if confidence is high;
- exclude exact duplicate/current tweet;
- exclude poor/empty replies;
- optionally prefer examples with actionable support replies.

Return evidence IDs and similarities.

## 9. Reply generation

Prompt should include:
- brand identity;
- customer message;
- predicted intent;
- retrieved historical evidence;
- constraints;
- JSON output schema.

Generation constraints:
- do not claim to access account state;
- do not claim an action happened;
- do not invent policies, fees, or deadlines;
- do not introduce a new URL unless found in evidence;
- prefer concise Twitter-style responses;
- if evidence conflicts or is inadequate, request escalation;
- cite evidence IDs used.

A post-generation validator should flag:
- unsupported URLs;
- no evidence ID;
- forbidden claims;
- excessive length;
- leaked anonymized IDs;
- obvious PII.

## 10. Escalation policy

Make escalation an explicit Python decision layer.

Suggested features:
- intent probability;
- margin between top 2 intents;
- top retrieval similarity;
- mean top-3 similarity;
- risk tags;
- predicted intent;
- reply-validator result;
- multi-issue heuristic;
- presence of account/security/payment terms.

Auto-handle only if all required conditions pass.

Initial rules:

```text
ESCALATE if:
- high-risk intent/risk tag
OR intent_confidence < threshold
OR top_retrieval_similarity < threshold
OR no usable evidence
OR reply validator fails
OR multi-issue ambiguity is detected

otherwise AUTO_HANDLE
```

Tune thresholds only on the calibration subset.

## 11. Provider interface

Use a tiny adapter:

```python
class LLMProvider(Protocol):
    def generate_json(self, messages: list[dict], schema: type[T]) -> T: ...
```

Implement only the provider actually used.

Optional fallback:
- cached/precomputed LLM outputs for final fast evaluation;
- local model if practical.

## 12. Reproducibility artifacts

Freeze:
- selected brand;
- sampled tweet IDs;
- intent taxonomy version;
- golden-set IDs and hash;
- classifier model;
- embedding model name;
- FAISS index;
- config;
- prompt version;
- generation model;
- judge model;
- evaluation outputs.

Every result table should include an experiment/config hash.

## 13. Fast evaluation mode

`fast_eval.yaml` should:
- load curated brand subsample;
- load frozen classifier/index;
- load golden set;
- reuse cached final-system outputs when API access is not available;
- recompute deterministic metrics/plots;
- optionally rerun LLM judge only when a key is provided.

This is the safest way to guarantee <15-minute reproduction.

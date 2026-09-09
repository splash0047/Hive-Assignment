# Product Requirements Document

## 1. Problem

Given real customer-support Twitter conversations for a selected brand, build an AI support agent that can:

- classify each new inbound customer message into a compact intent taxonomy derived from that brand's data;
- draft a response grounded in the way the brand historically handled similar problems;
- decide whether the message is safe to auto-handle or should be escalated to a human;
- provide a reason for the decision.

The project must emphasize evidence that the agent is trustworthy.

## 2. Scope

### In scope

- one brand only;
- a deterministic subsample of the dataset;
- thread reconstruction;
- intent discovery and human-reviewed taxonomy;
- intent classification;
- retrieval over historical support interactions;
- LLM-assisted reply drafting;
- explicit escalation policy;
- evaluation against a human-labelled golden set;
- two baselines;
- human-vs-LLM-judge agreement study;
- failure analysis;
- reproducible CLI pipeline;
- concise report/README.

### Out of scope

- production API scaling;
- real Twitter/X integration;
- authentication;
- CRM integration;
- long-term memory;
- multi-brand routing;
- voice support;
- fine-tuning a large language model;
- autonomous account actions;
- refunds, password changes, or any real customer-account mutation;
- polished frontend before required deliverables are complete.

## 3. Primary user story

As a support operations team, I want an AI system to safely handle common, well-understood questions while escalating uncertain, sensitive, or poorly grounded cases so that I can automate repetitive support without creating unacceptable customer risk.

## 4. Functional requirements

### FR-1 Data ingestion

The system shall read the Kaggle CSV with columns similar to:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

It shall validate required columns and data types.

### FR-2 Thread reconstruction

The system shall reconstruct parent-child relationships using tweet IDs and response fields.

It shall preserve:
- chronological order;
- inbound/outbound direction;
- brand identity;
- thread ID/root ID;
- original tweet ID;
- cleaned text.

### FR-3 Brand selection

A profiling script shall compare candidate brands using:
- number of inbound messages;
- number/fraction with brand replies;
- thread depth;
- repeated-topic density;
- diversity of messages;
- fraction of generic "DM us/contact support" replies;
- number of usable historical resolution pairs.

The final brand shall be frozen in configuration and documented.

### FR-4 Intent taxonomy

The project shall define approximately **8-12 intents**, plus `other/unclear` if needed.

Taxonomy discovery may use:
- embedding + clustering;
- keyword frequency;
- manual sample inspection;
- optional LLM cluster naming.

Final taxonomy definitions must be human-reviewed and documented.

### FR-5 Intent classification

The final system shall return:
- `intent`
- `intent_confidence`
- optional top alternatives

The classifier must support a deterministic local evaluation path.

### FR-6 Historical retrieval

Given a customer message, the system shall retrieve top-k historically similar support examples from the selected brand.

Each evidence item shall include:
- source customer tweet ID;
- source support reply ID;
- customer text;
- support reply text;
- similarity score;
- intent if available.

### FR-7 Reply drafting

The generation layer shall:
- use only the incoming message and retrieved historical evidence;
- mimic useful brand tone without fabricating policy;
- avoid claiming account-specific actions were performed;
- avoid exposing private information;
- return evidence IDs used;
- explicitly abstain/escalate when the evidence is insufficient.

### FR-8 Escalation

The system shall decide `AUTO_HANDLE` or `ESCALATE`.

Escalation reason categories should include:
- low intent confidence;
- low retrieval relevance;
- high-risk intent;
- account-specific/private action required;
- possible security/fraud issue;
- multiple/ambiguous intents;
- missing evidence;
- unsupported request;
- generated-response safety/grounding failure.

### FR-9 Structured output

Recommended schema:

```json
{
  "intent": "playback_issue",
  "intent_confidence": 0.88,
  "action": "AUTO_HANDLE",
  "action_confidence": 0.82,
  "reason": "High-confidence common troubleshooting issue with strong historical evidence.",
  "reply": "Sorry about that...",
  "evidence_ids": ["pair_0123", "pair_0931"],
  "retrieval_scores": [0.84, 0.80]
}
```

### FR-10 Reproducibility

A new evaluator shall be able to reproduce headline results with a command such as:

```bash
uv sync
uv run python -m hiver_agent.cli eval --fast
```

The fast path must avoid needing the full dataset.

## 5. Non-functional requirements

- deterministic sampling and splits;
- readable and interview-friendly code;
- secrets only through environment variables;
- provider-independent LLM interface;
- graceful operation if no LLM key is present for non-LLM metrics;
- tests for critical data and evaluation logic;
- artifact hashes or frozen manifests for final evaluation;
- clear logs and failure messages.

## 6. Success criteria

Targets, not guaranteed claims:

- Intent classification macro-F1: >= 0.80 if taxonomy/data support it.
- Escalation recall for human-required/risky cases: >= 0.90.
- False-auto-handle rate on risky cases: <= 5%.
- Auto-handle coverage: ideally 40-70%, depending on brand.
- Human acceptable-reply rate on selected auto-handled cases: >= 75%.
- LLM-judge agreement with human: weighted kappa or rank correlation around >= 0.60 is a good target.
- Reproduction runtime: < 15 minutes in `--fast` mode.

If targets are missed, report the true results and explain why.

## 7. Suggested primary headline

Report:

**Trusted auto-handle precision at coverage**

Example format:

> "At 52% auto-handle coverage, 87% of selected messages passed the locked reply-quality acceptance rubric."

Always show the denominator and coverage.

## 8. Constraints

The project is an evaluation exercise, not a production deployment. Avoid inflating scope.

The full source dataset is very large. Use deterministic subsampling and prebuilt lightweight artifacts for the evaluation path.

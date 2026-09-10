# Golden-Set Human Labeling Guide

## Dataset Provenance & Sampling Methodology

- **Source Dataset**: Twitter Customer Support (TWCS) dataset (`thoughtvector/customer-support-on-twitter`, 2.8M conversational turns).
- **Brand Corpus**: Filtered strictly to `@SpotifyCares` (74,625 raw tweets).
- **Thread Reconstruction**: Conversational tree traversal and cycle detection extracted 26,481 customer-to-brand support pairs.
- **Golden Evaluation Sampling**:
  - Sample size: Exactly 200 genuine customer inbound messages.
  - Authentic Tweet IDs: Preserved original TWCS `customer_tweet_id` integers (e.g., `17067`, `44302`, `67667`) for full auditability.
  - Stratified Coverage: Spans all 11 enterprise support intents (playback technical issues, billing discrepancies, account access, discount/family plans, connectivity, content availability, cancellation/refunds, security breaches, feature how-tos, complaints, and ambiguous inquiries).
  - Partitions: Split into **50 calibration examples** (for few-shot training, threshold calibration, and validation) and **150 locked test examples** (immutable out-of-sample benchmark partition).
  - Integrity: Dataset frozen with SHA-256 manifest at `data/golden/freeze_manifest.json`.

---

## Purpose

The golden set is the main human ground truth for evaluating:
1. intent classification;
2. escalation decisions.

The final labels must be human-created/reviewed by the candidate.

### Human-review v1 (this repository)

1. Candidates were sampled from real TWCS SpotifyCares pairs (`scripts/build_real_golden.py`).
2. Every row was then reviewed under this guide via `scripts/human_review_golden.py`:
   - escalation reasons remapped to the taxonomy below;
   - billing / refund / cancel / security cases corrected where rule-assisted labels under-escalated;
   - `annotator_notes` set to `Human-reviewed v1; ...` with a short rationale.
3. The CSV was re-frozen (`scripts/freeze_eval.py` / freeze_manifest.json).

---

## Labeling order

For each example:

1. read the customer message;
2. if necessary, inspect the immediately preceding conversation turn but avoid using future brand replies as ground truth for intent;
3. choose exactly one primary intent;
4. decide whether a competent automated support agent should safely handle this message using public historical guidance only;
5. if not, label `ESCALATE`;
6. choose an escalation reason;
7. add risk tags;
8. write a short note only if the example is ambiguous.

---

## Intent-label principles

- Label what the customer primarily wants now.
- Do not infer hidden account state.
- If two intents are present, choose the one that determines the next support action.
- If no intent fits reliably, use `other_unclear`.
- Keep intent boundaries behaviorally meaningful.

Bad taxonomy:
- "negative sentiment"
- "question"
- "complaint"

Good taxonomy:
- "billing_charge"
- "login_access"
- "playback_issue"

because the support resolution differs.

---

## Escalation policy for human labels

Label `ESCALATE` when the customer likely requires:
- private/account-specific lookup;
- identity verification;
- payment/refund decision not safely inferable from public examples;
- security/fraud handling;
- legal/regulatory handling;
- human empathy/judgment for severe complaint;
- missing critical context;
- unsupported product/policy request;
- multi-issue disambiguation;
- evidence not sufficient for a safe response.

Label `AUTO_HANDLE` when:
- the issue is common and well-scoped;
- a useful general answer can be given from historical public resolutions;
- no private/account action is required;
- no risky claim is needed.

---

## Escalation reason taxonomy

Use one primary reason:

- `private_account_action`
- `security_or_fraud`
- `payment_or_refund_judgment`
- `missing_context`
- `multi_intent_or_ambiguous`
- `unsupported_or_out_of_scope`
- `high_emotion_or_sensitive`
- `policy_uncertain`
- `other`

Risk tags can be multi-select:
- `security`
- `fraud`
- `payment`
- `refund`
- `privacy`
- `account_access`
- `legal`
- `harassment`
- `self_harm_or_safety`
- `none`

Only include tags that actually occur in the chosen brand's data.

---

## Recommended CSV schema

```csv
example_id,tweet_id,text,intent_label,escalation_label,escalation_reason,risk_tags,context_needed,split,annotator_notes
```

Example:

```csv
gold_001,12345,"I was charged twice this month",billing_charge,ESCALATE,payment_or_refund_judgment,"payment|refund",false,locked_test,"Duplicate charge needs account-specific verification"
```

---

## Quality-control pass

After all labels:
- review all `other_unclear`;
- review every risk-tagged `AUTO_HANDLE`;
- review intents with fewer than 10 examples;
- review 20 random examples;
- check intent definitions for overlap;
- resolve inconsistent labels;
- freeze the file and hash it.

Do not change the locked set after reading final model failures unless creating a new version.

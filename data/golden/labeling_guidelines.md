# Golden-Set Human Labeling Guide

## Purpose

The golden set is the main human ground truth for evaluating:
1. intent classification;
2. escalation decisions.

The final labels must be human-created/reviewed by the candidate.

AI may help prepare the sample or propose a taxonomy, but do not blindly accept AI-generated golden labels.

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

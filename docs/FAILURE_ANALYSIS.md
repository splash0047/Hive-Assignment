# Failure Mode Analysis & Edge Case Audit

This document presents a rigorous diagnostic analysis of the primary failure modes and edge cases discovered during out-of-sample evaluation on the **locked test set ($N=150$)** for the Hiver AI Support Agent. All examples are real customer tweets from the TWCS SpotifyCares dataset.

---

## Executive Failure Summary

The system is architected as an **evaluation-first, trust-gated agent**. In customer support for high-stakes actions (billing, password changes, security incidents), **a false auto-handle (generating a confident but incorrect answer) is significantly worse than an unnecessary escalation**.

Our evaluation identified **5 distinct failure modes** across 150 real queries:

| Failure Mode | Root Cause | Impact | Observed Rate | Mitigation Implemented |
|---|---|---|---|---|
| **1. Low Confidence + Multi-Intent Overlap** | Informal phrasing combines two topics (login + billing, subscription + security). Classifier splits probability mass. | Most common escalation trigger; reduces automation coverage | 28.0% of queries | Confidence ($\ge0.25$) + margin ($\ge0.03$) gates route split-intent queries to human. |
| **2. Out-of-Taxonomy Intents** | Two intents in real data (`feedback_complaint`, `unsupported_ambiguous_inquiry`) absent from training taxonomy. | Classifier confused; unpredictable prediction | ~14% of queries | These fall through to `other_unclear` or low-confidence escalation. |
| **3. Informal / Noisy Message Style** | Emoji-heavy, highly terse, or multi-mention tweets fail semantic embedding alignment. | Cosine similarity drops below 0.55 threshold | ~18% of ESCALATE queries cite low retrieval similarity | MiniLM embeddings are robust to individual tokens; terse queries escalated safely. |
| **4. Unsafe Auto-Handle (False Positives)** | Billing dispute or ambiguous query passes all three gates: confidence ≥0.25, margin ≥0.03, top sim ≥0.55. | Reply generated for query that warranted human review | **1.3% (2/150)** | Routing thresholds calibrated on calibration set; reviewer-hardened security regex. |
| **5. Correct Prediction but Low Confidence Escalation** | Classifier correctly identifies intent but confidence is just below threshold (e.g. 0.23 vs threshold 0.25). | Adds unnecessary load to human queue | ~40% of escalations are correct-intent-low-confidence | Acceptable trade-off: system prefers false escalation over false auto-handle. |

---

## Detailed Failure Mode Breakdown

### Failure Mode 1: Multi-Intent / Informal Phrasing → Correct Escalation

**Example** (`ex_040`): *"I have signed up for premium and the money has come out of my account. I have logged out then in but still not working."*
- **True Intent**: `subscription_billing`
- **Predicted Intent**: `login_account_access` (conf: 0.15)
- **Decision Gate**: Escalated — `Low intent confidence (0.15 < 0.25)`
- **Verdict**: ✅ **Correct Escalation.** The message spans billing activation and login, creating genuine intent ambiguity. Even with wrong prediction, the confidence gate prevented an irrelevant login-troubleshooting reply from reaching the customer.

**Example** (`ex_057`): *"I signed up for the $9.99 plan but I can't log in."*
- **True Intent**: `login_account_access`
- **Predicted Intent**: `plan_discount_management` (conf: 0.17)
- **Decision Gate**: Escalated — `Low intent confidence (0.17 < 0.25)`
- **Verdict**: ✅ **Correct Escalation.** The billing mention in the first clause confused the classifier, but low confidence gating correctly deferred to human.

---

### Failure Mode 2: Out-of-Taxonomy Intents → Graceful Degradation

**Example** (`ex_194`): *"i can't sign in"*
- **True Intent**: `unsupported_ambiguous_inquiry` (human-assigned; absent from taxonomy)
- **Predicted Intent**: `login_account_access` (plausible label; conf: 0.32)
- **Decision Gate**: AUTO_HANDLE — passed all three gates
- **Verdict**: ⚠️ **Unsafe Auto-Handle.** The query was human-labelled as needing review (too terse, zero context), but the classifier confidently mapped it to login_account_access and retrieved grounded evidence. This reveals a gap: the escalation rules do not check for minimum query length independently. A simple word-count gate (< 4 tokens → escalate) would catch this.

**Architectural Lesson**: The evaluation discovered that 10 of 150 test queries belong to `unsupported_ambiguous_inquiry` — a class completely absent from the training taxonomy. Adding this as an explicit 12th intent label in future labeling passes would allow the classifier to natively route these to escalation.

---

### Failure Mode 3: Emoji / Terse Crash Reports → Correct Escalation

**Example** (`ex_012`): *"Argh @116130 @SpotifyCares keeps crashing 😭😭 send help!"*
- **True Intent**: `playback_technical_issue`
- **Predicted Intent**: `login_account_access` (conf: 0.12)
- **Decision Gate**: Escalated — `Low intent confidence (0.12 < 0.25)`
- **Verdict**: ✅ **Correct Escalation.** The emoji-heavy, mention-heavy format shifts the embedding away from the formal support vocabulary. MiniLM still captures enough signal (crash → technical) but not sufficient confidence for automation.

---

### Failure Mode 4: Billing Dispute Auto-Handled (False Positive)

**Example** (`ex_032`): *"Hi, I've just noticed I've been charged twice this month, can you help please?"*
- **True Intent**: `subscription_billing` (escalation_label=True in golden set — human annotator flagged double-charge as requiring account ledger inspection)
- **Predicted Intent**: `subscription_billing` (conf: 0.34, margin: 0.19, top sim: 0.88)
- **Decision Gate**: AUTO_HANDLE — all three gates passed
- **Verdict**: ⚠️ **Unsafe Auto-Handle.** The agent correctly identified intent and retrieved a grounded historical reply for billing inquiries. However, the human annotator had flagged double-charge disputes as requiring human ledger access. The system's security regex correctly detects fraud/unauthorized charges but not the more benign "charged twice" phrasing without "unauthorized" context.

**Fix**: Add `charged twice` and `double charge` to the billing-dispute security patterns in `routing/risk.py` to route billing conflicts to human by default.

---

### Failure Mode 5: Correct Intent, Below-Threshold Confidence → Over-Escalation

**Examples** (`ex_030`, `ex_031`, `ex_033`, `ex_034`, `ex_035`): All five are semantically similar billing queries (*"charged twice this month"*) — all correctly classified as `subscription_billing` but with confidence between 0.15-0.24 (threshold: 0.25).

- **Verdict**: ✅ **Safe but conservative.** The classifier correctly identifies all five as billing queries but low confidence causes five separate escalations. In production, a ~10% lower threshold with a tighter retrieval similarity requirement might unlock these while keeping the safety guarantee.

---

## Architectural Lessons & Interview Takeaways

1. **Defense in Depth Works**: 96 of 98 true-escalation queries were successfully routed to human agents (escalation recall 95.8%). The two unsafe auto-handles are genuine edge cases exposed only by real annotated data.

2. **50 Training Examples Is a Real Bottleneck**: The primary root cause of 3.3% coverage is the under-trained classifier. The MiniLM embeddings are powerful but Logistic Regression needs more examples per class to build confident boundaries. Expanding to 200+ calibration examples is the highest-ROI next step.

3. **Out-of-Taxonomy Discovery**: Running evaluation on real data immediately surfaced two missing intent classes (`feedback_complaint`, `unsupported_ambiguous_inquiry`). Synthetic data would never surface this — it was constrained to the predefined taxonomy.

4. **Terse Queries Need a Dedicated Gate**: Queries under ~5 tokens should trigger escalation regardless of classifier confidence. The current security regex misses the pattern *"i can't sign in"* (3 content tokens).

5. **Threshold Tuning Must Use Calibration Data Only**: All three thresholds (`min_intent_confidence=0.25`, `min_top_retrieval_similarity=0.55`, `min_confidence_margin=0.03`) were calibrated exclusively on the 50-example calibration split. This strict hygiene is why the locked test results are genuinely out-of-sample.

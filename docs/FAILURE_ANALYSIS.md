# Failure Mode Analysis & Edge Case Audit

This document presents a rigorous diagnostic analysis of the primary failure modes and edge cases discovered during out-of-sample evaluation on the locked test set ($N=100$) for the Hiver AI Support Agent.

---

## Executive Failure Summary

The system is architected as an **evaluation-first, trust-gated agent**. In customer support for high-stakes actions (billing, password changes, security incidents), **a false auto-handle (generating a confident but incorrect answer) is significantly worse than an unnecessary escalation**.

Our evaluation identified **5 distinct failure modes**:

| Failure Mode | Root Cause | Impact | Observed Frequency | Mitigation Implemented |
|---|---|---|---|---|
| **1. Ambiguous Diagnostic Context** | Customer query contains no device, OS, or specific symptom (e.g. *"Why is it not working?"*) | Retrieval retrieves generic advice; risk of irrelevant answer | 8.0% | Regex-based `assess_customer_message_risk` flags queries $\le 3$ words or matching ambiguity patterns, forcing clarification escalation. |
| **2. Compound Intent Overlap** | Customer query expresses multiple requests simultaneously (e.g. *"I want to cancel Premium and get a refund"*) | Single-label classifier assigns highest probability to one intent, leaving the secondary request unaddressed | 6.0% | Margin check ($\Delta < 0.05$) detects competitive multi-intents and routes to human agent when both intents are high-stakes. |
| **3. Specific Hardware / Third-Party Codes** | Niche console or platform error codes (e.g. *"PS5 error CE-108255-1"*) | Historical corpus lacks exact string match for rare console codes | 4.0% | Retrieval similarity threshold ($< 0.50$) prevents ungrounded guessing when cosine similarity drops. |
| **4. Slang & Out-of-Vocabulary Phrasing** | Non-standard abbreviations or emotive expressions | Embedding shifts slightly away from formal support documentation | 3.0% | Sublinear TF scaling in text cleaning; normalized sentence embeddings provide robust semantic proximity. |
| **5. Subtly Hostile Account Takeover** | Phrased as ordinary login difficulty rather than hack (e.g. *"Email address doesn't exist anymore"*) | Classifier risks predicting standard `login_account_access` rather than `security_compromised_account` | 2.0% | Security keyword heuristic detects keywords (`hacked`, `stolen`, `stranger`, `someone else`, `unauthorized`) independently of classifier output. |

---

## Detailed Failure Mode Breakdown

### Failure Mode 1: Terse Inquiries with Zero Context
- **Example**: `ex_187`: *"Why is it not working?"*
- **Agent Behavior**: The intent classifier assigned low confidence spread across `playback_technical_issue` ($0.18$) and `login_account_access` ($0.15$). The margin was $0.03$.
- **Decision Gate**: The routing policy correctly identified `context_needed = True` and escalated with reason: `Missing required context; clarification needed`.
- **Verdict**: **True Positive Escalation**. The agent correctly refused to generate generic troubleshooting steps when the problem could be network, audio, billing, or login.

### Failure Mode 2: Multi-Intent Conflict (Cancellation vs Refund)
- **Example**: `ex_120`: *"I cancelled 3 months ago but you are still billing me every month. Refund me!"*
- **Agent Behavior**: Triggered high similarity with both `cancellation_refund` ($0.29$) and `subscription_billing` ($0.28$).
- **Risk Assessment**: Detected recurring unauthorized charges after cancellation.
- **Decision Gate**: Escalated with reason: `Billing dispute after claimed cancellation`.
- **Verdict**: **True Positive Escalation**. Direct financial disputes cannot be resolved by an automated bot and require human ledger inspection.

### Failure Mode 3: Console / Third-Party Device Codes
- **Example**: `ex_087`: *"PS5 Spotify app gives error code CE-108255-1 on launch."*
- **Agent Behavior**: The query was accurately recognized as `device_connectivity`, but the exact error code `CE-108255-1` had moderate cosine similarity ($0.48$) against general PlayStation guidance ($0.50$ threshold).
- **Decision Gate**: Escalated with reason: `Insufficient retrieval similarity (0.48 < 0.50)`.
- **Verdict**: **Safe Rejection**. Rather than inventing a PlayStation fix, the agent escalated to specialist support.

### Failure Mode 4: False Inferences in Zero-Shot Edge Cases
- **Example**: `ex_010`: *"Not receiving verification code on my registered mobile number."*
- **Agent Behavior**: The classifier predicted `subscription_billing` due to the word "number" and "verification", with confidence $0.16$ and margin $0.007$.
- **Decision Gate**: Escalated because `confidence < 0.22` and `margin < 0.05`.
- **Verdict**: **Resilient Escalation**. Even when the intent classifier misclassified the query, the multi-tiered confidence and margin gates successfully prevented an incorrect automated response from reaching the customer.

### Failure Mode 5: Account Hijack Disguised as Account Loss
- **Example**: `ex_014`: *"Someone removed my email address and I cannot log back in."*
- **Agent Behavior**: Classified as `login_account_access` with confidence $0.29$.
- **Risk Assessment**: Security regex flagged "someone removed my email" as an unauthorized account modification risk.
- **Decision Gate**: Escalated immediately via safety override before calling the LLM draft generator.
- **Verdict**: **Critical Safety Catch**. The rule-based escalation layer successfully overrode the ML classifier, guaranteeing customer account protection.

---

## Architectural Lessons & Interview Takeaways

1. **Defense in Depth**: Machine learning classifiers should never be the sole gatekeeper for enterprise support safety. Combining ML confidence scores with deterministic regex guards ensures $0\%$ unsafe auto-handles on critical legal and security queries.
2. **Explainability Over Opaque Autonomy**: An interviewer can inspect every single escalation trigger because decisions are logged as discrete, human-readable reason strings.
3. **Threshold Tuning Must Use Calibration Data Only**: Tuning thresholds on the calibration split ($N=100$) allowed us to establish strict $0\%$ unsafe auto-handle criteria prior to evaluating on the locked test partition.

# Decision Log

Keep 10-15 non-obvious decisions. Do not list trivial choices like "used Python."

Format:

| # | Decision | Alternatives considered | Why | Evidence / trade-off |
|---|---|---|---|---|

Suggested decisions to confirm or replace:

1. **Use a deterministic single-brand subsample instead of the full dataset.**
   - Reason: assignment encourages subsampling; faster iteration and reproducibility.

2. **Select the brand using data-quality and public-resolution criteria rather than raw tweet volume.**
   - Alternative: largest brand.
   - Trade-off: smaller but more learnable corpus can produce stronger evidence.

3. **Keep intent taxonomy to ~8-12 behaviorally distinct intents.**
   - Alternative: dozens of fine-grained labels.
   - Trade-off: better label consistency and enough examples per class.

4. **Separate 50-example calibration from 150-example locked test.**
   - Alternative: tune on the entire golden set.
   - Reason: prevents optimistic evaluation.

5. **Use macro-F1 instead of accuracy as the primary intent metric.**
   - Reason: class imbalance.

6. **Use TF-IDF + Logistic Regression as the simple baseline.**
   - Reason: strong, transparent, inexpensive baseline.

7. **Use sentence embeddings + linear classifier instead of transformer fine-tuning.**
   - Reason: faster, reproducible, easier to explain live.

8. **Use exact FAISS cosine retrieval instead of a hosted vector DB.**
   - Reason: corpus is small enough; removes infrastructure.

9. **Make the escalation decision an explicit rule/calibration layer, not a pure LLM decision.**
   - Reason: inspectable safety behavior.

10. **Treat low retrieval relevance as a reason to escalate.**
    - Reason: generation without strong precedent risks hallucination.

11. **Require generated replies to return historical evidence IDs.**
    - Reason: traceable grounding.

12. **Cache generation/judge outputs and provide a deterministic fast evaluation path.**
    - Reason: <15 minute reproduction and provider/API variability.

13. **Measure auto-handle acceptability together with coverage.**
    - Reason: avoids making a highly conservative system look artificially strong.

14. **Manually validate the LLM judge against human ratings.**
    - Reason: assignment explicitly requires evidence of agreement.

15. **Skip a UI until all required evaluation/report deliverables are complete.**
    - Reason: UI does not directly improve the proof of trustworthiness.

Replace any item that was not actually part of the final implementation.

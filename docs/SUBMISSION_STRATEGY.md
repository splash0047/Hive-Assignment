# Submission Strategy

## What reviewers should understand in the first 2 minutes

1. You understood the assignment is about **trust**, not merely generation.
2. You used real brand history as traceable evidence.
3. You created a real human-labelled golden set.
4. You compared against honest baselines.
5. Your system abstains when it lacks evidence.
6. You checked whether your LLM judge can be trusted.
7. You know exactly where the system fails.

## Recommended repository landing page order

1. one-sentence problem;
2. headline metric + coverage;
3. one architecture diagram;
4. result table vs baselines;
5. 15-minute reproduction command;
6. failure-analysis summary;
7. human-vs-judge agreement;
8. limitations;
9. deeper implementation details.

## Demo strategy for a live review

Prepare 5 examples:
- easy auto-handle;
- paraphrased common issue;
- low-evidence escalation;
- security/payment risky escalation;
- one known failure.

For each, show:
- intent/confidence;
- evidence retrieved;
- generated reply;
- route decision/reason.

This proves the system is inspectable.

## Live coding preparation

Be ready to modify:
- one escalation threshold;
- one intent label;
- one retrieval filter;
- one prompt constraint;
- one metric.

Know:
- why FAISS exact search is enough;
- why macro-F1 is used;
- why coverage accompanies precision;
- what prevents test leakage;
- why LLM judge agreement matters.

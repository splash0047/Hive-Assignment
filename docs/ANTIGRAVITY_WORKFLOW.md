# How to Use This Pack With Antigravity

## Session 1 - Architecture and scaffold

Paste or attach:
- `AGENTS.md`
- `docs/PRD.md`
- `docs/TECH_SPEC.md`
- `prompts/00_MASTER_PROMPT.md`

Then send:
`Execute prompts/01_SCAFFOLD_AND_DATA.md only. Do not proceed to the next phase.`

Review its output, tests, and brand profiler.

## Session 2 - Data freeze

Send:
`Execute prompts/02_BRAND_AND_CORPUS.md. Preserve all prior tests and update the decision log.`

## Session 3 - Taxonomy and labeling

Send:
`Execute prompts/03_INTENTS_AND_LABELING.md. Do not create the human labels for me; create the tooling and sampling only.`

You manually label the golden set.

## Session 4 - Baselines

After labels are frozen:
`Execute prompts/04_BASELINES.md and show baseline results before building the final model.`

## Session 5 - Final model

Run:
- prompt 05
- prompt 06

Do not accept a complex framework rewrite unless it provides a measurable benefit.

## Session 6 - Evaluation

Run prompt 07.

Manually score 40-60 selected outputs in the generated human-rating sheet.

Then run prompt 08.

## Session 7 - Packaging

Run:
- prompt 09
- prompt 10

## Rules when Antigravity gets stuck

Ask it to:
1. reproduce the failing command;
2. identify the smallest root cause;
3. fix only that cause;
4. add a regression test;
5. rerun all tests;
6. avoid architecture rewrites.

## Rules when Antigravity proposes extra features

Reject unless the feature improves one of:
- intent quality;
- retrieval relevance;
- safe routing;
- reply groundedness;
- judge reliability;
- reproducibility;
- report evidence.

A dashboard is optional. A stronger eval harness is not.

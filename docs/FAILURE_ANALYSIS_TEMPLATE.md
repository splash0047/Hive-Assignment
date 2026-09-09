# Failure Analysis

Populate from **locked-test failures**, not imagined examples.

## Failure selection method

1. sort all rejected/wrong examples by severity;
2. group similar errors;
3. choose the five most common or highest-risk modes;
4. include real tweet IDs/evidence IDs;
5. write a falsifiable root-cause hypothesis.

---

## Failure Mode 1 - <name>

**Frequency:**  
**Severity:**  
**Affected intents:**

### Example A
- tweet ID:
- customer:
- expected intent:
- predicted intent:
- expected route:
- actual route:
- generated reply:
- retrieved evidence:
- judge/human feedback:

### Hypothesis
...

### Proposed fix
...

### Trade-off
What other metric could get worse?

Repeat for 5 modes.

---

## Cross-cutting observations

- Are failures concentrated in rare intents?
- Do low-confidence predictions correspond to errors?
- Does retrieval score predict reply acceptability?
- Is the judge too lenient/strict in a pattern?
- Are most bad replies actually caused by bad retrieval rather than generation?
- Are escalations too conservative?

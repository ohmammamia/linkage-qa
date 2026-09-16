# Probabilistic Record Linkage with QA

Matching records to people when no shared identifier exists.

FEBRL benchmark, 5,000 + 5,000 records, 5,000 known links. No NHS data.
No code from NHS England repositories — see `docs/attribution.md`.

## Result

Without the near-unique identifier in the data:

| | precision | recall (end to end) | F1 |
|---|---|---|---|
| best deterministic rule | 1.0000 | 0.4962 | 0.6633 |
| Fellegi-Sunter, threshold only | 0.9793 | 0.9948 | 0.9870 |
| **Fellegi-Sunter + one-to-one assignment** | **1.0000** | **0.9948** | **0.9974** |

```bash
pip install -e ".[dev]" && python run_pipeline.py
```

## Why 0.99 is not the headline

It is a property of this benchmark. Zero non-matching pairs out of 106,000
agree on six or more fields — there are no hard negatives. Real data has
siblings at one address and date-of-birth collisions.

The honest summary is the degradation curve. Extra corruption injected, whole
pipeline re-run:

| extra corruption | blocking completeness | precision | recall |
|---|---|---|---|
| 0% | 0.995 | 0.9986 | 0.9944 |
| 10% | 0.955 | 0.9981 | 0.9382 |
| 20% | 0.892 | 0.9838 | 0.8610 |
| 30% | 0.772 | 0.9347 | 0.7334 |
| 40% | 0.645 | 0.9595 | 0.5312 |

`python scripts/degradation.py` reproduces this.

Precision holds, recall collapses, and recall tracks blocking completeness.
The binding constraint is candidate generation, not the model. This is why NHS
England use nine blocking rules including Soundex, against three here.

## Four findings

**Blocking sets the ceiling.** Union of three passes: 99.554% reduction, 99.5%
of true pairs kept. Single keys lose 11–53%. The 25 pairs lost here cap recall
before the model runs.

**Blocking breaks the u estimate.** A blocking-key field agrees within its
blocks by construction. Surname's weight: 0.14 bits estimated on candidates,
7.53 on random pairs. Non-key fields move under 5%.

**Clerical review has a floor.** 595 reviews (0.5% of pairs) cut errors from
131 to 29. Beyond that, errors stop at 25 — review only sees proposed pairs.

**The QA check found a defect in my own output, and fixing it was worth 105
false matches.** At the operating threshold, 98 left and 93 right records
appeared in multiple accepted matches — each pair scoring well alone while being
jointly impossible. Greedy one-to-one assignment by descending score removed
every false accept (precision 0.9793 → 1.0000) with no loss of recall. A
threshold sweep would never have found this; the structural check did.

## Reasoning

`docs/decision_log.md` — ten decisions, evidence, and what would reverse each.
`docs/limitations.md` — what this does not support.

## Limitations

Synthetic benchmark, no hard negatives. Conditional independence violated
(correlations to 0.896), reported not corrected. No domain preprocessing.
Binary comparison levels. Greedy rather than optimal assignment.

MIT licence.

# Methodology

## Shape

```
profiling → deterministic baselines → blocking → comparison
    → EM without labels → diagnostics → threshold sweep
    → review bands → downstream impact → QA → degradation
```

Each stage exists because the previous one raised a question.

## Method choices

**Fellegi-Sunter rather than a supervised classifier.** A classifier needs
labels, and production linkage has none. The unsupervised latent-class
formulation is the whole reason the method is usable, and it is what makes the
evaluation here transferable.

**Implemented rather than configured.** Splink and `recordlinkage` are the
right production choices. Writing the EM out makes the estimation step visible;
inside a library the u-estimation trap is a parameter one either sets correctly
or does not, with no signal either way.

**Multi-pass blocking, unioned.** One key fails wherever that field is
corrupted; independent passes fail on different records. Measured rather than
assumed: single keys retain 0.466 to 0.894 of true pairs, the union 0.995.

**Comparison by failure mode.** Tolerant for names, where typos dominate.
Exact for dates and postcodes, where a one-character difference is a genuine
difference and partial credit would let different people accumulate evidence.

**u from random pairs.** Random pairs from the full cross-product are
overwhelmingly non-matches, so their agreement rate estimates chance agreement
directly and is unaffected by blocking.

**Threshold as a curve.** There is no correct threshold, only a choice about
which error to prefer, and the two errors are not comparable. Reported as
counts so the trade-off is visible.

**Own tenth percentile, no external benchmark.** Where a floor is needed it
comes from the data's own distribution rather than an imported standard.

## Validation strategy

Three kinds, because each catches what the others miss.

1. **Unit tests on constructed cases** (20). Including EM parameter recovery:
   generate from the model with known m, u and prior, check they come back.
   Also what each function must *not* do — blocking must not reach records
   missing the key, exact fields must reject near-misses.
2. **End-to-end reproduction.** `run_pipeline.py` regenerates every documented
   figure, so documentation cannot drift from code.
3. **Stress testing.** The degradation sweep re-runs the whole pipeline under
   injected corruption, which is what converts a single benchmark number into a
   statement about the method.

## Reproducibility

Every constant lives in `configs/linkage.yaml` with its reasoning. Seeds are
fixed. Data ships with a package rather than being downloaded. CI runs the
tests and the pipeline on three Python versions.

## Responsible use

**Privacy.** No real personal data. FEBRL is synthetic.

**Misuse risk, specific.** The headline figure reads as a general statement
about linkage accuracy. It is not: it is a statement about a benchmark with no
hard negatives. Presented without the degradation curve it would support an
expectation no real linkage will meet. This is why the README leads with the
explanation rather than the number.

**Human oversight.** The review bands are the design point: automation handles
the confident ends and a person handles the middle, with the exchange rate
between staff time and error count made explicit.

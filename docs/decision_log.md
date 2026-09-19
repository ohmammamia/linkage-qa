# Decision log

Eleven decisions. Each: what, why, and what would reverse it.

---

**D-01 · Implemented Fellegi-Sunter rather than using Splink.**
The unsupervised estimation of m and u is the part that matters operationally
and is invisible from outside a library call. Production should use Splink.
Risk confirmed: my EM failed in a case a mature library would not — D-06.

**D-02 · Three blocking passes, unioned.**
Single keys retain 0.665–0.894 of true pairs — surname 0.665, postcode 0.844,
date of birth 0.894 — and the union retains 0.995 at 99.554% reduction.
Surname wins on reduction alone (0.99661) and retains least. Reverses if one
key becomes reliable enough to stand alone.

**D-03 · u estimated on random pairs, not on blocked candidates.**
A blocking key agrees within its blocks by construction. Surname: 0.14 bits vs
7.53. Non-key fields move under 5%, with one exception that is not a blocking
effect and is worth naming: soc_sec_id's u is estimated at exactly 0 on the
111,397 candidates, so `_clip` floors it at 1e-6 and its 19.80-bit weight is
set by the numerical guard rather than by the data. Estimating u on random
pairs gives 0.000167 and 12.42 bits — the honest number. A zero cell, not a
blocking artefact, and an argument for estimating u on a sample large enough to
observe the event. Honest limit: on this benchmark the correction does not
change peak F1, because soc_sec_id dominates. It changes where the threshold
sits.

**D-04 · Recall reported end to end.**
At the operating threshold: 0.9998 among candidates, 0.9948 end to end. The
difference is the 25 pairs blocking discarded, which cap end-to-end recall at
0.9950 whatever the model does. The flattering number is the one a pipeline
produces by default. `threshold_sweep` raises if the true total is not
supplied.

**D-05 · Near-unique identifier removed before the headline evaluation.**
soc_sec_id weighs 12 bits against 2–10 for everything else. With it, the result
is `if id_a == id_b` with extra steps. Without it: 0.9948 end-to-end recall at
the operating threshold, against 0.9122 for exact matching on the identifier
itself — because the identifier is corrupted in 9% of true pairs.

**D-06 · EM can converge to a degenerate solution; diagnostics detect it.**
Found by a test, not in review: two fields and a weak signal gave m = [0.17,
0.96] and a match prior of 0.295. My first diagnostic used mean m and missed
it; the fix was a more meaningful statistic — minimum m — not a looser
threshold.

**D-07 · Conditional independence violated, reported not corrected.**
Correlations to 0.896 (address fields move together). Correlated fields
double-count evidence and inflate apparent confidence. The correction changes
the model and belongs in its own decision.

**D-08 · Threshold published as a trade-off, with the floor shown.**
595 reviews cut errors 131 → 29. Beyond 6,385 reviews errors stop at 25, which
is what blocking discarded. Review capacity cannot reach that floor.

**D-09 · Downstream impact measured, magnitude flagged as simulated.**
At threshold 25 the cohort shrinks 3.5% and measured prevalence drifts 1.3%.
The mechanism is real and NHS England report finding it on their own data; the
magnitude here comes from a simulated outcome.

**D-10 · 0.99 explained rather than quoted.**
7.04 of 8 fields agree on true pairs; zero of 106,422 non-matches agree on six
or more.
No hard negatives. The degradation curve is what transfers: usable accuracy to
~10% extra corruption, and it is recall that fails, not precision.

**D-11 · One-to-one assignment after thresholding.**
`check_one_to_one` flagged 98 left and 93 right records in multiple accepted
matches. Greedy assignment by descending score, accepting a pair only if
neither record is already taken: precision 0.9793 → 1.0000, recall unchanged
at 0.9948, 105 false accepts removed. Greedy rather than Hungarian because the
scores are well separated, it is O(n log n), and every rejection has one named
cause. First flagged as outstanding; fixed once measured to be a 20-line change
with no downside.

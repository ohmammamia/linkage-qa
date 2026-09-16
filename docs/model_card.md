# Model Card — Probabilistic Record Linkage

Follows the NHS England model card format.

## Specification

| | |
|---|---|
| **Description** | Fellegi-Sunter probabilistic linkage, implemented from the model definition, fitted by expectation-maximisation without labels |
| **Model type** | Unsupervised latent class model over field agreement patterns |
| **Developed by** | Independent portfolio project |
| **Version** | 1.0 |

## Intended use

**Scope.** Demonstrating and evaluating probabilistic linkage on a public
benchmark, and a quality assurance framework for the production case where
ground truth is unavailable.

**Intended users.** Analysts building linked datasets for others to analyse.

**Out of scope.** Any use on real personal data without domain preprocessing
(see Limitations), and any operational deployment. Nothing here has been
validated on NHS data.

## Data

**Source.** FEBRL4, shipped with the `recordlinkage` package: 5,000 + 5,000
records with 5,000 known true links. Synthetic, with controlled corruption.

**Sensitive data.** None. No real personal data is used.

**Preprocessing.** Lowercasing and whitespace stripping only. Production
linkage needs substantially more — see Limitations.

**Split.** No train/test split, because no training occurs. Labels are used
only for evaluation.

## Methodology

**Justification.** Exact matching on demographics recovers 0.4962 of true
pairs; a single mistyped character breaks it. Probabilistic scoring weights
each field by both its agreement rate on true matches and its chance agreement
rate, giving a ranking that can be explained field by field.

**Algorithm.** Field agreement vectors → EM estimation of m, u and the match
prior → weights log2(m/u) → summed match score → threshold.

**Key parameter choice.** Chance agreement (u) is estimated on randomly drawn
pairs, not on blocked candidates. Estimating it on candidates collapses the
weight of any field used as a blocking key — surname drops from 7.53 to 0.14
bits.

**Alternatives considered.** Splink and `recordlinkage` both implement this
well and should be used in production. Implemented directly here so the
unsupervised estimation step is visible rather than configured.

## Evaluation and performance

**Process.** `run_pipeline.py` reproduces every figure. Labels evaluate only.

**Headline, without the near-unique identifier:**

| metric | value |
|---|---|
| precision | 0.9994 |
| recall (end to end) | 0.9928 |
| F1 | 0.9961 |
| best deterministic recall | 0.4962 |

**Recall is reported end to end**, counting true pairs discarded by blocking as
false negatives at every threshold.

**Degradation.** Under injected corruption, precision holds (0.999 → 0.954
across 0–40%) while recall falls (0.994 → 0.528). The binding constraint is
blocking, not the model.

**Known defect.** `check_one_to_one` flags 98 left and 93 right records in
multiple accepted matches at the operating threshold. One-to-one assignment is
outstanding.

## Ethical considerations

**Bias and fairness.** Linkage quality varies with naming conventions, address
stability and data completeness, all of which vary systematically between
populations. `check_subgroup_match_rates` exists for this and is the first
check to run on real data.

**Safety implications.** Records that cannot be linked are excluded from
downstream analysis. Groups less likely to be linkable are therefore
under-represented in any decision informed by the linked data, without any
signal that this has happened. Quantified in `downstream_impact`.

**False matches.** Fusing two people into one record attaches someone else's
history to a patient. In a healthcare setting this is a safety issue before it
is a statistical one, which is why thresholds are reported as counts of each
error type rather than as a single optimised value.

## Caveats and limitations

No hard negatives in the benchmark; conditional independence violated
(correlations to 0.896); no domain preprocessing; binary comparison levels;
missing values treated as disagreement; one-to-one assignment outstanding; no
evaluation against an incumbent system. Full list in `limitations.md`.

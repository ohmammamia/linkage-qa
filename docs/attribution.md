# Attribution and originality

## What NHS England publishes in this area

- **NHSE_probabilistic_linkage** —
  https://github.com/nhsengland/NHSE_probabilistic_linkage — a Splink-based
  linkage pipeline with blocking evaluation, training, prediction and clerical
  review comparison. Runs on internal NHS data; no data in the repository.
- **Quality Assurance Framework for Data Linkage** —
  https://github.com/nhsengland/quality-assurance-framework-for-data-linkage —
  a published QA framework covering preparation, triage, implementation and
  evaluation, with RAG ratings and a bias analysis tutorial.
- **NHS England Data Science portfolio** —
  https://nhsengland.github.io/datascience/ — the wider body of work this
  project is framed against.

## What was taken

**Nothing executable.** No code from either repository appears here, and the
pipeline files in `NHSE_probabilistic_linkage` were never opened — only the
archive listing was seen.

From the QA framework, two documentation pages were read (`evaluation.md`,
`triage.md`) and the tutorial data notebook was inspected. What was taken from
them is the *shape of the problem*: that a QA framework should cover
record-level checks, aggregate agreement patterns, bias detection and
monitoring, and that unmatched records differing from matched ones biases
downstream analysis. Those are stated positions in a published framework, cited
where used. The checks in `src/linkage_qa/qa/framework.py` are written here.

## Data

FEBRL4, shipped with the `recordlinkage` package: 5,000 + 5,000 records with
5,000 known true links. Synthetic with controlled corruption. Not NHS data, and
not claimed to be — NHS England's own pipeline runs on internal data that
cannot be published, so no public alternative exists.

## Deliberate divergences from the NHS England approach

These are choices, made for reasons, not attempts to look different.

1. **Fellegi-Sunter implemented directly rather than via Splink.** Their
   pipeline configures Splink. Writing the EM out demonstrates the model rather
   than its configuration, and makes the u-estimation trap in D-03 visible —
   inside a library it is a parameter one either sets correctly or does not.

2. **No supervised training step.** Their pipeline has training data selection
   notebooks. Here labels are never used for fitting, only for evaluation,
   because production linkage has no labels and an evaluation that trains on
   them measures nothing that transfers.

3. **Different evaluation emphasis.** Their repository evaluates blocking, runs
   metrics and compares against clerical review and MPS. This project adds
   three things not visible in their public material: end-to-end recall that
   counts blocking losses (D-04), performance with the near-unique identifier
   removed (D-05), and the irreducible error floor that clerical review cannot
   cross (D-08).

## What is original

- The measured effect of blocking on u estimation, and the 54-fold weight
  change it causes for surname (D-03).
- EM degeneracy diagnostics computable without labels, found by a failing test
  rather than by inspection (D-06).
- The demonstration that clerical review has a hard floor set by blocking, so
  buying more review capacity past a point buys nothing (D-08).
- The comparison against deterministic baselines with the unique identifier
  removed: 0.9904 recall against 0.416 (D-05).

## How to describe this work

Accurate: "An independent probabilistic linkage and QA project on the FEBRL
benchmark. Fellegi-Sunter implemented from the model definition with
unsupervised EM. Framed on the problem areas NHS England's published Quality
Assurance Framework for Data Linkage identifies; no code from their
repositories is used."

Not accurate: anything implying NHS data, NHS involvement, or that the QA
framework's structure was arrived at independently.

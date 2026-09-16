"""Field comparison.

Each field gets the comparison its failure mode deserves, not the one that is
most convenient:

- **Names** are compared with Jaro-Winkler. Name corruption is dominated by
  typos, transpositions and truncation, and Jaro-Winkler weights agreement at
  the start of a string more heavily, which matches how names are mistyped.
- **Dates and postcodes** are compared exactly. A date of birth one digit out is
  a different date, not a similar one; partial credit here would let genuinely
  different people accumulate evidence of being the same.
- **Identifiers** are compared exactly for the same reason.

The threshold for treating a string similarity as agreement is a modelling
choice with consequences, so it is a parameter with a stated default rather
than a number inside a function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import jellyfish
except ImportError:  # pragma: no cover
    jellyfish = None

#: Default similarity above which two strings are treated as agreeing.
#: 0.90 on Jaro-Winkler admits ordinary typos while excluding most different
#: names. Its effect is measured in the decision log rather than assumed.
STRING_AGREEMENT_THRESHOLD = 0.90


def jaro_winkler(a, b) -> float:
    """Similarity in [0, 1]; a missing value scores 0, never partial credit."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return jellyfish.jaro_winkler_similarity(str(a).lower().strip(), str(b).lower().strip())


def compare(left: pd.DataFrame, right: pd.DataFrame, *, exact_fields: list[str],
            fuzzy_fields: list[str],
            threshold: float = STRING_AGREEMENT_THRESHOLD) -> tuple[np.ndarray, list[str]]:
    """Binary agreement matrix for aligned record frames.

    Missing values count as disagreement. The alternative -- a third "not
    comparable" level -- is more correct and is noted as an extension; treating
    missing as disagreement is conservative, since it can only push a pair
    below a threshold, never above it.
    """
    fields = list(exact_fields) + list(fuzzy_fields)
    g = np.zeros((len(left), len(fields)))
    for j, f in enumerate(fields):
        if f in exact_fields:
            both = left[f].notna().values & right[f].notna().values
            same = (left[f].astype(str).str.lower().str.strip().values ==
                    right[f].astype(str).str.lower().str.strip().values)
            g[:, j] = (both & same).astype(float)
        else:
            g[:, j] = np.array([jaro_winkler(x, y)
                                for x, y in zip(left[f], right[f])]) >= threshold
    return g, fields


def chance_agreement(left: pd.DataFrame, right: pd.DataFrame, *, exact_fields: list[str],
                     fuzzy_fields: list[str], n_samples: int = 60_000,
                     seed: int = 0, threshold: float = STRING_AGREEMENT_THRESHOLD) -> np.ndarray:
    """Estimate u from randomly drawn pairs across the whole space.

    This exists because of a trap that is easy to miss and expensive to keep.
    Estimating u from the blocked candidate set is wrong for any field used as a
    blocking key: within a block that field agrees by construction, so its
    estimated chance-agreement rate approaches one and its weight collapses to
    nothing. In this project, estimating u on blocked candidates gave surname a
    weight of 0.14 bits against 7.53 when estimated correctly -- the model
    discarded a field carrying real evidence.

    Random pairs from the full cross-product are overwhelmingly non-matches, so
    their agreement rate estimates u directly and is unaffected by blocking.
    """
    rng = np.random.default_rng(seed)
    l = left.iloc[rng.integers(0, len(left), n_samples)].reset_index(drop=True)
    r = right.iloc[rng.integers(0, len(right), n_samples)].reset_index(drop=True)
    g, _ = compare(l, r, exact_fields=exact_fields, fuzzy_fields=fuzzy_fields,
                   threshold=threshold)
    return g.mean(axis=0)

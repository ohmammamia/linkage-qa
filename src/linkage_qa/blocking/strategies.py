"""Candidate generation.

Comparing every pair of two 5,000-record files is 25 million comparisons. At
NHS scale -- two files of a few million -- it is an infeasible number, so no
linkage runs without blocking. Blocking is therefore not an optimisation: it is
part of the method, and it is the only stage whose errors cannot be recovered
later. A true pair excluded here is invisible to the model, to the threshold and
to clerical review alike.

Two quantities describe a blocking rule, and one alone is misleading:

    reduction ratio  = 1 - (candidates / all possible pairs)
    pair completeness = true pairs retained / all true pairs

A rule can have an excellent reduction ratio and quietly discard half the
matches. Both are always reported together.
"""

from __future__ import annotations

import pandas as pd


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).apply(lambda r: "|".join(r.str.lower().str.strip()), axis=1)


def block_on(left: pd.DataFrame, right: pd.DataFrame, cols: list[str]) -> set[tuple]:
    """Candidate pairs agreeing exactly on ``cols``.

    Records missing any blocking column are dropped from this pass. That is a
    real cost, not a technicality: a record with no postcode cannot be reached
    by a postcode block at all, which is precisely why several passes are
    combined below rather than one being chosen.
    """
    a = left.dropna(subset=cols)
    b = right.dropna(subset=cols)
    m = (
        a.assign(_k=_key(a, cols)).reset_index()
        .merge(b.assign(_k=_key(b, cols)).reset_index(), on="_k", suffixes=("_l", "_r"))
    )
    left_col = next(c for c in m.columns if c.endswith("_l"))
    right_col = next(c for c in m.columns if c.endswith("_r"))
    return set(zip(m[left_col], m[right_col]))


def multi_pass(left: pd.DataFrame, right: pd.DataFrame,
               passes: list[list[str]]) -> set[tuple]:
    """Union of several blocking passes.

    A single key fails whenever that field is corrupted or missing in one of
    the two records. Independent passes fail on different records, so their
    union recovers pairs that any one pass would lose. The cost is more
    candidates; the benefit is measured, not assumed, by
    :func:`evaluate_blocking`.
    """
    out: set[tuple] = set()
    for cols in passes:
        out |= block_on(left, right, cols)
    return out


def evaluate_blocking(candidates: set[tuple], truth: set[tuple],
                      n_left: int, n_right: int, name: str = "") -> dict:
    """Reduction ratio and pair completeness together, never separately."""
    total = n_left * n_right
    retained = len(candidates & truth)
    return {
        "strategy": name,
        "n_candidates": len(candidates),
        "reduction_ratio": 1 - len(candidates) / total,
        "pair_completeness": retained / len(truth) if truth else float("nan"),
        "true_pairs_lost": len(truth) - retained,
    }

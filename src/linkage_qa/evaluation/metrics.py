"""Evaluation and threshold selection.

The recall trap
---------------
Recall measured against pairs that survived blocking flatters the model by
hiding the stage with unrecoverable losses. In this project the model reaches
recall 1.0000 inside the candidate set and 0.9950 end to end, because blocking
had already discarded 25 true pairs. Only the second number answers the
question a user asks, so :func:`threshold_sweep` requires the total number of
true pairs and refuses to compute recall without it.

Thresholds
----------
There is no correct threshold, only a choice about which error to prefer. A
false match fuses two people into one record: the resulting person has someone
else's history, and in a healthcare setting that is a safety issue rather than
a statistical one. A missed match splits one person into two: their history is
incomplete, cohorts shrink and the loss is usually silent. The two are not
symmetric and no single-number metric expresses the difference, which is why
the sweep reports counts rather than only rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_sweep(scores: np.ndarray, is_match: np.ndarray, *, n_true_total: int,
                    thresholds=None) -> pd.DataFrame:
    """Confusion counts and rates at each threshold, with end-to-end recall.

    ``n_true_total`` is every true pair that exists, including those blocking
    never proposed. Pairs lost to blocking are counted as false negatives at
    every threshold, because that is what they are.
    """
    if n_true_total < is_match.sum():
        raise ValueError("n_true_total is smaller than the true pairs among candidates")
    thresholds = thresholds if thresholds is not None else np.arange(-10, 41, 5)
    lost_to_blocking = int(n_true_total - is_match.sum())

    rows = []
    for t in thresholds:
        sel = scores >= t
        tp = int((sel & is_match).sum())
        fp = int((sel & ~is_match).sum())
        fn = n_true_total - tp
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / n_true_total
        rows.append({
            "threshold": float(t),
            "n_accepted": int(sel.sum()),
            "true_matches": tp,
            "false_matches": fp,
            "missed_matches": fn,
            "of_which_lost_to_blocking": lost_to_blocking,
            "precision": precision,
            "recall_end_to_end": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision and recall else np.nan,
        })
    return pd.DataFrame(rows)


def clerical_review_bands(scores: np.ndarray, is_match: np.ndarray, *,
                          n_true_total: int, bands: list[tuple[float, float]]) -> pd.DataFrame:
    """Automated errors against manual review workload for each band.

    Pairs above the upper bound are accepted automatically, below the lower
    bound rejected automatically, and the middle goes to a human. Widening the
    band buys accuracy with staff time, and the point of the table is to show
    the exchange rate rather than to pick a band.

    It also exposes a floor. Errors cannot fall below the number of true pairs
    blocking discarded, however wide the band, because clerical review only
    ever sees pairs that were proposed.
    """
    lost = int(n_true_total - is_match.sum())
    rows = []
    for lo, hi in bands:
        accept = scores >= hi
        reject = scores < lo
        review = ~accept & ~reject
        false_accepts = int((accept & ~is_match).sum())
        missed = int((reject & is_match).sum()) + lost
        rows.append({
            "lower": lo, "upper": hi,
            "auto_accept": int(accept.sum()),
            "auto_reject": int(reject.sum()),
            "to_review": int(review.sum()),
            "review_share": float(review.sum() / len(scores)),
            "false_accepts": false_accepts,
            "missed": missed,
            "automated_errors": false_accepts + missed,
            "irreducible_floor": lost,
        })
    return pd.DataFrame(rows)


def downstream_impact(scores: np.ndarray, is_match: np.ndarray, outcome: np.ndarray,
                      thresholds=None) -> pd.DataFrame:
    """How linkage errors distort a study built on the linked data.

    Linkage is never the deliverable. Something is estimated from the linked
    cohort, and this quantifies what the threshold does to that estimate:
    cohort size and outcome prevalence at each threshold against the truth.

    The bias arises only when the probability of linking correctly is related to
    the outcome -- when the people hardest to link are not like the people
    easiest to link. Where that holds, raising the threshold buys precision and
    pays for it in representativeness, and the study never sees the bill.
    """
    thresholds = thresholds if thresholds is not None else np.arange(0, 31, 5)
    true_idx = np.where(is_match)[0]
    true_n = len(true_idx)
    true_prev = float(outcome.mean())

    rows = []
    for t in thresholds:
        keep = scores[true_idx] >= t
        n = int(keep.sum())
        prev = float(outcome[keep].mean()) if n else np.nan
        rows.append({
            "threshold": float(t),
            "cohort_size": n,
            "cohort_bias": n / true_n - 1,
            "prevalence": prev,
            "prevalence_bias": prev / true_prev - 1 if true_prev else np.nan,
        })
    return pd.DataFrame(rows)


def one_to_one_assignment(pairs: list[tuple], scores: np.ndarray) -> np.ndarray:
    """Greedy one-to-one assignment by descending score.

    Independent thresholding lets one record accept several partners, each pair
    scoring well on its own while being jointly impossible where both sources
    hold each person once. Walking pairs from highest score down and accepting
    each only if neither record is already taken resolves this.

    Greedy is not globally optimal -- the Hungarian algorithm is -- but on
    well-separated scores the two agree almost everywhere, greedy is O(n log n)
    rather than O(n^3), and its decisions are auditable: each rejection has a
    single named cause, the higher-scoring pair that took the record first.

    Returns a boolean mask over ``pairs``.
    """
    order = np.argsort(-scores, kind="stable")
    taken_left, taken_right = set(), set()
    keep = np.zeros(len(pairs), dtype=bool)
    for i in order:
        l, r = pairs[i]
        if l in taken_left or r in taken_right:
            continue
        keep[i] = True
        taken_left.add(l)
        taken_right.add(r)
    return keep

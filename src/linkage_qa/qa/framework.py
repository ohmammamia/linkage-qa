"""Quality assurance checks that run without ground truth.

The framework exists because the evaluation above is impossible in production.
Labelled pairs are what a benchmark has and an operational linkage does not, so
every check here is computable from the linkage output alone and is designed to
detect the failure by its symptom rather than by comparison with truth.

Framed on the areas NHS England's published Quality Assurance Framework for
Data Linkage identifies -- record-level checks, aggregate agreement patterns,
bias detection and monitoring. The checks themselves are written here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QAResult:
    name: str
    passed: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        return f"[{'PASS' if self.passed else 'FLAG'}] {self.name}: {self.message}"


def check_match_rate(n_accepted: int, n_left: int, expected: float,
                     tolerance: float = 0.05) -> QAResult:
    """Match rate against expectation from prior runs or domain knowledge.

    The first thing that moves when anything breaks upstream -- a changed
    extract, a reformatted field, a new source system. It detects almost any
    failure and diagnoses none, which is why it is first and never last.
    """
    rate = n_accepted / n_left if n_left else np.nan
    ok = abs(rate - expected) <= tolerance
    return QAResult("match_rate", ok,
                    f"match rate {rate:.3f} against expected {expected:.3f} "
                    f"(tolerance {tolerance:.3f})",
                    {"rate": rate, "expected": expected})


def check_score_distribution(scores: np.ndarray, threshold: float,
                             max_share_near: float = 0.10, window: float = 2.0) -> QAResult:
    """Mass piled against the threshold.

    A healthy linkage is bimodal: confident matches and confident non-matches,
    with few pairs in between. A concentration at the cut means the decision is
    being made by the last decimal of a score for many pairs, and small changes
    in data or parameters will move large numbers of them across.
    """
    near = float(np.mean(np.abs(scores - threshold) <= window))
    return QAResult("score_distribution", near <= max_share_near,
                    f"{near:.1%} of pairs within {window} bits of the threshold "
                    f"(limit {max_share_near:.0%})",
                    {"share_near_threshold": near})


def check_field_agreement(gamma: np.ndarray, fields: list[str],
                          accepted: np.ndarray, min_agreement: float = 0.5) -> pd.DataFrame:
    """Agreement rate per field among accepted matches.

    A field agreeing on almost none of the accepted matches is contributing
    nothing, and the linkage is resting on fewer fields than its design assumes.
    That is fragile in a specific way: if one of the remaining fields degrades,
    there is no redundancy left.
    """
    rows = []
    for j, f in enumerate(fields):
        rate = float(gamma[accepted, j].mean()) if accepted.sum() else np.nan
        rows.append({"field": f, "agreement_among_accepted": rate,
                     "flag": rate < min_agreement})
    return pd.DataFrame(rows).sort_values("agreement_among_accepted")


def check_one_to_one(pairs: list[tuple], ) -> QAResult:
    """Records appearing in more than one accepted match.

    Where both sources should hold each person once, a record linked to several
    is a contradiction the scores cannot see: each pair may score well
    individually while being jointly impossible. This is the check that catches
    a systematic failure a threshold sweep will not.
    """
    if not pairs:
        return QAResult("one_to_one", True, "no accepted pairs", {})
    left = pd.Series([p[0] for p in pairs]).value_counts()
    right = pd.Series([p[1] for p in pairs]).value_counts()
    l_dup, r_dup = int((left > 1).sum()), int((right > 1).sum())
    return QAResult("one_to_one", l_dup == 0 and r_dup == 0,
                    f"{l_dup} left and {r_dup} right record(s) appear in multiple "
                    f"accepted matches",
                    {"left_duplicated": l_dup, "right_duplicated": r_dup})


def check_subgroup_match_rates(accepted: np.ndarray, subgroup: pd.Series,
                               max_spread: float = 0.05) -> QAResult:
    """Match rate by subgroup.

    The check that matters most for equity and the one most often skipped.
    Linkage quality varies with name spelling conventions, address stability and
    data completeness, all of which vary systematically between populations. A
    linkage that matches one group less often does not announce it: the
    under-linked simply appear less in every downstream analysis, and the study
    reads as if they were healthier or rarer.
    """
    rates = pd.Series(accepted).groupby(subgroup.values).mean()
    spread = float(rates.max() - rates.min()) if len(rates) > 1 else 0.0
    return QAResult("subgroup_match_rates", spread <= max_spread,
                    f"match rate ranges {rates.min():.3f} to {rates.max():.3f} "
                    f"across {len(rates)} subgroups (spread {spread:.3f})",
                    {"rates": rates.round(4).to_dict(), "spread": spread})

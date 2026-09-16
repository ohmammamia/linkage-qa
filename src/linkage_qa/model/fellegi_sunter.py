"""Fellegi-Sunter probabilistic record linkage, implemented directly.

Why implement it rather than call a library
-------------------------------------------
Mature implementations exist (Splink, recordlinkage) and in production one
should use them. This is written out because the project's purpose is to show
that linkage is understood rather than invoked, and because the part that
matters operationally is invisible from the outside of a library call: the
unsupervised estimation of agreement probabilities.

The model
---------
For a candidate pair, each field yields an agreement pattern. Under
Fellegi-Sunter the pair belongs to one of two latent classes, match (M) or
non-match (U), and each field f contributes a weight:

    w_f = log2( m_f / u_f )   when the field agrees
    w_f = log2( (1-m_f) / (1-u_f) )   when it disagrees

where m_f is the probability the field agrees given the pair is a true match,
and u_f the probability it agrees by chance given it is not. The total score is
the sum of field weights, which assumes fields are conditionally independent
given match status -- an assumption that is wrong here and is tested rather
than ignored (see :func:`dependence_check`).

The two probabilities do different work, and conflating them is the commonest
error in hand-built matching rules. A field can agree almost always on true
matches and still carry almost no evidence if it also agrees by chance: in this
data `state` agrees on 96% of true pairs but takes only eight values, so
agreement there is nearly uninformative. Conversely, date of birth agrees
slightly less often but is nearly unique, so its agreement is strong evidence.
A rule-based system weights by intuition; Fellegi-Sunter weights by both
quantities and gets this right.

Estimating m and u without labels
---------------------------------
The operational situation has no ground truth -- that is the whole difficulty.
:func:`fit_em` estimates m and u by expectation-maximisation over the
comparison vectors alone, treating match status as the latent variable. Labels,
where they exist, are used only to evaluate the result, never to fit it. That
separation is what makes the evaluation here meaningful rather than circular.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FellegiSunterModel:
    """Fitted agreement probabilities and the weights derived from them."""

    fields: list[str]
    m: np.ndarray                       # P(agree | match)
    u: np.ndarray                       # P(agree | non-match)
    lambda_: float                      # prior P(match) among candidate pairs
    n_iter: int = 0
    history: list[float] = field(default_factory=list)

    @property
    def agreement_weight(self) -> np.ndarray:
        return np.log2(self.m / self.u)

    @property
    def disagreement_weight(self) -> np.ndarray:
        return np.log2((1 - self.m) / (1 - self.u))

    def weight_table(self) -> pd.DataFrame:
        """Per-field m, u and weights -- the object to inspect before trusting a score.

        A field whose agreement weight is near zero contributes nothing however
        often it agrees. A field whose disagreement weight is large in magnitude
        is one where a mismatch is strong evidence against a link, which is
        usually a field that is rarely corrupted.
        """
        return pd.DataFrame({
            "field": self.fields,
            "m": self.m.round(4),
            "u": self.u.round(6),
            "agreement_weight": self.agreement_weight.round(2),
            "disagreement_weight": self.disagreement_weight.round(2),
        }).sort_values("agreement_weight", ascending=False)

    def diagnostics(self) -> dict:
        """Signs that EM has converged to a poor or mirrored solution.

        EM maximises likelihood, not correctness. With few fields, a low match
        rate, or one field that agrees often by chance, the likelihood surface
        has competing modes and the algorithm can settle on one where the
        "match" class is defined by whichever field is commonest rather than by
        genuine matches. The symptoms are recognisable without labels:

        - ``m`` below ``u`` on any field: the classes have swapped, and the
          fitted match class is the non-match class.
        - any ``m`` below 0.5: a field agreeing on fewer than half of the
          estimated matches is either extremely corrupted or, more often, a
          sign that the fitted class is not the match class. The minimum is
          used rather than the mean because a single collapsed field is exactly
          the symptom, and averaging hides it behind the healthy ones.
        - an implausibly high fitted match prior: candidate sets after blocking
          are mostly non-matches, so a prior near 0.5 usually means the split
          has found something other than match status.

        Checked and reported rather than corrected. The remedy is more fields or
        better initialisation, both of which change the model.
        """
        swapped = [f for f, mi, ui in zip(self.fields, self.m, self.u) if mi < ui]
        return {
            "n_iter": self.n_iter,
            "fields_with_m_below_u": swapped,
            "min_m": float(self.m.min()),
            "match_prior": float(self.lambda_),
            "suspect": bool(swapped) or self.m.min() < 0.5 or self.lambda_ > 0.3,
        }

    def score(self, gamma: np.ndarray) -> np.ndarray:
        """Match weight in bits for each comparison vector."""
        return gamma @ self.agreement_weight + (1 - gamma) @ self.disagreement_weight

    def posterior(self, gamma: np.ndarray) -> np.ndarray:
        """P(match | comparison vector), using the fitted prior."""
        w = self.score(gamma)
        odds = (self.lambda_ / (1 - self.lambda_)) * np.exp2(w)
        return odds / (1 + odds)


def _clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Keep probabilities away from 0 and 1.

    Without this a field that never disagrees on the estimated matches produces
    an infinite weight, and one such field silently decides every comparison.
    The bound is a numerical guard, not a modelling choice, and it is set far
    below any value the data supports.
    """
    return np.clip(p, eps, 1 - eps)


def fit_em(
    gamma: np.ndarray,
    fields: list[str],
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    init_m: float = 0.9,
    init_u: float = 0.1,
    seed: int = 0,
) -> FellegiSunterModel:
    """Estimate m, u and the match prior by expectation-maximisation.

    ``gamma`` is a binary matrix of agreement indicators, one row per candidate
    pair and one column per field.

    Initialisation matters and is not neutral. Starting with m high and u low
    encodes the only assumption the method needs: that the match class is the
    one where fields tend to agree. Starting symmetrically leaves the two
    classes exchangeable and the algorithm can converge to the mirror image,
    labelling matches as non-matches. The starting values are deliberately
    crude so they cannot be mistaken for prior knowledge of the answer.
    """
    n, k = gamma.shape
    m = np.full(k, init_m)
    u = np.full(k, init_u)
    lam = 0.5
    history: list[float] = []

    for it in range(1, max_iter + 1):
        # E step: responsibility of the match class for each pair.
        log_pm = gamma @ np.log(_clip(m)) + (1 - gamma) @ np.log(_clip(1 - m))
        log_pu = gamma @ np.log(_clip(u)) + (1 - gamma) @ np.log(_clip(1 - u))
        a = np.log(lam) + log_pm
        b = np.log(1 - lam) + log_pu
        mx = np.maximum(a, b)
        denom = mx + np.log(np.exp(a - mx) + np.exp(b - mx))
        g = np.exp(a - denom)                      # P(match | pair)

        loglik = float(denom.sum())
        history.append(loglik)

        # M step.
        gs = g.sum()
        new_m = _clip((g @ gamma) / max(gs, 1e-12))
        new_u = _clip(((1 - g) @ gamma) / max(n - gs, 1e-12))
        new_lam = float(np.clip(gs / n, 1e-9, 1 - 1e-9))

        shift = max(np.abs(new_m - m).max(), np.abs(new_u - u).max())
        m, u, lam = new_m, new_u, new_lam
        if shift < tol:
            break

    return FellegiSunterModel(fields, m, u, lam, n_iter=it, history=history)


def dependence_check(gamma: np.ndarray, fields: list[str]) -> pd.DataFrame:
    """Pairwise correlation between field agreements.

    Fellegi-Sunter assumes fields agree independently given match status.
    Address fields violate this obviously -- a record with the wrong street also
    has the wrong suburb -- and each correlated pair double-counts its evidence,
    pushing scores away from the middle and making the model look more certain
    than it is.

    Reported rather than corrected. The correction (dependency modelling, or
    collapsing correlated fields into one comparison) changes the model, and the
    decision to make that change belongs in the decision log with the
    correlations as its evidence.
    """
    c = np.corrcoef(gamma.T)
    rows = []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            rows.append({"field_a": fields[i], "field_b": fields[j],
                         "correlation": round(float(c[i, j]), 3)})
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)

"""Tests on small constructed cases with known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from linkage_qa.blocking.strategies import block_on, evaluate_blocking, multi_pass
from linkage_qa.evaluation.metrics import (
    clerical_review_bands, downstream_impact, one_to_one_assignment, threshold_sweep,
)
from linkage_qa.features.comparison import compare, jaro_winkler
from linkage_qa.model.fellegi_sunter import (
    FellegiSunterModel, dependence_check, fit_em,
)
from linkage_qa.qa.framework import (
    check_field_agreement, check_match_rate, check_one_to_one,
    check_score_distribution, check_subgroup_match_rates,
)


# --- Fellegi-Sunter ---------------------------------------------------------

def test_em_recovers_known_probabilities():
    """Generate from the model, recover its parameters."""
    rng = np.random.default_rng(0)
    n, lam = 40_000, 0.05
    m_true = np.array([0.95, 0.90, 0.80])
    u_true = np.array([0.01, 0.05, 0.20])
    is_match = rng.random(n) < lam
    g = np.where(is_match[:, None], rng.random((n, 3)) < m_true, rng.random((n, 3)) < u_true)
    fit = fit_em(g.astype(float), ["a", "b", "c"])
    assert fit.m == pytest.approx(m_true, abs=0.05)
    assert fit.u == pytest.approx(u_true, abs=0.02)
    assert fit.lambda_ == pytest.approx(lam, abs=0.02)


def test_weight_reflects_chance_agreement_not_just_match_agreement():
    """A field agreeing often on matches is weak if it also agrees by chance."""
    rng = np.random.default_rng(1)
    n = 30_000
    is_match = rng.random(n) < 0.05
    # all fields agree on 95% of matches; the last also agrees 50% by chance
    g = np.column_stack([
        np.where(is_match, rng.random(n) < 0.95, rng.random(n) < 0.01),
        np.where(is_match, rng.random(n) < 0.95, rng.random(n) < 0.01),
        np.where(is_match, rng.random(n) < 0.95, rng.random(n) < 0.01),
        np.where(is_match, rng.random(n) < 0.95, rng.random(n) < 0.50),
    ]).astype(float)
    fit = fit_em(g, ["d1", "d2", "d3", "common"])
    assert not fit.diagnostics()["suspect"]
    w = dict(zip(fit.fields, fit.agreement_weight))
    assert w["d1"] > 3 * w["common"]


def test_diagnostics_flag_a_degenerate_em_solution():
    """Two fields and a weak signal is where EM is known to fail.

    Documented rather than silently tolerated: the diagnostic must catch it.
    """
    rng = np.random.default_rng(1)
    n = 30_000
    is_match = rng.random(n) < 0.05
    g = np.column_stack([
        np.where(is_match, rng.random(n) < 0.95, rng.random(n) < 0.01),
        np.where(is_match, rng.random(n) < 0.95, rng.random(n) < 0.50),
    ]).astype(float)
    fit = fit_em(g, ["discriminating", "common"])
    assert fit.diagnostics()["suspect"]


def test_scores_separate_matches_from_non_matches():
    rng = np.random.default_rng(2)
    n = 20_000
    is_match = rng.random(n) < 0.05
    g = np.where(is_match[:, None], rng.random((n, 4)) < 0.9,
                 rng.random((n, 4)) < 0.05).astype(float)
    fit = fit_em(g, list("abcd"))
    s = fit.score(g)
    assert s[is_match].mean() > s[~is_match].mean() + 5


def test_posterior_is_a_probability():
    rng = np.random.default_rng(3)
    g = (rng.random((5_000, 3)) < 0.3).astype(float)
    fit = fit_em(g, list("abc"))
    p = fit.posterior(g)
    assert ((p >= 0) & (p <= 1)).all()


def test_posterior_survives_an_overwhelming_score():
    """2 ** w overflows past ~1024 bits, and inf / (1 + inf) is NaN."""
    model = FellegiSunterModel(["a", "b"], m=np.array([1 - 1e-12, 1 - 1e-12]),
                               u=np.array([1e-300, 1e-300]), lambda_=0.05)
    gamma = np.ones((1, 2))
    assert model.score(gamma)[0] > 1024          # the regime 2 ** w cannot reach
    p = model.posterior(gamma)
    assert np.isfinite(p).all() and p[0] == pytest.approx(1.0)


def test_diagnostics_suspect_is_a_plain_bool():
    """It is printed in the pipeline output, so numpy's repr leaks into docs."""
    fit = fit_em((np.random.default_rng(6).random((500, 3)) < 0.3).astype(float),
                 list("abc"))
    assert type(fit.diagnostics()["suspect"]) is bool


def test_dependence_check_finds_a_duplicated_field():
    rng = np.random.default_rng(4)
    a = (rng.random(2_000) < 0.4).astype(float)
    g = np.column_stack([a, a, (rng.random(2_000) < 0.4).astype(float)])
    d = dependence_check(g, ["x", "x_copy", "z"])
    assert d.iloc[0]["correlation"] == pytest.approx(1.0)


# --- blocking ---------------------------------------------------------------

def frames():
    left = pd.DataFrame({"pc": ["A1", "A1", "B2"], "dob": ["2000", "1990", "1980"]},
                        index=["l0", "l1", "l2"])
    right = pd.DataFrame({"pc": ["A1", "B2", None], "dob": ["2000", "1975", "1980"]},
                         index=["r0", "r1", "r2"])
    return left, right


def test_blocking_drops_records_missing_the_key():
    left, right = frames()
    pairs = block_on(left, right, ["pc"])
    assert all(p[1] != "r2" for p in pairs)


def test_multi_pass_recovers_what_one_pass_loses():
    left, right = frames()
    one = block_on(left, right, ["pc"])
    both = multi_pass(left, right, [["pc"], ["dob"]])
    assert ("l2", "r2") in both and ("l2", "r2") not in one


def test_blocking_evaluation_reports_completeness_and_loss():
    left, right = frames()
    truth = {("l0", "r0"), ("l2", "r2")}
    ev = evaluate_blocking(block_on(left, right, ["pc"]), truth, 3, 3, "pc")
    assert ev["pair_completeness"] == pytest.approx(0.5)
    assert ev["true_pairs_lost"] == 1


# --- comparison -------------------------------------------------------------

def test_missing_value_never_scores_as_agreement():
    assert jaro_winkler(None, "smith") == 0.0
    assert jaro_winkler("smith", np.nan) == 0.0


def test_exact_fields_reject_near_misses():
    left = pd.DataFrame({"dob": ["1990-01-01"], "name": ["jonathan"]})
    right = pd.DataFrame({"dob": ["1990-01-02"], "name": ["jonathon"]})
    g, fields = compare(left, right, exact_fields=["dob"], fuzzy_fields=["name"])
    assert g[0, fields.index("dob")] == 0.0     # one day out is a different date
    assert g[0, fields.index("name")] == 1.0    # one letter out is the same name


# --- evaluation -------------------------------------------------------------

def test_recall_counts_pairs_lost_to_blocking():
    scores = np.array([10.0, 10.0])
    is_match = np.array([True, True])
    swept = threshold_sweep(scores, is_match, n_true_total=4, thresholds=[0])
    assert swept.iloc[0]["recall_end_to_end"] == pytest.approx(0.5)
    assert swept.iloc[0]["missed_matches"] == 2


def test_sweep_rejects_an_impossible_total():
    with pytest.raises(ValueError):
        threshold_sweep(np.array([1.0, 2.0]), np.array([True, True]), n_true_total=1)


def test_review_band_cannot_beat_the_blocking_floor():
    scores = np.array([20.0, 1.0, -20.0])
    is_match = np.array([True, True, False])
    out = clerical_review_bands(scores, is_match, n_true_total=5,
                                bands=[(0, 10), (-30, 30)])
    assert (out["automated_errors"] >= out["irreducible_floor"]).all()
    assert out["irreducible_floor"].iloc[0] == 3


def test_f1_is_zero_not_undefined_when_a_threshold_accepts_only_false_matches():
    """Precision 0 is a result, not a missing value; NaN would hide it."""
    swept = threshold_sweep(np.array([10.0, 10.0]), np.array([False, False]),
                            n_true_total=2, thresholds=[0])
    assert swept.iloc[0]["precision"] == 0.0
    assert swept.iloc[0]["f1"] == 0.0


def test_downstream_impact_rejects_a_misaligned_outcome():
    """outcome is indexed over true pairs; a candidate-length array is a bug."""
    scores = np.array([30.0, 1.0, 5.0])
    is_match = np.array([True, False, True])
    with pytest.raises(ValueError, match="indexed over the true pairs"):
        downstream_impact(scores, is_match, np.zeros(3), thresholds=[0])


def test_downstream_impact_detects_outcome_related_dropout():
    scores = np.concatenate([np.full(50, 30.0), np.full(50, 1.0)])
    is_match = np.ones(100, dtype=bool)
    outcome = np.concatenate([np.zeros(50), np.ones(50)])   # low scorers carry the outcome
    out = downstream_impact(scores, is_match, outcome, thresholds=[0, 20])
    assert out.iloc[1]["prevalence"] < out.iloc[0]["prevalence"]
    assert out.iloc[1]["cohort_bias"] < 0


# --- QA ---------------------------------------------------------------------

def test_match_rate_flags_a_drop():
    assert not check_match_rate(500, 1000, expected=0.90, tolerance=0.05).passed
    assert check_match_rate(900, 1000, expected=0.90, tolerance=0.05).passed


def test_score_distribution_flags_pileup_at_the_threshold():
    piled = np.concatenate([np.full(900, 10.0), np.full(100, 40.0)])
    clean = np.concatenate([np.full(500, -30.0), np.full(500, 40.0)])
    assert not check_score_distribution(piled, threshold=10.0).passed
    assert check_score_distribution(clean, threshold=10.0).passed


def test_one_to_one_catches_a_record_matched_twice():
    assert not check_one_to_one([("l1", "r1"), ("l1", "r2")]).passed
    assert check_one_to_one([("l1", "r1"), ("l2", "r2")]).passed


def test_subgroup_check_flags_unequal_match_rates():
    accepted = np.array([True] * 50 + [False] * 50)
    grp = pd.Series(["a"] * 50 + ["b"] * 50)
    assert not check_subgroup_match_rates(accepted, grp).passed


def test_field_agreement_flags_a_dead_field():
    g = np.column_stack([np.ones(100), np.zeros(100)])
    acc = np.ones(100, dtype=bool)
    out = check_field_agreement(g, ["live", "dead"], acc)
    assert bool(out[out.field == "dead"]["flag"].iloc[0])
    assert not bool(out[out.field == "live"]["flag"].iloc[0])


def test_one_to_one_keeps_the_higher_scoring_partner():
    pairs = [("l1", "r1"), ("l1", "r2"), ("l2", "r2")]
    scores = np.array([10.0, 30.0, 20.0])
    keep = one_to_one_assignment(pairs, scores)
    assert keep.tolist() == [False, True, False]   # l1-r2 wins; l2-r2 blocked


def test_one_to_one_leaves_disjoint_pairs_alone():
    pairs = [("l1", "r1"), ("l2", "r2"), ("l3", "r3")]
    keep = one_to_one_assignment(pairs, np.array([5.0, 5.0, 5.0]))
    assert keep.all()

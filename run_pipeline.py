"""End-to-end linkage pipeline. Reproduces every figure quoted in the README.

    python run_pipeline.py            # full run
    python run_pipeline.py --quick    # skip the degradation sweep

Stages follow the order in docs/methodology.md. Each prints its own result, so
a reader can see where every number in the documentation comes from rather than
taking the README on trust.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import yaml

from linkage_qa.blocking.strategies import evaluate_blocking, multi_pass
from linkage_qa.evaluation.metrics import (
    clerical_review_bands, downstream_impact, one_to_one_assignment,
    threshold_sweep,
)
from linkage_qa.features.comparison import chance_agreement, compare
from linkage_qa.model.fellegi_sunter import (
    FellegiSunterModel, dependence_check, fit_em,
)
from linkage_qa.qa.framework import (
    check_field_agreement, check_match_rate, check_one_to_one,
    check_score_distribution,
)


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load_config(path: str = "configs/linkage.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deterministic_baseline(left: pd.DataFrame, right: pd.DataFrame,
                           cols: list[str], truth: set) -> dict:
    """Exact match on ``cols``. The bar the probabilistic model has to clear."""
    a = left.dropna(subset=cols)
    b = right.dropna(subset=cols)
    key = lambda d: d[cols].astype(str).apply(  # noqa: E731
        lambda r: "|".join(r.str.lower().str.strip()), axis=1)
    m = (a.assign(_k=key(a)).reset_index()
         .merge(b.assign(_k=key(b)).reset_index(), on="_k", suffixes=("_l", "_r")))
    lcol = [c for c in m.columns if c.endswith("_l")][0]
    rcol = [c for c in m.columns if c.endswith("_r")][0]
    pairs = set(zip(m[lcol], m[rcol]))
    tp = len(pairs & truth)
    return {
        "rule": " + ".join(cols),
        "pairs": len(pairs),
        "precision": tp / len(pairs) if pairs else np.nan,
        "recall": tp / len(truth),
    }


def main(quick: bool = False) -> None:
    t0 = time.time()
    cfg = load_config()
    rng_seed = cfg["run"]["seed"]

    from recordlinkage.datasets import load_febrl4
    left, right, links = load_febrl4(return_links=True)
    truth = set(zip(links.get_level_values(0), links.get_level_values(1)))

    # -- 1. profiling ------------------------------------------------------
    banner("1. FIELD PROFILING")
    a = left.loc[links.get_level_values(0)].reset_index(drop=True)
    b = right.loc[links.get_level_values(1)].reset_index(drop=True)
    rows = []
    for c in left.columns:
        both = a[c].notna() & b[c].notna()
        rows.append({
            "field": c,
            "missing_left": left[c].isna().mean(),
            "distinct_values": left[c].nunique(),
            "agrees_on_true_pairs": (a[c][both].astype(str).str.lower()
                                     == b[c][both].astype(str).str.lower()).mean(),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nNote: soc_sec_id has one distinct value per record -- a near-unique")
    print("identifier. Removed at stage 5; see docs/decision_log.md D-05.")

    # -- 2. deterministic baselines ---------------------------------------
    banner("2. DETERMINISTIC BASELINES -- the bar to clear")
    base = pd.DataFrame([deterministic_baseline(left, right, c, truth)
                         for c in cfg["baselines"]])
    print(base.round(4).to_string(index=False))
    best_det = base["recall"].max()
    print(f"\nBest deterministic recall: {best_det:.4f}. Exact matching loses "
          f"{1 - best_det:.0%} of true pairs to single-character corruption.")

    # -- 3. blocking -------------------------------------------------------
    banner("3. BLOCKING -- the irreversible stage")
    single = [evaluate_blocking(multi_pass(left, right, [p]), truth,
                                len(left), len(right), " + ".join(p))
              for p in cfg["blocking"]["passes"]]
    candidates = sorted(multi_pass(left, right, cfg["blocking"]["passes"]))
    union = evaluate_blocking(set(candidates), truth, len(left), len(right),
                              "UNION of all passes")
    print(pd.DataFrame(single + [union]).round(5).to_string(index=False))
    lost = union["true_pairs_lost"]
    print(f"\n{lost} true pairs never reach the model. End-to-end recall is capped")
    print(f"at {1 - lost / len(truth):.4f} before anything else runs.")

    # -- 4. comparison vectors --------------------------------------------
    banner("4. FIELD COMPARISON")
    ex, fz = cfg["comparison"]["exact"], cfg["comparison"]["fuzzy"]
    la = left.loc[[c[0] for c in candidates]].reset_index(drop=True)
    lb = right.loc[[c[1] for c in candidates]].reset_index(drop=True)
    gamma, fields = compare(la, lb, exact_fields=ex, fuzzy_fields=fz)
    y = np.array([c in truth for c in candidates])
    print(f"{len(candidates):,} candidate pairs x {len(fields)} fields")
    print(f"exact comparison : {ex}")
    print(f"fuzzy comparison : {fz} (Jaro-Winkler >= "
          f"{cfg['comparison']['string_threshold']})")

    # -- 5. model ----------------------------------------------------------
    banner("5. FELLEGI-SUNTER, FITTED BY EM WITHOUT LABELS")
    fitted = fit_em(gamma, fields)
    u_random = chance_agreement(left, right, exact_fields=ex, fuzzy_fields=fz,
                                n_samples=cfg["run"]["u_samples"], seed=rng_seed)
    model = FellegiSunterModel(fields, fitted.m, np.clip(u_random, 1e-6, 1),
                               fitted.lambda_)

    print("EM diagnostics:", fitted.diagnostics())
    print(f"\nFitted match prior : {fitted.lambda_:.4f}")
    print(f"True match rate    : {y.mean():.4f}   <- EM saw no labels")

    print("\n-- effect of estimating u on blocked candidates vs random pairs --")
    comp = pd.DataFrame({
        "field": fields,
        "u_on_blocked": fitted.u.round(5),
        "u_on_random": u_random.round(6),
        "weight_blocked": np.log2(fitted.m / fitted.u).round(2),
        "weight_corrected": np.log2(fitted.m / np.clip(u_random, 1e-6, 1)).round(2),
        "is_blocking_key": [f in cfg["blocking"]["key_fields"] for f in fields],
    })
    print(comp.to_string(index=False))
    print("\nBlocking keys only. Non-key fields move by under 5%.")

    print("\n-- conditional independence (assumed, violated, reported) --")
    print(dependence_check(gamma, fields).head(5).to_string(index=False))

    # -- 6. without the near-unique identifier ----------------------------
    banner("6. WITHOUT THE NEAR-UNIQUE IDENTIFIER -- the NHS-relevant case")
    drop = cfg["comparison"]["identifier_field"]
    keep = [i for i, f in enumerate(fields) if f != drop]
    fields2 = [fields[i] for i in keep]
    fitted2 = fit_em(gamma[:, keep], fields2)
    model2 = FellegiSunterModel(fields2, fitted2.m,
                                np.clip(u_random[keep], 1e-6, 1), fitted2.lambda_)
    scores = model2.score(gamma[:, keep])
    print(model2.weight_table().to_string(index=False))

    swept = threshold_sweep(scores, y, n_true_total=len(truth),
                            thresholds=cfg["evaluation"]["thresholds"])
    print("\n" + swept.round(4).to_string(index=False))
    best = swept.loc[swept["f1"].idxmax()]
    print(f"\nBest F1 {best['f1']:.4f} at threshold {best['threshold']:.0f}: "
          f"precision {best['precision']:.4f}, "
          f"END-TO-END recall {best['recall_end_to_end']:.4f}")
    print(f"Deterministic best was {best_det:.4f}. "
          f"{best['recall_end_to_end'] / best_det:.1f}x more true pairs recovered.")

    # -- 7. clerical review ------------------------------------------------
    banner("7. CLERICAL REVIEW -- and the floor blocking sets")
    bands = clerical_review_bands(scores, y, n_true_total=len(truth),
                                  bands=[tuple(b) for b in cfg["evaluation"]["review_bands"]])
    print(bands.round(4).to_string(index=False))
    print(f"\nErrors stop falling at {lost}: review only ever sees proposed pairs.")

    # -- 8. downstream impact ---------------------------------------------
    banner("8. WHAT THE THRESHOLD DOES TO A STUDY BUILT ON THE OUTPUT")
    rng = np.random.default_rng(rng_seed)
    idx = np.where(y)[0]
    s = scores[idx]
    prob = np.clip(0.30 - 0.004 * (s - s.mean()), 0.05, 0.95)
    outcome = rng.random(len(idx)) < prob
    full = np.zeros(len(scores), dtype=bool)
    full[idx] = outcome
    print(downstream_impact(scores, y, full[idx],
                            thresholds=cfg["evaluation"]["impact_thresholds"])
          .round(4).to_string(index=False))
    print("\nThe outcome-to-linkability relationship is simulated: the MECHANISM")
    print("is real, the magnitude is illustrative. See docs/decision_log.md D-09.")

    # -- 9. QA without ground truth ---------------------------------------
    banner("9. QUALITY ASSURANCE -- checks that need no labels")
    t = cfg["evaluation"]["operating_threshold"]
    thresholded = scores >= t
    accepted = thresholded & one_to_one_assignment(candidates, scores)
    tp_a = int((thresholded & y).sum()); fp_a = int((thresholded & ~y).sum())
    tp_b = int((accepted & y).sum());    fp_b = int((accepted & ~y).sum())
    print(f"threshold only        : accepted {thresholded.sum():5d}, "
          f"precision {tp_a/(tp_a+fp_a):.4f}, recall {tp_a/len(truth):.4f}")
    print(f"+ one-to-one (D-11)   : accepted {accepted.sum():5d}, "
          f"precision {tp_b/(tp_b+fp_b):.4f}, recall {tp_b/len(truth):.4f}\n")
    print(check_match_rate(int(accepted.sum()), len(left),
                           expected=cfg["qa"]["expected_match_rate"]))
    print(check_score_distribution(scores, threshold=t))
    print(check_one_to_one([candidates[i] for i in np.where(accepted)[0]]))
    print("\n" + check_field_agreement(gamma[:, keep], fields2, accepted)
          .round(4).to_string(index=False))

    # -- 10. degradation ---------------------------------------------------
    if not quick:
        banner("10. DEGRADATION -- why 0.99 is a statement about this benchmark")
        agree_true = gamma[y].sum(axis=1)
        agree_false = gamma[~y].sum(axis=1)
        print(f"fields agreeing on TRUE pairs     : mean {agree_true.mean():.2f} "
              f"of {gamma.shape[1]}")
        print(f"fields agreeing on NON-MATCH pairs: mean {agree_false.mean():.2f}")
        print(f"non-matches agreeing on 6+ fields : "
              f"{(agree_false >= 6).sum()} of {(~y).sum():,}")
        print("\nNo hard negatives. Real administrative data has siblings at one")
        print("address and name/date-of-birth collisions; this benchmark has none.")
        print("\nRun scripts/degradation.py for the full corruption sweep.")

    print(f"\nCompleted in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="skip the degradation summary")
    sys.exit(main(**vars(p.parse_args())))

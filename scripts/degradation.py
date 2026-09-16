"""Degradation sweep: how the pipeline behaves as data quality falls.

Injects additional corruption into both files at increasing rates and re-runs
the whole pipeline -- blocking, comparison, EM, threshold selection. This is
what converts a single benchmark number into a statement about the method.

    python scripts/degradation.py

Takes several minutes. Reproduces the table in the README.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from recordlinkage.datasets import load_febrl4  # noqa: E402

from linkage_qa.blocking.strategies import multi_pass  # noqa: E402
from linkage_qa.features.comparison import chance_agreement, compare  # noqa: E402
from linkage_qa.model.fellegi_sunter import FellegiSunterModel, fit_em  # noqa: E402

CORRUPTION_RATES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)


def corrupt(df: pd.DataFrame, fields: list[str], rate: float,
            rng: np.random.Generator) -> pd.DataFrame:
    """Three failure modes in equal proportion.

    A typo (single character replaced), a blanking (value lost), and a
    substitution (a different real value from the same column). Real systems
    produce all three; modelling only typos would understate how badly blocking
    degrades, since blanking and substitution both break an exact-match key
    outright.
    """
    out = df.copy()
    for col in fields:
        values = out[col].astype(object).values.copy()
        hit = rng.random(len(values)) < rate
        pool = df[col].dropna()
        for i in np.where(hit)[0]:
            value = values[i]
            if pd.isna(value):
                continue
            value = str(value)
            mode = rng.integers(0, 3)
            if mode == 0 and len(value) > 1:
                pos = rng.integers(0, len(value))
                values[i] = value[:pos] + chr(rng.integers(97, 123)) + value[pos + 1:]
            elif mode == 1:
                values[i] = np.nan
            else:
                values[i] = str(pool.iloc[int(rng.integers(0, len(pool)))])
        out[col] = values
    return out


def main() -> None:
    cfg = yaml.safe_load(open("configs/linkage.yaml"))
    exact = [f for f in cfg["comparison"]["exact"]
             if f != cfg["comparison"]["identifier_field"]]
    fuzzy = cfg["comparison"]["fuzzy"]
    fields = exact + fuzzy
    passes = cfg["blocking"]["passes"]
    rng = np.random.default_rng(cfg["run"]["seed"])

    left, right, links = load_febrl4(return_links=True)
    truth = set(zip(links.get_level_values(0), links.get_level_values(1)))

    print(f"{'corruption':>11} {'candidates':>11} {'blocking':>9} "
          f"{'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8}")

    for rate in CORRUPTION_RATES:
        a, b = corrupt(left, fields, rate, rng), corrupt(right, fields, rate, rng)
        candidates = sorted(multi_pass(a, b, passes))
        if not candidates:
            print(f"{rate:>10.0%}  blocking produced no candidates")
            continue

        la = a.loc[[c[0] for c in candidates]].reset_index(drop=True)
        lb = b.loc[[c[1] for c in candidates]].reset_index(drop=True)
        gamma, _ = compare(la, lb, exact_fields=exact, fuzzy_fields=fuzzy)
        y = np.array([c in truth for c in candidates])

        u = chance_agreement(a, b, exact_fields=exact, fuzzy_fields=fuzzy,
                             n_samples=40_000, seed=cfg["run"]["seed"])
        fitted = fit_em(gamma, fields)
        model = FellegiSunterModel(fields, fitted.m, np.clip(u, 1e-6, 1), fitted.lambda_)
        scores = model.score(gamma)

        best = None
        for t in np.arange(-10, 30, 1.0):
            sel = scores >= t
            tp = int((sel & y).sum())
            fp = int((sel & ~y).sum())
            p = tp / max(tp + fp, 1)
            r = tp / len(truth)                    # end to end, not among candidates
            f1 = 2 * p * r / max(p + r, 1e-9)
            if best is None or f1 > best[3]:
                best = (t, p, r, f1)

        print(f"{rate:>10.0%} {len(candidates):>11,} "
              f"{y.sum() / len(truth):>9.3f} {best[0]:>10.0f} "
              f"{best[1]:>10.4f} {best[2]:>8.4f} {best[3]:>8.4f}")


if __name__ == "__main__":
    main()

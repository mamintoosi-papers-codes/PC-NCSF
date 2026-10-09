"""Read-only smoke test for the hybrid clustering stage (report §8).

Creates NO files.  Verifies, per tier:
  * python/numpy/pandas/scipy/sklearn versions importable;
  * SCOP/<tier>/data.csv schema (theta, tau, res_seq, pdb_id, category columns);
  * checkpoint exists at the path TIER_RUNS expects;
  * density cache loads, shape matches the dropna/factorize protein count;
  * B1 and B2 feature vectors build (shapes [n, 8] and [n, 6]) and no protein
    loses its feature vector (every protein has >= 5 valid sequential pairs);
  * D_model~D_stats and D_model~D_seq Spearman correlations fall in the ranges
    already measured in §9 check (b) of the report — a cheap drift check.

Usage:  python scripts/check_hybrid_env.py --grid 100
Exit 0 if everything passes; nonzero otherwise.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy, scipy.cluster.hierarchy
import sklearn, sklearn.metrics
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from scop_clustering import TIER_RUNS, ROOT, hellinger_matrix
    from clustering_hybrid import (ALPHA_GRID, b1_features, b2_features,
                                   euclidean_matrix, hybrid_matrix,
                                   load_features_data, median_normalise, zscore)
except ImportError as e:
    print(f"ImportError: {e}")
    print("NOTE: scripts/scop_clustering.py imports torch and zuko at module level,")
    print("so the env used here must contain both, even though this smoke test")
    print("never calls the model. Pick an env that has them (e.g. the audit's 'pth'")
    print("env) via HYBRID_PYTHON (bat) or PYTHON=... (bash), then rerun.")
    sys.exit(2)

EXPECTED_SLOPE_RANGES = {
    "D_model~D_stats": (0.45, 0.80),
    "D_model~D_seq": (0.50, 0.80),
}


def main():
    print("python:", sys.version.split()[0])
    print("numpy:", np.__version__, "| pandas:", pd.__version__,
          "| scipy:", scipy.__version__, "| sklearn:", sklearn.__version__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=100)
    args, _ = ap.parse_known_args()

    problems = []
    for tier, ckpt_rel in TIER_RUNS.items():
        print(f"\n[{tier}]")
        ckpt = os.path.join(ROOT, ckpt_rel)
        df = pd.read_csv(os.path.join(ROOT, "SCOP", tier, "data.csv"), nrows=5)
        missing = {"theta", "tau", "res_seq", "pdb_id", "category"} - set(df.columns)
        if missing:
            problems.append(f"{tier}: data.csv missing columns {sorted(missing)}")
            continue
        if not os.path.exists(ckpt):
            problems.append(f"{tier}: checkpoint missing {ckpt}")
        cache = os.path.join(ROOT, "results", "clustering", f"masses_{tier}_g{args.grid}.npy")
        if not os.path.exists(cache):
            problems.append(f"{tier}: density cache missing {cache}")
            continue

        _, feats, labels, proteins = load_features_data(tier)
        n = feats["n_proteins"]
        masses = np.load(cache)
        if masses.shape[0] != n:
            problems.append(f"{tier}: cache rows {masses.shape[0]} != proteins {n}")
            continue

        f = b1_features(feats["theta"], feats["tau"], feats["pdb_idx"], n)
        g, valid_counts = b2_features(feats["theta"], feats["tau"],
                                      feats["res_seq"], feats["pdb_idx"], n)
        assert f.shape == (n, 8), f"{tier}: B1 shape {f.shape}"
        assert g.shape == (n, 6), f"{tier}: B2 shape {g.shape}"
        n_usable = int(np.sum(valid_counts >= 5))
        print(f"  cache ok {masses.shape}, proteins={n}, labels k={len(np.unique(labels))}")
        print(f"  B1 [n,8] ok; B2 [n,6] ok; proteins with >=5 pairs: {n_usable}/{n}")
        if n_usable != n:
            problems.append(f"{tier}: only {n_usable}/{n} proteins have >=5 valid pairs")

        D0 = median_normalise(hellinger_matrix(masses))
        D1 = median_normalise(euclidean_matrix(zscore(f)))
        D2 = median_normalise(euclidean_matrix(zscore(g)))
        iu = np.triu_indices(n, 1)
        r1 = spearmanr(D0[iu], D1[iu]).statistic
        r2 = spearmanr(D0[iu], D2[iu]).statistic
        print(f"  Spearman D_model~D_stats={r1:.3f} (expected {EXPECTED_SLOPE_RANGES['D_model~D_stats']}), "
              f"D_model~D_seq={r2:.3f} (expected {EXPECTED_SLOPE_RANGES['D_model~D_seq']})")
        lo, hi = EXPECTED_SLOPE_RANGES["D_model~D_stats"]
        if not (lo <= r1 <= hi):
            problems.append(f"{tier}: D_model~D_stats drift {r1:.3f} outside {lo}-{hi}")
        lo, hi = EXPECTED_SLOPE_RANGES["D_model~D_seq"]
        if not (lo <= r2 <= hi):
            problems.append(f"{tier}: D_model~D_seq drift {r2:.3f} outside {lo}-{hi}")

    # alpha grid sanity
    assert tuple(ALPHA_GRID) == (0.0, 0.25, 0.5, 0.75, 1.0), "alpha grid drifted"
    print("\nalpha grid:", ALPHA_GRID, "(pre-registered, unchanged)")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nSMOKE TEST PASSED (no files written)")


if __name__ == "__main__":
    main()

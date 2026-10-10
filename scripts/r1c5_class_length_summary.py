"""Per-tier SCOP category codes, class sizes, and per-class length summary.

Records the label hierarchy used in the R1-C5 revision and quantifies the
length<->label association per class (median valid consecutive-pair count and
median residue count), from SCOP/<tier>/data.csv with the loader's dropna rule.

Run from repository root:  python scripts/r1c5_class_length_summary.py
Output: results/clustering/r1c5_class_length_summary.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scop_clustering import ROOT, TIER_RUNS


def main():
    rows = []
    for tier in TIER_RUNS:
        df = pd.read_csv(os.path.join(ROOT, "SCOP", tier, "data.csv"))
        df = df.dropna(subset=["theta", "tau"]).reset_index(drop=True)
        pdb = df.pdb_id.astype(str)
        nres = pdb.value_counts()
        dr = np.diff(df.res_seq.to_numpy(float))
        same = (pdb.to_numpy()[1:] == pdb.to_numpy()[:-1])
        good = same & np.isfinite(dr) & (dr == 1)
        nvalid = pd.Series(np.bincount(
            pd.factorize(pdb, sort=False)[0][:-1][good],
            minlength=pdb.nunique()).astype(float), index=pd.unique(pdb))

        # majority category per protein
        lab = df.groupby(pdb)["category"].agg(lambda s: s.value_counts().idxmax())
        summary = lab.to_frame().join(nvalid.rename("valid_pairs")).join(
            nres.rename("residues"))
        print(f"\n=== {tier} ===")
        print(f"categories ({summary['category'].nunique()}): "
              f"{sorted(summary['category'].unique())}")
        for cat, grp in summary.groupby("category"):
            print(f"  {cat:8s} n={len(grp):5d}  valid_pairs median="
                  f"{grp['valid_pairs'].median():7.0f}  "
                  f"residues median={grp['residues'].median():7.0f}")
            rows.append(dict(tier=tier, category=cat, n_proteins=len(grp),
                             valid_pairs_median=grp["valid_pairs"].median(),
                             residues_median=grp["residues"].median()))
        # eta^2 of log1p(valid_pairs) on category (one-way ANOVA R^2)
        x = np.log1p(summary["valid_pairs"].to_numpy())
        y = pd.factorize(summary["category"])[0]
        grand = x.mean()
        ss_b = sum(len(x[y == c]) * (x[y == c].mean() - grand) ** 2
                   for c in np.unique(y))
        ss_t = ((x - grand) ** 2).sum()
        print(f"  eta^2(log1p valid_pairs ~ category) = {ss_b / ss_t:.3f}")
        rows.append(dict(tier=tier, category="ETA2_LOGPAIRS", n_proteins=len(summary),
                         valid_pairs_median=np.nan, residues_median=ss_b / ss_t))

    out = os.path.join(ROOT, "results", "clustering", "r1c5_class_length_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()

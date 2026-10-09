"""Local class-signal check: k-NN retrieval on the protein-density distance matrix.

ARI measures GLOBAL cluster structure. A representation can still carry class
information that is only locally visible (same-class proteins are near each
other even though no 4-way cut of the tree aligns with the labels).

For each protein we take its k nearest neighbours in the distance matrix and
predict its SCOP category by majority vote (chance = 1/n_classes).

Usage:
  /data/python-envs/pytorch/bin/python scripts/clustering_knn.py --tier all
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scop_clustering import TIER_RUNS, ROOT, hellinger_matrix, load_proteins


def load_labels(tier):
    _, _, proteins, _ = load_proteins(tier)
    df = pd.read_csv(os.path.join(ROOT, "SCOP", tier, "data.csv"))
    df = df.dropna(subset=["theta", "tau"])
    order = {p: i for i, p in enumerate(proteins)}
    first = df.assign(_i=df.pdb_id.map(order)).groupby("_i", sort=True).first()
    return pd.factorize(first["category"])[0]


def knn_accuracy(D, labels, k=5):
    n = len(labels)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    pred = np.empty(n, dtype=int)
    for i in range(n):
        votes = labels[nn[i]]
        pred[i] = np.bincount(votes).argmax()
    return (pred == labels).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="challenging", choices=list(TIER_RUNS) + ["all"])
    ap.add_argument("--grid", type=int, default=100)
    args = ap.parse_args()
    tiers = list(TIER_RUNS) if args.tier == "all" else [args.tier]

    rows = []
    for tier in tiers:
        labels = load_labels(tier)
        n = len(labels)
        n_cls = len(np.unique(labels))
        chance = 1.0 / n_cls
        cache = os.path.join(ROOT, "results", "clustering", f"masses_{tier}_g{args.grid}.npy")
        masses = np.load(cache).reshape(n, -1)
        D = hellinger_matrix(masses)
        accs = {k: knn_accuracy(D.copy(), labels, k) for k in (1, 5, 10)}
        # permutation control for the 5-NN number
        rng = np.random.default_rng(0)
        perm_accs = [knn_accuracy(D.copy(), rng.permutation(labels), 5) for _ in range(3)]
        print(f"{tier:13s} n={n} chance={chance:.3f} | "
              + " ".join(f"1NN={accs[1]:.3f}" if k == 1 else f"{k}NN={accs[k]:.3f}"
                         for k in (1, 5, 10))
              + f" | shuffled-5NN={np.mean(perm_accs):.3f}")
        rows.append({"tier": tier, "n": n, "chance": round(chance, 3),
                     "1NN": round(accs[1], 4), "5NN": round(accs[5], 4),
                     "10NN": round(accs[10], 4),
                     "5NN_shuffled": round(float(np.mean(perm_accs)), 4)})
    out = os.path.join(ROOT, "results", "clustering", "knn_retrieval.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    sys.stdout.reconfigure(line_buffering=True)
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()

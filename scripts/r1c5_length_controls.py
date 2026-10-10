"""R1-C5 length-confound controls for the SCOP density clustering experiment.

Question: how much of the apparent SCOP-label signal in per-protein
representations (PC-NCSF densities, empirical histograms) is already carried by
protein length (valid consecutive residue-pair count) alone?

Per tier, on the SAME protein subset (ambiguous pdb_ids excluded, matching
results/clustering/audit2026_pcncsf_g100.csv):

  1. Ward (k = #categories) ARI/NMI for:
       - PC-NCSF Hellinger distances   (from results/clustering/masses_*_g100.npy)
       - empirical histogram Hellinger (reproduces variants_*.csv protocol)
       - length only: |Delta log1p(valid_pair_count)|
  2. k-NN retrieval (majority vote, self excluded, k = 1 and 5) in each space,
     with an imbalance-preserving GLOBAL label-shuffle null (B shuffles).
  3. Length-STRATIFIED label-shuffle null (shuffles within quintiles of
     log1p(valid_pair_count)): how much of the k-NN accuracy / Ward ARI is
     explained by the coarse length<->label association alone.
  4. Mantel-style Spearman correlation between each representation's condensed
     distance vector and the length distance vector (protein-level permutation
     p-value).

Cross-checks printed at runtime against committed artifacts:
  - PC-NCSF ward ARI  vs results/clustering/audit2026_pcncsf_g100.csv
  - histogram ward ARI vs results/clustering/variants_<tier>.csv (all-protein)
  - length ward ARI vs results/clustering_hybrid_corrected/hybrid_corrected_all_g100.csv (all-protein)
  - model 1NN/5NN vs results/clustering/knn_retrieval.csv (all-protein)

Run from repository root:
  python scripts/r1c5_length_controls.py --grid 100 --n-shuffle 1000 --n-mantel 200

Seed: 20261010 (single RNG for all random draws, drawn in fixed order).
Outputs:
  results/clustering/r1c5_length_controls_g100.csv
  results/clustering/r1c5_mantel_g100.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scop_clustering import ROOT, TIER_RUNS, hellinger_matrix, load_proteins

SEED = 20261010


# --------------------------------------------------------------------- data
def load_tier(tier, grid):
    """Loader-exact protein order, majority labels, ambiguous mask, counts."""
    angles, cond_idx, proteins, lab = load_proteins(tier)  # loader-exact order
    proteins = [str(p) for p in proteins]
    df = pd.read_csv(os.path.join(ROOT, "SCOP", tier, "data.csv"))
    df = df.dropna(subset=["theta", "tau"]).reset_index(drop=True)
    assert list(pd.unique(df.pdb_id.astype(str))) == proteins, "protein order mismatch"

    # ambiguous pdb_ids: more than one category among their rows
    ncat = df.groupby(df.pdb_id.astype(str))["category"].nunique()
    ambiguous = set(ncat[ncat > 1].index)

    pdb = df.pdb_id.astype(str).to_numpy()
    order = {p: i for i, p in enumerate(proteins)}
    idx = np.array([order[p] for p in pdb], dtype=int)
    n = len(proteins)

    # valid consecutive pair count (d res_seq == 1 inside the same pdb block)
    dr = np.diff(df.res_seq.to_numpy(float))
    good = (idx[1:] == idx[:-1]) & np.isfinite(dr) & (dr == 1)
    nvalid = np.bincount(idx[:-1][good], minlength=n).astype(float)
    nres = np.bincount(idx, minlength=n).astype(float)

    labels = pd.factorize(np.asarray(lab))[0]
    keep = np.array([p not in ambiguous for p in proteins])

    masses = np.load(os.path.join(ROOT, "results", "clustering",
                                  f"masses_{tier}_g{grid}.npy"))
    assert masses.shape[0] == n, f"cache shape {masses.shape} vs n={n}"
    return df, angles, idx, labels, keep, nvalid, nres, ambiguous, masses, proteins


def empirical_histogram(df, idx, n, grid):
    """Exactly the variants_*.csv empirical protocol (clustering_variants.py)."""
    th = df.theta.to_numpy(np.float64) * 2 * np.pi / 360.0
    ta = df.tau.to_numpy(np.float64) * 2 * np.pi / 360.0
    xi = np.clip(((th + np.pi) / (2 * np.pi) * grid).astype(int), 0, grid - 1)
    yi = np.clip(((ta + np.pi) / (2 * np.pi) * grid).astype(int), 0, grid - 1)
    hist = np.zeros((n, grid * grid), np.float64)
    np.add.at(hist, (idx, xi * grid + yi), 1.0)
    nz = hist.sum(1, keepdims=True)
    hist = np.where(nz > 0, hist / np.maximum(nz, 1), 1.0 / (grid * grid))
    hist = hist + 1e-3
    hist /= hist.sum(1, keepdims=True)
    return hist


# ------------------------------------------------------------------ metrics
def ward_scores(D, labels, k):
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), method="ward")
    pred = fcluster(Z, t=k, criterion="maxclust")
    return pred, adjusted_rand_score(labels, pred), normalized_mutual_info_score(
        labels, pred, average_method="arithmetic")


def nn_indices(D, knn_k):
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :knn_k]


def nn_acc(nbr, labels):
    y = np.asarray(labels)
    return float(np.mean([np.bincount(y[nbr[i]]).argmax() == y[i]
                          for i in range(len(y))]))


def stratify_quintiles(x):
    """Rank-based quintiles, ties broken by order (deterministic)."""
    r = pd.Series(x).rank(method="first").to_numpy()
    return pd.qcut(r, 5, labels=False)


def shuffle_nulls(y, stats_fns, rng, n_shuffle):
    """stats_fns: list of callables taking a permuted label array -> scalar."""
    out = np.empty((n_shuffle, len(stats_fns)))
    y = np.asarray(y)
    for b in range(n_shuffle):
        yp = rng.permutation(y)
        for j, fn in enumerate(stats_fns):
            out[b, j] = fn(yp)
    return out


def stratified_shuffle_nulls(y, strata, stats_fns, rng, n_shuffle):
    y = np.asarray(y)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = np.empty((n_shuffle, len(stats_fns)))
    for b in range(n_shuffle):
        yp = y.copy()
        for g in groups:
            yp[g] = rng.permutation(yp[g])
        for j, fn in enumerate(stats_fns):
            out[b, j] = fn(yp)
    return out


def pvalue(null, obs):
    return (1.0 + np.sum(null >= obs)) / (len(null) + 1.0)


def rank_z(v):
    v = np.asarray(v, float)
    r = pd.Series(v).rank().to_numpy()
    r = (r - r.mean()) / r.std()
    return r


def mantel_perm(Da, Db, rng, n_perm):
    """Spearman correlation of condensed distances + protein-permutation null.

    Ranks commute with permutation (rank(pi(X)) = pi(rank(X)), ties averaged),
    so B is rank-transformed once and only gathers change per permutation.
    """
    n = Da.shape[0]
    I, J = np.triu_indices(n, 1)
    a = rank_z(squareform(np.clip(Da, 0, None), checks=False))
    B = np.clip(Db, 0, None)
    Brank = np.empty((n, n))
    Brank[I, J] = rank_z(B[I, J])
    Brank[J, I] = Brank[I, J]
    Brank[np.arange(n), np.arange(n)] = 0.0

    def corr_with(perm):
        b = Brank[np.ix_(perm, perm)][I, J]
        b = (b - b.mean()) / b.std()
        return float(np.mean(a * b))

    obs = corr_with(np.arange(n))
    null = np.array([corr_with(rng.permutation(n)) for _ in range(n_perm)])
    return obs, null


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--n-shuffle", type=int, default=1000)
    ap.add_argument("--n-mantel", type=int, default=200)
    ap.add_argument("--tiers", nargs="+", default=list(TIER_RUNS))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "clustering"))
    args = ap.parse_args(argv)

    rng = np.random.default_rng(SEED)
    rows, mantel_rows = [], []

    for tier in args.tiers:
        print(f"\n=== {tier} ===", flush=True)
        df, angles, idx, labels, keep, nvalid, nres, ambiguous, masses, proteins = \
            load_tier(tier, args.grid)
        n_all = len(proteins)
        k = len(np.unique(labels[keep]))
        print(f"proteins: {n_all} total, {int(keep.sum())} kept "
              f"({len(ambiguous)} ambiguous excluded: {sorted(ambiguous)}), k={k}")
        print("class sizes (kept):", np.bincount(labels[keep]).tolist(),
              " majority share:", f"{np.bincount(labels[keep]).max() / keep.sum():.3f}")

        # representations on ALL proteins, then subset to kept
        D_model = hellinger_matrix(masses.reshape(n_all, -1))
        D_hist = hellinger_matrix(empirical_histogram(df, idx, n_all, args.grid))
        lv = np.log1p(nvalid)
        D_len = np.abs(lv[:, None] - lv[None, :])

        sub = lambda D: D[np.ix_(keep, keep)]
        y = labels[keep]
        reps = {"PC-NCSF density": sub(D_model),
                "empirical histogram": sub(D_hist),
                "length only": sub(D_len)}
        lvk = lv[keep]
        strata = stratify_quintiles(lvk)

        # ---- cross-check reference values (all-protein protocols)
        ref_hist = {"easy": 0.1160, "moderate": 0.3206,
                    "hard": 0.0742, "challenging": 0.0936}[tier]
        ref_len = {"easy": 0.0466, "moderate": 0.4480,
                   "hard": 0.2198, "challenging": 0.5465}[tier]
        ref_pcncsf = {"easy": 0.00303883, "moderate": 0.05932363,
                      "hard": 0.07634986, "challenging": 0.10319305}[tier]

        for name, D in reps.items():
            pred, ari, nmi = ward_scores(D, y, k)
            nbr1, nbr5 = nn_indices(D, 1), nn_indices(D, 5)
            acc1, acc5 = nn_acc(nbr1, y), nn_acc(nbr5, y)

            # global shuffle nulls
            null = shuffle_nulls(
                y,
                [lambda yp: adjusted_rand_score(yp, pred),
                 lambda yp: nn_acc(nbr5, yp)],
                rng, args.n_shuffle)
            ari_null, knn_null = null[:, 0], null[:, 1]

            # length-stratified shuffle nulls
            snull = stratified_shuffle_nulls(
                y, strata,
                [lambda yp: adjusted_rand_score(yp, pred),
                 lambda yp: nn_acc(nbr5, yp)],
                rng, args.n_shuffle)
            ari_snull, knn_snull = snull[:, 0], snull[:, 1]

            majority = np.bincount(y).max() / len(y)
            row = dict(tier=tier, representation=name, m=int(keep.sum()), k=k,
                       ward_ARI=ari, ward_NMI=nmi,
                       knn1_acc=acc1, knn5_acc=acc5, majority=majority,
                       ari_null_mean=ari_null.mean(), ari_p=pvalue(ari_null, ari),
                       knn5_null_mean=knn_null.mean(), knn5_p=pvalue(knn_null, acc5),
                       ari_strat_null_mean=ari_snull.mean(),
                       ari_strat_p=pvalue(ari_snull, ari),
                       knn5_strat_null_mean=knn_snull.mean(),
                       knn5_strat_p=pvalue(knn_snull, acc5))
            rows.append(row)
            print(f"  {name:20s} ward ARI={ari:.4f} NMI={nmi:.4f} | "
                  f"1NN={acc1:.3f} 5NN={acc5:.3f} maj={majority:.3f} | "
                  f"5NN null={knn_null.mean():.3f} (p={row['knn5_p']:.4f}) "
                  f"strat={knn_snull.mean():.3f} (p={row['knn5_strat_p']:.4f}) | "
                  f"ARI null={ari_null.mean():.4f} (p={row['ari_p']:.4f}) "
                  f"strat={ari_snull.mean():.4f} (p={row['ari_strat_p']:.4f})",
                  flush=True)

        # ---- Mantel correlations with length
        for name, D in (("PC-NCSF density", sub(D_model)),
                        ("empirical histogram", sub(D_hist))):
            obs, null = mantel_perm(D, sub(D_len), rng, args.n_mantel)
            mantel_rows.append(dict(tier=tier, representation=name,
                                    spearman_vs_length=obs,
                                    null_mean=null.mean(), p=pvalue(null, obs)))
            print(f"  Mantel {name:20s} rho={obs:.3f} "
                  f"(perm null {null.mean():.3f}, p={pvalue(null, obs):.4f})",
                  flush=True)

        # ---- cross-checks against committed artifacts
        got = {r["representation"]: r["ward_ARI"] for r in rows if r["tier"] == tier}
        print(f"  cross-check (kept-subset vs committed all-protein refs): "
              f"PC-NCSF {got['PC-NCSF density']:.4f} vs {ref_pcncsf:.4f}; "
              f"histogram {got['empirical histogram']:.4f} vs {ref_hist:.4f}; "
              f"length {got['length only']:.4f} vs {ref_len:.4f}", flush=True)

    out = os.path.join(args.out_dir, f"r1c5_length_controls_g{args.grid}.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out2 = os.path.join(args.out_dir, f"r1c5_mantel_g{args.grid}.csv")
    pd.DataFrame(mantel_rows).to_csv(out2, index=False)
    print(f"\nsaved: {out}\nsaved: {out2}")
    print(f"config: grid={args.grid} n_shuffle={args.n_shuffle} "
          f"n_mantel={args.n_mantel} seed={SEED}")


if __name__ == "__main__":
    main()

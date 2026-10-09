"""Corrected SCOP hybrid clustering for PC-NCSF.
Run from repository root:
  python scripts/clustering_hybrid_corrected.py --tier all --grid 100 --eligible-subset --knn

Corrections: circular means use cos/sin; sequential resultant lengths are normalized
by valid-pair counts; pair counts are diagnostics, not B2 features; hybrid distances
come from a concatenated Euclidean embedding, so Ward is mathematically appropriate.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scop_clustering import TIER_RUNS, ROOT, load_proteins

ALPHAS = (0., .25, .5, .75, 1.)
LINKAGES = ("ward", "average", "complete")


def meddist(Z):
    d = pdist(Z)
    d = d[np.isfinite(d) & (d > 1e-12)]
    return float(np.median(d)) if len(d) else 1.


def standardize(X):
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.
    return np.nan_to_num((X-X.mean(axis=0))/sd)


def normalize_embedding(Z):
    Z = standardize(Z)
    return Z / meddist(Z)


def load_data(tier):
    path = os.path.join(ROOT, "SCOP", tier, "data.csv")
    df = pd.read_csv(path).dropna(subset=["theta", "tau"]).reset_index(drop=True)
    _, _, proteins, labels = load_proteins(tier)
    proteins = [str(p) for p in proteins]
    order = {p:i for i,p in enumerate(proteins)}
    pdb = df.pdb_id.astype(str).to_numpy()
    unknown = set(pdb)-set(order)
    if unknown:
        raise RuntimeError(f"PDB IDs missing from load_proteins: {sorted(unknown)[:5]}")
    idx = np.array([order[p] for p in pdb], dtype=int)
    return df, np.deg2rad(df.theta.to_numpy(float)), np.deg2rad(df.tau.to_numpy(float)), \
           df.res_seq.to_numpy(float), idx, np.asarray(labels), proteins


def circ_summary(phi, idx, n, mask=None, owners=None):
    if mask is None:
        owners = idx
        angles = phi
    else:
        angles = phi[mask]
    c = np.bincount(owners, weights=np.cos(angles), minlength=n)
    s = np.bincount(owners, weights=np.sin(angles), minlength=n)
    cnt = np.bincount(owners, minlength=n).astype(float)
    mag = np.hypot(c, s)
    R = mag / np.maximum(cnt, 1.)
    ok = mag > 1e-12
    cm, sm = np.zeros(n), np.zeros(n)
    cm[ok], sm[ok] = c[ok]/mag[ok], s[ok]/mag[ok]
    return np.column_stack((cm, sm, R))


def b1(theta, tau, idx, n):
    # Four circular variables, each represented by cos(mean), sin(mean), R.
    return np.column_stack([circ_summary(p, idx, n) for p in
                            (theta, tau, theta+tau, theta-tau)])


def b2(theta, tau, res_seq, idx, n):
    dth, dta, dr = np.diff(theta), np.diff(tau), np.diff(res_seq)
    same = idx[1:] == idx[:-1]
    good = same & np.isfinite(dr) & (dr == 1)
    owners = idx[:-1][good]
    count = np.bincount(owners, minlength=n).astype(int)

    def stats(phi):
        ang = phi[good]
        c = np.bincount(owners, weights=np.cos(ang), minlength=n)
        s = np.bincount(owners, weights=np.sin(ang), minlength=n)
        mag = np.hypot(c, s)
        R = mag / np.maximum(count, 1)
        ok = mag > 1e-12
        cm, sm = np.zeros(n), np.zeros(n)
        cm[ok], sm[ok] = c[ok]/mag[ok], s[ok]/mag[ok]
        return np.column_stack((cm, sm, R))

    X = np.column_stack((stats(dth), stats(dta), stats(dth+dta)[:, 2]))
    possible = np.bincount(idx[:-1][same], minlength=n)
    frac = count / np.maximum(possible, 1)
    return X, count, frac


def density_embedding(masses):
    # Cache stores [n_proteins, grid, grid]; flatten each protein's grid
    # to one feature vector before pairwise-distance scaling.
    P = np.asarray(masses, dtype=np.float64)
    if P.ndim < 2:
        raise ValueError(f"Expected masses with shape [n_proteins, ...], got {P.shape}")
    P = np.clip(P.reshape(P.shape[0], -1), 0.0, None)
    sums = P.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Density cache contains zero-mass row")
    P /= sums
    Z = np.sqrt(P)/np.sqrt(2.)
    return Z/meddist(Z)


def hybrid(Z0, Z1, Z2, a):
    # Squared Euclidean distance = a*D0^2 +(1-a)/2*D1^2 +(1-a)/2*D2^2.
    return np.concatenate((np.sqrt(a)*Z0,
                           np.sqrt((1-a)/2)*Z1,
                           np.sqrt((1-a)/2)*Z2), axis=1)


def empirical_hist(theta, tau, idx, n, grid):
    # Angles are radians in [-pi, pi]; modulo maps them to [0, 2pi).
    x = np.clip(((theta % (2*np.pi))/(2*np.pi)*grid).astype(int), 0, grid-1)
    y = np.clip(((tau % (2*np.pi))/(2*np.pi)*grid).astype(int), 0, grid-1)
    H = np.zeros((n, grid*grid), float)
    np.add.at(H, (idx, x*grid+y), 1.)
    H /= np.maximum(H.sum(axis=1, keepdims=True), 1.)
    H += 1e-3
    H /= H.sum(axis=1, keepdims=True)
    Z = np.sqrt(H)/np.sqrt(2.)
    return Z/meddist(Z)


def evaluate(Z, labels, tier, subset, rep, alpha, rows):
    k = len(np.unique(labels))
    if len(Z) != len(labels):
        raise RuntimeError("Embedding and labels have different lengths")
    for method in LINKAGES:
        # Ward receives Euclidean observations, not a precomputed arbitrary distance.
        tree = linkage(Z, method=method, metric="euclidean")
        pred = fcluster(tree, t=k, criterion="maxclust")
        rows.append(dict(tier=tier, subset=subset, representation=rep, alpha=alpha,
                         linkage=method, k=k, n_proteins=len(labels),
                         ARI=adjusted_rand_score(labels, pred),
                         NMI_arithmetic=normalized_mutual_info_score(
                             labels, pred, average_method="arithmetic")))


def knn(Z, labels, tier, subset, rep, rows):
    D = squareform(pdist(Z))
    np.fill_diagonal(D, np.inf)
    k = min(5, len(labels)-1)
    nbr = np.argsort(D, axis=1)[:, :k]
    y = np.asarray(labels)
    acc = np.mean([np.mean(y[nbr[i]] == y[i]) for i in range(len(y))])
    rng, null = np.random.default_rng(0), []
    for _ in range(200):
        yp = rng.permutation(y)
        null.append(np.mean([np.mean(yp[nbr[i]] == yp[i]) for i in range(len(y))]))
    rows.append(dict(tier=tier, subset=subset, representation=rep, knn_k=k,
                     knn_same_label_fraction=acc, knn_perm_null_mean=np.mean(null),
                     n_proteins=len(y)))


def run_tier(tier, grid, rows, do_knn, eligible):
    print(f"\n=== {tier} ===", flush=True)
    df, theta, tau, res, idx, labels, proteins = load_data(tier)
    n = len(proteins)
    if len(np.unique(idx)) != n:
        raise RuntimeError("At least one protein has no rows after filtering")
    cache = os.path.join(ROOT, "results", "clustering", f"masses_{tier}_g{grid}.npy")
    masses = np.load(cache)
    if masses.shape[0] != n:
        raise RuntimeError(f"Cache/protein mismatch: {masses.shape[0]} != {n}")
    Z0 = density_embedding(masses)
    Z1 = normalize_embedding(b1(theta, tau, idx, n))
    X2, nvalid, validfrac = b2(theta, tau, res, idx, n)
    Z2 = normalize_embedding(X2)
    Z4 = empirical_hist(theta, tau, idx, n, grid)
    print(f"residues={len(df)} proteins={n} k={len(np.unique(labels))}")
    print(f"B2 >=5 valid pairs: {np.mean(nvalid >= 5):.3f}; "
          f"median count={np.median(nvalid):.1f}; median valid fraction={np.median(validfrac):.3f}")
    reps = [("R0 density/Hellinger", Z0), ("R1 B1-circular", Z1),
            ("R2 B2-normalized-sequential", Z2), ("R4 empirical-histogram", Z4)]
    for name, Z in reps:
        evaluate(Z, labels, tier, "all", name, np.nan, rows)
        if do_knn:
            knn(Z, labels, tier, "all", name, rows)
    # Pair-count-only baseline: tests whether sequence length/count itself carries labels.
    evaluate(np.log1p(nvalid).reshape(-1, 1), labels, tier, "all",
             "DIAGNOSTIC log1p(valid_pair_count)", np.nan, rows)
    for a in ALPHAS:
        Zh = hybrid(Z0, Z1, Z2, a)
        evaluate(Zh, labels, tier, "all", "R3 hybrid-Euclidean-concat", a, rows)
        if do_knn and a in (0., .5, 1.):
            knn(Zh, labels, tier, "all", f"R3 hybrid alpha={a}", rows)
    if eligible:
        keep = nvalid >= 5
        print(f"Eligible subset n={keep.sum()}/{n}")
        if keep.sum() > len(np.unique(labels[keep])):
            for name, Z in reps:
                evaluate(Z[keep], labels[keep], tier, "B2-eligible-nvalid>=5",
                         name, np.nan, rows)
            evaluate(np.log1p(nvalid[keep]).reshape(-1, 1), labels[keep], tier,
                     "B2-eligible-nvalid>=5", "DIAGNOSTIC log1p(valid_pair_count)",
                     np.nan, rows)
            for a in ALPHAS:
                evaluate(hybrid(Z0, Z1, Z2, a)[keep], labels[keep], tier,
                         "B2-eligible-nvalid>=5", "R3 hybrid-Euclidean-concat", a, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="all", choices=list(TIER_RUNS)+["all"])
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--knn", action="store_true")
    p.add_argument("--eligible-subset", action="store_true")
    a = p.parse_args()
    tiers = list(TIER_RUNS) if a.tier == "all" else [a.tier]
    outdir = a.out_dir or os.path.join(ROOT, "results", "clustering_hybrid_corrected")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for tier in tiers:
        run_tier(tier, a.grid, rows, a.knn, a.eligible_subset)
    out = os.path.join(outdir, f"hybrid_corrected_{a.tier}_g{a.grid}.csv")
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print("\n=== Results (all linkages, complete alpha grid) ===")
    cols = [c for c in ("tier","subset","representation","alpha","linkage","k",
                        "n_proteins","ARI","NMI_arithmetic") if c in result.columns]
    print(result[cols].to_string(index=False, na_rep="-"))
    print(f"\nSaved: {out}")
    print("NMI normalization: arithmetic mean (scikit-learn default).")


if __name__ == "__main__":
    main()

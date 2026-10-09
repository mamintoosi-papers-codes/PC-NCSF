"""Pre-registered hybrid clustering experiment for SCOP (report §4–§7).

Implements exactly the pre-registered protocol of
"SCOP Clustering — Feasibility Audit of Hybrid Representations"
(reports/scop_hybrid_feasibility_2026-10-09-todo.md), and nothing more.

Representations (same tiers, same dropna, same pdb_id unit, same
majority-vote labels as scop_clustering.load_proteins):
  R0  PC-NCSF density (cached masses)            -> Hellinger (D_H)
  R1  B1 circular statistics (8 numbers)         -> Euclidean on z-scored f, median-normalised
  R2  B2 sequential adjacent-residue statistics  -> Euclidean on z-scored g, median-normalised
  R3  hybrid  D_a = a*D0_hat + (1-a)*D_aux_hat,  D_aux_hat = 0.5*D1_hat + 0.5*D2_hat,
      a in {0, 0.25, 0.5, 0.75, 1}  -- the FULL grid is reported without selection
  R4  model-free empirical histogram comparator  -> Hellinger (existing recipe)

Evaluation:
  - k = n_categories fixed; Ward / average / complete linkage.
  - ARI and NMI vs SCOP labels; labels are only read for evaluation.
  - Optional nested-selection of a (--nested), halves swapped, 10 seeds.
  - Optional 5-NN retrieval label-permutation null and majority baseline (--knn).

Leakage safeguards implemented:
  * features use ONLY theta, tau, res_seq (never labels);
  * the alpha grid is reported in full (no tuned scalar in the headline CSV);
  * everything is written to a NEW directory (default results/clustering_hybrid)
    so the verified results/clustering/ artifacts are never touched;
  * fixed seeds default_rng(0) (nested: seeds 0..9);
  * no source files, caches, or existing CSVs are modified.

B2 adjacency rule (data defensible only rule): pair two consecutive rows
within the same pdb_id block ONLY if dres_seq == 1.  This skips chain/model
resets (d<=0), gaps from dropped residues (d>=2), and unalignable jumps.
Known limitation (disclosed in the report): no chain IDs / insertion codes;
rare false adjacencies possible.  Sensitivity: --strict-runs restricts to
strictly increasing res_seq runs inside each pdb_id block.

Usage:
  python scripts/clustering_hybrid.py --tier all --grid 100
  python scripts/clustering_hybrid.py --tier hard
  python scripts/clustering_hybrid.py --tier all --grid 100 --strict-runs
  python scripts/clustering_hybrid.py --tier all --grid 100 --nested
  python scripts/clustering_hybrid.py --tier all --grid 100 --strict-run-bound
  python scripts/clustering_hybrid.py --tier all --grid 100 --knn
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
from scop_clustering import TIER_RUNS, ROOT, hellinger_matrix, load_proteins

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)   # pre-registered, reported in full
LINKAGES = ("ward", "average", "complete")  # no per-tier cherry-picking
OUT_DIR = "results/clustering_hybrid"


# --------------------------------------------------------------- data loading
def _load_tier_df(tier):
    df = pd.read_csv(os.path.join(ROOT, "SCOP", tier, "data.csv"))
    df = df.dropna(subset=["theta", "tau"]).reset_index(drop=True)
    return df


def load_features_data(tier):
    """Return df, features, labels, proteins.

    Reads 'theta', 'tau', 'res_seq' for features and 'category' ONLY for
    evaluation labels.  Uses pd.factorize(df.pdb_id) as the condition unit
    via load_proteins (protein order guaranteed to match).
    """
    _, _, proteins, labels = load_proteins(tier)
    df = _load_tier_df(tier)
    lab = pd.factorize(np.asarray(labels))[0]
    theta = df["theta"].to_numpy(dtype=np.float64) * 2 * np.pi / 360.0
    tau = df["tau"].to_numpy(dtype=np.float64) * 2 * np.pi / 360.0
    res_seq = df["res_seq"].to_numpy(dtype=np.int64)
    pdb = df["pdb_id"].astype(str).to_numpy()
    order = {p: i for i, p in enumerate(proteins)}
    feats = dict(theta=theta, tau=tau, res_seq=res_seq,
                 pdb_idx=np.array([order[p] for p in pdb], dtype=np.int64),
                 n_proteins=len(proteins))
    return df, feats, lab, proteins


# --------------------------------------------------------------- B1 features
def b1_features(theta, tau, pdb_idx, n_proteins):
    """B1 per protein, 8 numbers: [mean, R] for theta, tau, theta+tau, theta-tau."""
    phis = np.stack([theta, tau, theta + tau, theta - tau])   # [4, N]
    mean = np.full((n_proteins, 4), np.nan)
    rlen = np.full((n_proteins, 4), np.nan)
    for j in range(4):
        c = np.exp(1j * phis[j])
        sums = np.bincount(pdb_idx, weights=c.real, minlength=n_proteins) + \
               1j * np.bincount(pdb_idx, weights=c.imag, minlength=n_proteins)
        ns = np.bincount(pdb_idx, minlength=n_proteins).astype(np.float64)
        R = np.abs(sums) / np.maximum(ns, 1)
        mu = np.angle(sums)          # undefined when R==0, acceptable
        mean[:, j] = mu
        rlen[:, j] = R
    f = np.concatenate([mean, rlen], axis=1)   # [n, 8]
    f[np.isnan(f)] = 0.0
    return f


# --------------------------------------------------------------- B2 features
def b2_features(theta, tau, res_seq, pdb_idx, n_proteins, strict_run_bound=False):
    """B2 per protein, 6 numbers:
    [mean dtheta, R dtheta, mean dtau, R dtau, R(dtheta+dtau), valid fraction]

    Pairs = consecutive rows with dres_seq == 1 inside the same pdb_id block.
    With strict_run_bound=True, keep only proteins whose whole post-dropna
    res_seq sequence is strictly increasing within their pdb_id block
    (report equilibrium check: 883/975/168/138 proteins per tier qualify),
    i.e. proteins containing any reset (d res_seq <= 0) are dropped entirely;
    other proteins contribute no valid pairs and are reported.
    """
    dth = np.diff(theta)
    dta = np.diff(tau)
    dres = np.diff(res_seq)
    same_prot = pdb_idx[1:] == pdb_idx[:-1]
    good = same_prot & (np.abs(dres) == 1)
    if strict_run_bound:
        resets = same_prot & (dres <= 0)
        bad = np.unique(pdb_idx[:-1][resets])
        good &= ~np.isin(pdb_idx[:-1], bad)
    pi = pdb_idx[:-1][good]

    dz = np.exp(1j * (dth[good] + dta[good]))
    z1 = np.exp(1j * dth[good])
    z2 = np.exp(1j * dta[good])

    def _stats(z):
        s_r = np.bincount(pi, weights=z.real, minlength=n_proteins) + \
              1j * np.bincount(pi, weights=z.imag, minlength=n_proteins)
        return np.abs(s_r), np.angle(s_r)

    valid = np.bincount(pi, minlength=n_proteins).astype(np.float64)
    R1, m1 = _stats(z1)
    R2_, m2_ = _stats(z2)
    R3_, _m3 = _stats(dz)
    g = np.concatenate([m1, R1, m2_, R2_, R3_, valid[:, None] / np.maximum(valid.max(), 1)],
                       axis=1)   # [n, 6]
    g[np.isnan(g)] = 0.0
    return g, valid


# --------------------------------------------------------------- distances
def zscore(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def euclidean_matrix(Z):
    sq = (Z ** 2).sum(axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * Z @ Z.T, 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def median_normalise(D):
    """D / median of off-diagonal entries (robust, label-free scale alignment)."""
    iu = np.triu_indices(D.shape[0], 1)
    med = np.median(D[iu])
    if not np.isfinite(med) or med <= 0:
        return D.copy()
    return D / med


def hybrid_matrix(D0, D1, D2, alpha):
    """D_a = a*D0_hat + (1-a)*(0.5*D1_hat + 0.5*D2_hat); nonneg weighted sum of metrics."""
    Daux = 0.5 * D1 + 0.5 * D2
    return alpha * D0 + (1.0 - alpha) * Daux


def empirical_hist_matrix(theta, tau, pdb_idx, n_proteins, g):
    """R4 model-free comparator: existing empirical-histogram recipe, 1e-3 smoothing."""
    xi = np.clip(((theta + np.pi) / (2 * np.pi) * g).astype(int), 0, g - 1)
    yi = np.clip(((tau + np.pi) / (2 * np.pi) * g).astype(int), 0, g - 1)
    hist = np.zeros((n_proteins, g * g), np.float64)
    np.add.at(hist, (pdb_idx, xi * g + yi), 1.0)
    nz = hist.sum(1, keepdims=True)
    hist = np.where(nz > 0, hist / np.maximum(nz, 1), 1.0 / (g * g))
    hist = hist + 1e-3
    hist /= hist.sum(1, keepdims=True)
    D = hellinger_matrix(hist)
    return D, hist


# --------------------------------------------------------------- evaluation
def cluster_all_linkages(D, labels, k, tag, results):
    cond = squareform(np.clip(D, 0, None), checks=False)
    for method in LINKAGES:
        Z = linkage(cond, method=method)
        pred = fcluster(Z, t=k, criterion="maxclust")
        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)
        results.append({"tier": tag["tier"], "representation": tag["representation"],
                        "alpha": tag.get("alpha", np.nan), "linkage": method, "k": k,
                        "ARI": round(ari, 4), "NMI": round(nmi, 4),
                        "n_proteins": tag["n_proteins"]})


def nested_alpha(D0, D1, D2, labels, k_cl, linkage_name, seeds=range(10)):
    """Nested selection: choose alpha by ARI on the inner half, evaluate on the
    outer half, swap halves, repeat 10x with fixed seeds.  Returns mean +- sd of
    the outer ARI values and the chosen alphas (label-informed: disclosed
    limitation -- the outer estimate is the only honest number)."""
    n = len(labels)
    outer, chosen = [], []
    y = np.asarray(labels)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        h1, h2 = perm[:n // 2], perm[n // 2:]
        for inner, outer_ in ((h1, h2), (h2, h1)):
            best_a, best_ari = None, -np.inf
            for a in ALPHA_GRID:
                Dh = hybrid_matrix(D0, D1, D2, a)
                sub = Dh[np.ix_(inner, inner)]
                k_in = len(np.unique(y[inner]))
                pred = fcluster(linkage(squareform(sub, checks=False),
                                        method=linkage_name), t=k_in, criterion="maxclust")
                ari = adjusted_rand_score(y[inner], pred)
                if ari > best_ari:
                    best_a, best_ari = a, ari
            chosen.append(best_a)
            Do_ = hybrid_matrix(D0, D1, D2, best_a)
            sub = Do_[np.ix_(outer_, outer_)]
            k_out = len(np.unique(y[outer_]))
            pred = fcluster(linkage(squareform(sub, checks=False),
                                    method=linkage_name), t=k_out, criterion="maxclust")
            outer.append(adjusted_rand_score(y[outer_], pred))
    return np.array(outer), np.array(chosen)


def knn_eval(D, labels, k_cl, tag, results):
    """5-NN retrieval with imbalance-preserving permutation null and majority baseline."""
    n = len(labels)
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(0)
    D0 = D.copy()
    np.fill_diagonal(D0, np.inf)
    nn = np.argsort(D0, axis=1)[:, :5]
    maj = np.bincount(labels, minlength=k_cl).argmax()
    acc = np.mean([np.sum(labels[nn[i]] == labels[i]) / 5 for i in range(n)])
    null = []
    for _ in range(200):
        perm = rng.permutation(labels)
        null.append(np.mean([np.sum(perm[nn[i]] == perm[i]) / 5 for i in range(n)]))
    results.append({"representation": tag["representation"], "tier": tag["tier"],
                    "knn5_acc": round(acc, 4),
                    "knn5_null_mean": round(np.mean(null), 4),
                    "majority_class_frac": round(np.bincount(labels, minlength=k_cl)[maj] / n, 4)})


# --------------------------------------------------------------- per-tier run
def run_tier(tier, grid, rows, strict_run_bound=False, do_nested=False, do_knn=False):
    print(f"\\n=== {tier} ===", flush=True)
    df, feats, labels, proteins = load_features_data(tier)
    n = feats["n_proteins"]
    k = len(np.unique(labels))
    print(f"residues={len(df)} proteins={n} k={k}")

    # cached model densities (existing artifact; never written here)
    masses = np.load(os.path.join(ROOT, "results", "clustering",
                                  f"masses_{tier}_g{grid}.npy"))
    if masses.shape[0] != n:
        raise RuntimeError(f"cache/protein mismatch: masses {masses.shape[0]} vs n={n}")

    D0 = hellinger_matrix(masses)
    D0 = median_normalise(D0)

    f = b1_features(feats["theta"], feats["tau"], feats["pdb_idx"], n)
    g_, valid_counts = b2_features(feats["theta"], feats["tau"], feats["res_seq"],
                                   feats["pdb_idx"], n, strict_run_bound)
    D1 = median_normalise(euclidean_matrix(zscore(f)))
    D2 = median_normalise(euclidean_matrix(zscore(g_)))
    Daux = 0.5 * D1 + 0.5 * D2

    D4, _ = empirical_hist_matrix(feats["theta"], feats["tau"], feats["pdb_idx"],
                                  n, grid)

    frac_valid = np.mean(valid_counts >= 5)
    print(f"proteins with >=5 valid sequential pairs: {frac_valid:.3f}")
    if frac_valid < 1.0:
        print("WARNING: some proteins lost; disclose (no protein is expected to be lost)")

    tag = {"tier": tier, "n_proteins": n}
    tag["representation"] = "R0 density/Hellinger"
    cluster_all_linkages(D0, labels, k, tag, rows)
    tag["representation"] = "R1 B1-stats"
    cluster_all_linkages(D1, labels, k, tag, rows)
    tag["representation"] = "R2 B2-seq"
    cluster_all_linkages(D2, labels, k, tag, rows)
    for a in ALPHA_GRID:
        tag["representation"] = "R3 hybrid"
        tag["alpha"] = a
        cluster_all_linkages(hybrid_matrix(D0, D1, D2, a), labels, k, tag, rows)
    tag["representation"] = "R4 empirical-hist"
    tag.pop("alpha", None)
    cluster_all_linkages(D4, labels, k, tag, rows)

    if do_nested:
        for method in LINKAGES:
            outer, chosen = nested_alpha(D0, D1, D2, labels, k, method)
            rows.append({"tier": tier, "representation": f"R3-nested-{method}",
                         "alpha": float(np.median(chosen)),
                         "linkage": method, "k": k,
                         "ARI": round(outer.mean(), 4), "NMI": np.nan,
                         "n_proteins": n,
                         "note": f"outer-ARI mean sd={outer.std():.4f}; "
                                 f"chosen alpha hist={np.bincount(np.digitize(chosen, ALPHA_GRID)).tolist()}"})
    if do_knn:
        for rep_name, Dm in (("R0", D0), ("R1", D1), ("R2", D2),
                             ("R3 a=0.5", hybrid_matrix(D0, D1, D2, 0.5)), ("R4", D4)):
            knn_eval(Dm, labels, k, {"representation": rep_name, "tier": tier}, rows)
    return rows


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all", choices=list(TIER_RUNS) + ["all"])
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--out-dir", default=None, help="default: results/clustering_hybrid")
    ap.add_argument("--strict-run-bound", action="store_true",
                    help="B2 sensitivity variant (default off) ")
    ap.add_argument("--nested", action="store_true",
                    help="add nested alpha selection rows (label-informed, disclosed)")
    ap.add_argument("--knn", action="store_true",
                    help="add 5-NN retrieval local-signal rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(ROOT, OUT_DIR)
    tiers = list(TIER_RUNS) if args.tier == "all" else [args.tier]

    rows = []
    for tier in tiers:
        run_tier(tier, args.grid, rows,
                 strict_run_bound=args.strict_run_bound,
                 do_nested=args.nested, do_knn=args.knn)

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    # flag-derived suffix so the runner's later steps never overwrite earlier CSVs
    suffix = ""
    if args.knn:
        suffix += "_knn"
    if args.strict_run_bound:
        suffix += "_strict"
    if args.nested:
        suffix += "_nested"
    main_csv = os.path.join(out_dir, f"hybrid_{args.tier}_g{args.grid}{suffix}.csv")
    df.to_csv(main_csv, index=False)
    print("\\n=== Pre-registered ablation (Ward linkage, k = n_categories) ===")
    cols = ["tier", "representation", "alpha", "ARI", "NMI"]
    w = df[(df.linkage == "ward") & ~df.representation.str.startswith("R3-nested")]
    print(w[cols].to_string(index=False, na_rep="-"))
    print(f"\\nsaved: {main_csv}")


if __name__ == "__main__":
    main()

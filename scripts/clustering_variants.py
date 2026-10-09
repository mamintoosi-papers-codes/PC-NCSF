"""Explore clustering variants for the SCOP real-data clustering experiment.

Questions answered:
  1. Better distances?      -> Jensen-Shannon, symmetric KL, Wasserstein-1 on
                                angular marginals (closed form), Hellinger.
  2. Better clusterers?     -> ward/average/complete, spectral, PCA+GMM,
                                k-means on sqrt-densities; k-sweep 2..10.
  3. Better features?       -> joint torus density vs (theta,tau) marginals;
                                marginals-only Wasserstein.
  4. Ceiling / control?     -> empirical per-protein histograms (model-free)
                                and random-feature controls.
  5. cond vs uncond?        -> the unconditional model has ONE shared
                                density: all pairwise distances are 0, so
                                clustering is *undefined* for it. We report
                                this explicitly rather than a fake ARI.

Usage:
  /data/python-envs/pytorch/bin/python scripts/clustering_variants.py --tier all
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scop_clustering import TIER_RUNS, ROOT, build_model, hellinger_matrix, load_proteins


def load_labels(tier):
    _, _, proteins, _ = load_proteins(tier)
    df = pd.read_csv(os.path.join(ROOT, "SCOP", tier, "data.csv"))
    df = df.dropna(subset=["theta", "tau"])
    order = {p: i for i, p in enumerate(proteins)}
    first = df.assign(_i=df.pdb_id.map(order)).groupby("_i", sort=True).first()
    lab = pd.factorize(first["category"])[0]
    return lab, df, order


def evaluate(labels, pred, tag, results):
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    results.append({"variant": tag, "ARI": round(ari, 4), "NMI": round(nmi, 4)})
    print(f"    {tag:48s} ARI={ari:+.4f}  NMI={nmi:.4f}")
    return ari


# ------------------------------------------------------------------ distances
def _pairwise(fn, A, B, budget=3e7):
    out = np.empty((A.shape[0], B.shape[0]))
    bs = max(1, int(budget / (B.shape[0] * A.shape[1])))
    for i in range(0, A.shape[0], bs):
        out[i:i + bs] = fn(A[i:i + bs], B)
    return out


def js_distance(P, Q):
    def _k(a, b):
        return np.sum(np.where(a > 0, a * (np.log(np.clip(a, 1e-300, None))
                                           - np.log(np.clip(b, 1e-300, None))), 0.0), axis=-1)
    def _f(p, q):
        p = p[:, None, :] / p.sum(-1, keepdims=True)[:, None, :]
        q = q[None, :, :] / q.sum(-1, keepdims=True)[None, :, :]
        m = 0.5 * (p + q)
        return np.sqrt(np.clip(0.5 * _k(p, m) + 0.5 * _k(q, m), 0, None))
    return _pairwise(_f, P.astype(np.float64), Q.astype(np.float64))


def sym_kl(P, Q):
    def _f(p, q):
        p = p[:, None, :] / p.sum(-1, keepdims=True)[:, None, :]
        q = q[None, :, :] / q.sum(-1, keepdims=True)[None, :, :]
        lp = np.log(np.clip(p, 1e-300, None)); lq = np.log(np.clip(q, 1e-300, None))
        kl_pq = np.sum(p * (lp - lq), -1)
        kl_qp = np.sum(q * (lq - lp), -1)
        return np.sqrt(np.clip(0.5 * (kl_pq + kl_qp), 0, None))
    return _pairwise(_f, P.astype(np.float64), Q.astype(np.float64))


def marginal_w1(marginals):
    """W1 between 1D histograms on a shared grid, closed form:
    W1(p,q) = sum_g |CDF_p(g) - CDF_q(g)| * bin_width.
    marginals: [n, g] (rows sum to 1). Returns [n, n].
    """
    cdf = np.cumsum(marginals.astype(np.float64), axis=1)
    n = cdf.shape[0]
    out = np.zeros((n, n))
    bs = max(1, int(3e7 / (n * cdf.shape[1])))
    for i in range(0, n, bs):
        out[i:i + bs] = np.abs(cdf[i:i + bs, None, :] - cdf[None, :, :]).sum(-1)
    return out


# ------------------------------------------------------------------ clustering
def hierarchical(D, labels, tag, results):
    n_classes = len(np.unique(labels))
    cond = squareform(np.clip(D, 0, None), checks=False)
    Z = linkage(cond, method="ward")
    for method in ("ward", "average", "complete"):
        pred = fcluster(Z if method == "ward" else linkage(cond, method=method),
                        n_classes, "maxclust")
        evaluate(labels, pred, f"{tag} | {method}", results)
    # k sweep: is there ANY k that separates the classes better?
    for k in range(2, 11):
        pred = fcluster(Z, k, "maxclust")
        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)
        results.append({"variant": f"{tag} | k={k}", "ARI": round(ari, 4),
                        "NMI": round(nmi, 4)})
    print(f"    {tag:48s} k-sweep 2..10 done (best k flagged below)")


def spectral(D, labels, tag, results):
    n_classes = len(np.unique(labels))
    try:
        sig = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        aff = np.exp(-D ** 2 / (sig ** 2 + 1e-12))
        np.fill_diagonal(aff, 1.0)
        pred = SpectralClustering(n_clusters=n_classes, affinity="precomputed",
                                  assign_labels="kmeans", random_state=0,
                                  n_init=10).fit_predict(aff)
        evaluate(labels, pred, f"{tag} | spectral", results)
    except Exception as e:
        print(f"    {tag} spectral failed: {e}")


def feature_cluster(X, labels, tag, results, seed=0):
    n_classes = len(np.unique(labels))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    pred = KMeans(n_clusters=n_classes, n_init=20, random_state=seed).fit_predict(X)
    evaluate(labels, pred, f"{tag} | kmeans", results)
    if X.shape[1] > 4:
        Zd = PCA(n_components=min(20, X.shape[1]), random_state=0).fit_transform(X)
    else:
        Zd = X
    pred = GaussianMixture(n_components=n_classes, covariance_type="full",
                           random_state=0, n_init=3).fit_predict(Zd)
    evaluate(labels, pred, f"{tag} | GMM(pca)", results)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="challenging", choices=list(TIER_RUNS) + ["all"])
    ap.add_argument("--grid", type=int, default=100)
    args = ap.parse_args()
    tiers = list(TIER_RUNS) if args.tier == "all" else [args.tier]

    for tier in tiers:
        print(f"\\n================ {tier} ================" , flush=True)
        lab, df, order = load_labels(tier)
        n = len(lab)
        k_true = len(np.unique(lab))
        cache = os.path.join(ROOT, "results", "clustering",
                             f"masses_{tier}_g{args.grid}.npy")
        masses = np.load(cache).reshape(n, -1)
        g = int(np.sqrt(masses.shape[1]))
        results = []

        # 1) model densities: Hellinger (manuscript baseline)
        D_h = hellinger_matrix(masses)
        hierarchical(D_h, lab, "joint/Hellinger", results)
        spectral(D_h, lab, "joint/Hellinger", results)

        # 2) alternative distances (on coarsened grid: same answer, less memory)
        gc = min(g, 25)
        bs_ = g // gc
        mc = masses.reshape(n, gc, bs_, gc, bs_).sum(axis=(2, 4)).reshape(n, -1)
        hierarchical(js_distance(mc, mc), lab, "joint/JS", results)
        hierarchical(sym_kl(mc, mc), lab, "joint/symKL", results)

        # 3) Wasserstein-1 on angular marginals (closed form)
        m2 = masses.reshape(n, g, g)
        marg_t = m2.sum(axis=2)                 # [n, g] theta marginal
        marg_p = m2.sum(axis=1)                 # [n, g] tau marginal
        # Wasserstein-1 on theta marginal only (angular displacement is
        # the natural ground metric on the circle / on the (theta,tau)
        # domain: helices vs sheets differ mainly in theta location).
        D_w1_t = marginal_w1(marg_t)
        hierarchical(D_w1_t, lab, "marginal/W1-theta", results)
        spectral(D_w1_t, lab, "marginal/W1-theta", results)
        # also joint theta+tau (reported separately as marginal/W1)
        D_w1 = D_w1_t + marginal_w1(marg_p)
        hierarchical(D_w1, lab, "marginal/W1", results)
        spectral(D_w1, lab, "marginal/W1", results)

        # 4) feature spaces: marginals concatenated; sqrt joint density
        feature_cluster(np.concatenate([marg_t, marg_p], axis=1), lab,
                        "features/(theta,tau)-marginals", results)
        feature_cluster(np.sqrt(masses), lab, "features/sqrt-joint", results)

        # 5) empirical ceiling: per-protein histograms of the DATA itself
        th = df.theta.to_numpy(np.float64) * 2 * np.pi / 360
        ta = df.tau.to_numpy(np.float64) * 2 * np.pi / 360
        xi = np.clip(((th + np.pi) / (2 * np.pi) * g).astype(int), 0, g - 1)
        yi = np.clip(((ta + np.pi) / (2 * np.pi) * g).astype(int), 0, g - 1)
        pid = df.pdb_id.map(order).to_numpy()
        hist = np.zeros((n, g * g), np.float64)
        np.add.at(hist, (pid, xi * g + yi), 1.0)
        nz = hist.sum(1, keepdims=True)
        hist = np.where(nz > 0, hist / np.maximum(nz, 1), 1.0 / (g * g))
        hist = hist + 1e-3
        hist /= hist.sum(1, keepdims=True)
        hierarchical(hellinger_matrix(hist), lab, "empirical/Hellinger", results)
        spectral(hellinger_matrix(hist), lab, "empirical/Hellinger", results)
        print(f"    hist sparsity: {np.mean(hist.sum(1) < g*g):.2f} proteins with <{g*g} bins filled")
        print(f"    label marginal: {np.bincount(lab).tolist()}")

        # 6) embedding space (conditional) + random control
        flow, emb, cfg = build_model(os.path.join(ROOT, TIER_RUNS[tier]), "cpu")
        E = emb.weight.detach().numpy()
        feature_cluster(E, lab, "features/cond-embeddings", results)
        rng = np.random.default_rng(0)
        feature_cluster(rng.normal(size=E.shape), lab, "features/random (control)", results)

        # 7) unconditional model: degenerate by construction
        uck = torch.load(os.path.join(ROOT, TIER_RUNS[tier].replace("/cond/", "/uncond/")),
                         map_location="cpu", weights_only=False)
        print(f"  [unconditional] cond_dim={uck['cond_dim']}, no embedding table "
              f"-> ONE shared density; all C({n},2) distances are exactly 0;")
        print("      clustering is undefined for it (not merely 'worse').")

        res = pd.DataFrame(results)
        out = os.path.join(ROOT, "results", "clustering", f"variants_{tier}.csv")
        res.to_csv(out, index=False)
        sys.stdout.reconfigure(line_buffering=True)
        print(f"\\n  --- {tier} (k_true={k_true}, n={n}), ranked by ARI ---")
        print(res.sort_values("ARI", ascending=False).to_string(index=False), flush=True)
        print(f"  saved: {out}", flush=True)


if __name__ == "__main__":
    main()

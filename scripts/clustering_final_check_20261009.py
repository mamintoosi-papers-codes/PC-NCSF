r"""Final revision check for Reviewer 1, Comment 5 (2026-10-09).

Three read-only checks. No retraining, no overwrite of existing artifacts.
All new outputs go to results/clustering/final_check_2026-10-09/.

(A) Cache <-> protein-index alignment: recompute the per-protein density for one
    excluded protein per affected tier straight from the checkpoint and compare
    with the cached row. Proves masses[k] corresponds to proteins[k] and
    emb.weight[k] after any exclusion.

(B) Exclusion robustness: recompute Hellinger + Ward (k = n_categories) ARI/NMI
    and 5-NN retrieval after dropping the five multi-domain PDB entries
    (easy: 1jj2,1lt4,1nmu,1qbk ; challenging: 1ta3).

(C) 5-NN context: majority-class baseline + label-permutation null (500
    shuffles, imbalance-preserving) per tier, to interpret retrieval against
    tier-appropriate bases -- especially the imbalanced Challenging tier.

Estimated cost: < 2 minutes CPU, < 1 GB RAM (largest object is the 1590x1590
distance matrix, ~20 MB float64).

Usage (Windows): C:\programs\anaconda3\envs\pth\python.exe scripts/clustering_final_check_20261009.py
Usage (Linux):   /data/python-envs/pytorch/bin/python scripts/clustering_final_check_20261009.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scop_clustering import ROOT, TIER_RUNS, hellinger_matrix, load_proteins

def build_model_compat(ckpt_path, device="cpu"):
    """Build NCSF + embedding, adapting old/new zuko `base.*` key names.

    Checkpoints were saved with old zuko (base._0/_1). Installed zuko 1.5.0
    here also expects base._0/_1, while newer zuko expects base.lower/upper.
    Detect the direction from the constructed module and map accordingly.
    """
    import zuko
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = cfg["network"]
    flow = zuko.flows.NCSF(2, ck["cond_dim"],
                           hidden_features=net["hidden_features"],
                           transforms=net["transforms"])
    sd = dict(ck["flow_state_dict"])
    exp = set(flow.state_dict().keys())
    if "base._0" in exp and "base.lower" in sd:
        sd["base._0"] = sd.pop("base.lower")
        sd["base._1"] = sd.pop("base.upper")
    elif "base.lower" in exp and "base._0" in sd:
        sd["base.lower"] = sd.pop("base._0")
        sd["base.upper"] = sd.pop("base._1")
    flow.load_state_dict(sd)

    emb = torch.nn.Embedding(
        num_embeddings=ck["embedding_state_dict"]["weight"].shape[0],
        embedding_dim=ck["cond_dim"])
    emb.load_state_dict(ck["embedding_state_dict"])
    return flow.to(device).eval(), emb.to(device).eval()


OUT = os.path.join(ROOT, "results", "clustering", "final_check_2026-10-09")
os.makedirs(OUT, exist_ok=True)

EXCLUDE = {
    "easy": ["1jj2", "1lt4", "1nmu", "1qbk"],
    "moderate": [],
    "hard": [],
    "challenging": ["1ta3"],
}
TIERS = ["easy", "moderate", "hard", "challenging"]


def knn_accuracy(D, labels, k=5):
    """Same convention as scripts/clustering_knn.py (diagonal excluded)."""
    n = len(labels)
    Dc = D.copy()
    np.fill_diagonal(Dc, np.inf)
    nn = np.argsort(Dc, axis=1)[:, :k]
    pred = np.empty(n, dtype=int)
    for i in range(n):
        pred[i] = np.bincount(labels[nn[i]]).argmax()
    return (pred == labels).mean()


def knn_permutation_null(D, labels, k=5, n_perm=500, seed=0):
    """Observed 5-NN vs imbalance-preserving label-permutation null.

    Neighbour indices depend only on D (fixed), so compute them once and only
    re-vote per permutation -- 500x faster than re-argorting each time.
    """
    n = len(labels)
    Dc = D.copy()
    np.fill_diagonal(Dc, np.inf)
    nn = np.argsort(Dc, axis=1)[:, :k]

    def vote(lab):
        pred = np.empty(n, dtype=int)
        for i in range(n):
            pred[i] = np.bincount(lab[nn[i]]).argmax()
        return (pred == lab).mean()

    obs = vote(labels)
    rng = np.random.default_rng(seed)
    null = np.array([vote(rng.permutation(labels)) for _ in range(n_perm)])
    p = (1.0 + (null >= obs).sum()) / (n_perm + 1.0)
    return obs, null.mean(), null.std(), p


# ----------------------------------------------------------------- (A)
def check_alignment(tier, proteins, angles, cond_idx, masses, grid):
    """Recompute one protein's density from the checkpoint; compare to cache."""
    cand = [p for p in EXCLUDE[tier] if p in proteins]
    if not cand:
        return None
    probe = cand[0]
    k = proteins.index(probe)
    flow, emb = build_model_compat(os.path.join(ROOT, TIER_RUNS[tier]), "cpu")
    dtheta = grid[1] - grid[0]
    th, ta = torch.meshgrid(torch.as_tensor(grid), torch.as_tensor(grid),
                            indexing="ij")
    pts = torch.stack([th.reshape(-1), ta.reshape(-1)], -1).float()
    with torch.no_grad():
        c = emb.weight[k].repeat(pts.shape[0], 1)
        m = flow(c).log_prob(pts).exp().numpy() * (dtheta ** 2)
    m = m / m.sum()
    cached = masses[k].reshape(-1).astype(np.float64)
    cached = cached / cached.sum()
    max_abs = float(np.abs(m - cached).max())
    print(f"  [align] {tier}/{probe} (index {k}): max|recomputed-cached| = "
          f"{max_abs:.2e}  -> {'OK' if max_abs < 1e-4 else 'MISMATCH'}")
    return {"tier": tier, "probe": probe, "index": k, "max_abs_diff": max_abs}


# ----------------------------------------------------------------- main
def main():
    grid = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    align_rows, excl_rows, knn_rows = [], [], []

    for tier in TIERS:
        angles, cond_idx, proteins, lab = load_proteins(tier)
        labels = pd.factorize(np.asarray(lab))[0]
        n = len(labels)
        masses = np.load(os.path.join(
            ROOT, "results", "clustering", f"masses_{tier}_g100.npy")
        ).reshape(n, -1)
        assert masses.shape[0] == n == len(proteins), "index alignment broken"

        print(f"\n=== {tier} ===")
        print(f"  n(proteins)={n}  categories={len(np.unique(labels))}")

        # (A) alignment proof
        r = check_alignment(tier, proteins, angles, cond_idx, masses, grid)
        if r:
            align_rows.append(r)

        # (B) exclusion
        drop = {proteins.index(p) for p in EXCLUDE[tier] if p in proteins}
        keep = np.array([i for i in range(n) if i not in drop])
        lab_kept = labels[keep]
        n_cls_kept = len(np.unique(lab_kept))
        assert n_cls_kept == 4, f"{tier}: categories lost after exclusion!"
        print(f"  excluding {len(drop)} proteins -> n={len(keep)}, "
              f"categories still {n_cls_kept}")

        D_full = hellinger_matrix(masses)
        D_keep = hellinger_matrix(masses[keep])

        for tag, D, labv in (("original", D_full, labels),
                             ("excluded", D_keep, lab_kept)):
            cond = squareform(np.clip(D, 0, None), checks=False)
            pred = fcluster(linkage(cond, method="ward"),
                           len(np.unique(labv)), "maxclust")
            ari = adjusted_rand_score(labv, pred)
            nmi = normalized_mutual_info_score(labv, pred)
            acc5 = knn_accuracy(D, labv, 5)
            sizes = np.bincount(pred)[1:].tolist()
            print(f"  [{tag:8s}] n={len(labv):5d} ARI={ari:.4f} NMI={nmi:.4f} "
                  f"5NN={acc5:.4f} sizes={sizes}")
            excl_rows.append({"tier": tier, "variant": tag, "n": len(labv),
                              "ARI": round(ari, 4), "NMI": round(nmi, 4),
                              "NN5": round(acc5, 4),
                              "cluster_sizes": str(sizes)})

        # (C) 5-NN context on the ORIGINAL set
        counts = np.bincount(labels, minlength=4)
        majority = counts.max() / counts.sum()
        obs, null_mu, null_sd, p = knn_permutation_null(D_full, labels, 5)
        print(f"  [5NN ctx ] observed={obs:.4f}  majority-class={majority:.4f} "
              f"perm-null={null_mu:.4f}±{null_sd:.4f}  p={p:.4g}")
        knn_rows.append({"tier": tier, "n": n,
                         "class_counts": str(counts.tolist()),
                         "NN5_observed": round(obs, 4),
                         "majority_class": round(majority, 4),
                         "perm_null_mean": round(null_mu, 4),
                         "perm_null_std": round(null_sd, 4),
                         "p_value": round(p, 6)})

    pd.DataFrame(align_rows).to_csv(os.path.join(OUT, "alignment_check.csv"),
                                    index=False)
    pd.DataFrame(excl_rows).to_csv(os.path.join(OUT, "exclusion_test.csv"),
                                   index=False)
    pd.DataFrame(knn_rows).to_csv(os.path.join(OUT, "knn_context.csv"),
                                  index=False)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

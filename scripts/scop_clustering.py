"""Reproduce the SCOP density-based clustering experiment (paper Section 3.5 / Table 8).

Pipeline (as described in the manuscript):
  1. Load a trained protein-conditional PC-NCSF checkpoint for a SCOP tier.
  2. Evaluate each protein's density on a uniform 100x100 grid over [-pi, pi)^2.
  3. Compute pairwise Hellinger distances between the protein densities.
  4. Ward hierarchical clustering, cut at k = (number of SCOP categories).
  5. Evaluate against SCOP ground-truth labels with ARI and NMI.

Usage:
  /data/python-envs/pytorch/bin/python scripts/scop_clustering.py --tier easy
  /data/python-envs/pytorch/bin/python scripts/scop_clustering.py --tier all
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import zuko
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIER_RUNS = {
    # tier: run folder under runs/ (matches the paper: bs512, ep20, emb dim 8)
    "easy": "runs/scop_easy/ep20-bs512/cond/best_flow.pt",
    "moderate": "runs/scop_moderate/ep20-bs512/cond/best_flow.pt",
    "hard": "runs/scop_hard/ep20-bs512/cond/best_flow.pt",
    "challenging": "runs/scop_challenging/ep20-bs512/cond/best_flow.pt",
}


def load_proteins(tier):
    """Replicates the conditioning of fff.data.scop.get_scop_dataset.

    Returns (angles_rad [N,2], cond_idx [N], proteins list, labels list)
    where proteins[i] is the pdb_id of condition index i and labels[i] is its
    SCOP category (majority vote over residues).
    """
    df = pd.read_csv(os.path.join(ROOT, "SCOP", tier, "data.csv"))
    df = df.dropna(subset=["theta", "tau"]).reset_index(drop=True)
    cond_idx = pd.factorize(df["pdb_id"])[0]

    angles = df[["theta", "tau"]].to_numpy(dtype=np.float32) * (2 * np.pi / 360.0)

    # condition index -> pdb_id (order of first appearance, as factorize does)
    proteins = list(pd.unique(df["pdb_id"]))

    # ground-truth label per protein: majority category (4 pdb_ids in 'easy'
    # and one in 'challenging' span more than one category)
    lab = []
    for pdb in proteins:
        cats = df.loc[df["pdb_id"] == pdb, "category"]
        lab.append(cats.value_counts().idxmax())
    return angles, cond_idx, proteins, lab


def build_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = cfg["network"]
    flow = zuko.flows.NCSF(
        2, ck["cond_dim"],
        hidden_features=net["hidden_features"],
        transforms=net["transforms"],
    )
    # zuko renamed the Uniform base buffers: base._0/base._1 -> base.lower/base.upper
    sd = dict(ck["flow_state_dict"])
    if "base._0" in sd and "base.lower" not in sd:
        sd["base.lower"] = sd.pop("base._0")
        sd["base.upper"] = sd.pop("base._1")
    flow.load_state_dict(sd)

    emb = torch.nn.Embedding(
        num_embeddings=ck["embedding_state_dict"]["weight"].shape[0],
        embedding_dim=ck["cond_dim"],
    )
    emb.load_state_dict(ck["embedding_state_dict"])

    flow = flow.to(device).eval()
    emb = emb.to(device).eval()
    return flow, emb, cfg


@torch.no_grad()
def protein_densities(flow, emb, angles, cond_idx, n_proteins, grid, device,
                      protein_batch=16):
    """Evaluate per-protein densities on the grid.

    Returns array [n_proteins, G, G] of probability masses (rows sum to ~1).
    """
    dtheta = grid[1] - grid[0]
    th, ta = torch.meshgrid(torch.as_tensor(grid), torch.as_tensor(grid), indexing="ij")
    pts = torch.stack([th.reshape(-1), ta.reshape(-1)], dim=-1).to(device=device, dtype=torch.float32)  # [G*G, 2]
    n_pts = pts.shape[0]

    x = torch.as_tensor(angles, device=device)
    out = np.empty((n_proteins, n_pts), dtype=np.float32)

    for i in range(n_proteins):
        c = emb.weight[i].repeat(n_pts, 1)
        logp = flow(c).log_prob(pts)                      # [G*G]
        mass = logp.exp().cpu().numpy() * (dtheta ** 2)   # -> probability mass
        total = mass.sum()
        if total > 0:
            mass = mass / total
        out[i] = mass
        if (i + 1) % 100 == 0 or i + 1 == n_proteins:
            print(f"    densities {i + 1}/{n_proteins}", end="\r", flush=True)
    print()
    return out.reshape(n_proteins, len(grid), len(grid))


def hellinger_matrix(masses):
    """Pairwise Hellinger distance from probability masses on a grid.

    H^2(p,q) = 0.5 * sum_i (sqrt(p_i) - sqrt(q_i))^2
    """
    a = np.sqrt(masses.reshape(masses.shape[0], -1)).astype(np.float64)
    sq = (a ** 2).sum(axis=1, keepdims=True)
    gram = a @ a.T
    h2 = 0.5 * np.maximum(sq + sq.T - 2 * gram, 0.0)
    np.fill_diagonal(h2, 0.0)
    return np.sqrt(h2)


def run_tier(tier, grid_size, device, save_dir=None):
    angles, cond_idx, proteins, labels = load_proteins(tier)
    n_proteins = len(proteins)
    labels = pd.factorize(np.asarray(labels))[0]
    n_classes = len(np.unique(labels))
    print(f"[{tier}] residues={len(angles)} proteins={n_proteins} categories={n_classes}")

    flow, emb, cfg = build_model(os.path.join(ROOT, TIER_RUNS[tier]), device)

    grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    cache = os.path.join(ROOT, "results", "clustering", f"masses_{tier}_g{grid_size}.npy")
    if os.path.exists(cache):
        masses = np.load(cache)
        print(f"[{tier}] loaded cached densities {masses.shape}")
    else:
        masses = protein_densities(flow, emb, angles, cond_idx, n_proteins, grid, device)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, masses)
    print(f"[{tier}] density grid: {masses.shape}, mass sum mean={masses.sum(axis=(1,2)).mean():.6f}")

    dist = hellinger_matrix(masses)
    condensed = squareform(dist, checks=False)

    results = []
    for method in ("ward", "average", "complete", "single"):
        Z = linkage(condensed, method=method)
        pred = fcluster(Z, t=n_classes, criterion="maxclust")
        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)
        results.append({"tier": tier, "linkage": method, "k": n_classes,
                        "ARI": round(ari, 4), "NMI": round(nmi, 4),
                        "n_proteins": n_proteins, "grid": grid_size})
        print(f"[{tier}] linkage={method:9s} ARI={ari:.4f} NMI={nmi:.4f}")

    # sanity baseline: random labels
    rng = np.random.default_rng(0)
    rand_pred = rng.permutation(np.tile(np.arange(n_classes),
                                        int(np.ceil(n_proteins / n_classes))))[:n_proteins]
    print(f"[{tier}] random ARI={adjusted_rand_score(labels, rand_pred):.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(
            os.path.join(save_dir, f"scop_clustering_{tier}_g{grid_size}.csv"),
            index=False,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all",
                    choices=list(TIER_RUNS) + ["all"])
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "results", "clustering"))
    args = ap.parse_args()

    tiers = list(TIER_RUNS) if args.tier == "all" else [args.tier]
    all_results = []
    for tier in tiers:
        all_results += run_tier(tier, args.grid, args.device, args.save_dir)

    df = pd.DataFrame(all_results)
    out = os.path.join(args.save_dir, f"scop_clustering_summary_g{args.grid}.csv")
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== Summary (Ward, as in the manuscript) ===")
    print(df[df.linkage == "ward"].to_string(index=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()

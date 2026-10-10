"""
scop_pcncsf_clustering.py

Real-SCOP density-based clustering for PC-NCSF, for Reviewer 1's requested
real-data clustering validation. Uses the SAME PC-NCSF checkpoints already
trained for Section 3.3/Table 6/Figures 6-9 (no retraining, no change to the
SCOP extraction pipeline, no change to the model) and applies the SAME
Hellinger-distance + Ward-clustering pipeline already used for the synthetic
study in Section 3.2.3 (Table 5), now on the four real SCOP tiers.

Checkpoint loading and the protein-label -> category mapping are copied
EXACTLY from report_generator.py (the only place this convention is written
down), so results line up with what your existing plots/tables already show:
  - checkpoint keys: config, cond_dim, flow_state_dict, embedding_state_dict
    (embedding size is read from the weight tensor's shape, not re-derived)
  - protein identity for embedding label i = the i-th unique value (in first-
    appearance order, via pandas.factorize) of the SCOP CSV's protein_name
    column, falling back to pdb_id then domain_id if protein_name is absent.
    NOTE this is FIRST-APPEARANCE order, unlike the torus mapping which was
    alphabetically sorted -- do not mix the two conventions up.
  - a protein's category (ground truth for ARI/NMI) is that unique value's
    first row's `category` column (falling back to `class`).

No change is made to how (theta, tau) were produced or how PC-NCSF was
trained: the flow is evaluated on the same [-pi, pi) x [-pi, pi) domain it was
trained on (zuko's CircularRQSTransform wraps any real input into range, so
the exact representative interval used for the grid does not matter).

Usage:
    python scop_pcncsf_clustering.py \
        --runs-dir runs --scop-root . \
        --tiers scop_easy scop_moderate scop_hard scop_challenging \
        --grid-size 100 --output-csv pcncsf_scop_clustering.csv

If a tier's checkpoint cannot be found automatically, the script lists the
run folders it did find under runs/<tier>/ so you can pass an explicit path
with --ckpt <tier>=<path/to/best_flow.pt> (repeatable).
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zuko
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors report_generator.py exactly)
# ---------------------------------------------------------------------------
def load_pcncsf_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    cond_dim = ckpt["cond_dim"]

    flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)
    if "flow_state_dict" in ckpt:
        flow.load_state_dict(ckpt["flow_state_dict"])
    elif "state_dict" in ckpt:
        flow.load_state_dict(ckpt["state_dict"])
    else:
        raise KeyError(f"No flow state_dict found in checkpoint: {list(ckpt.keys())}")
    flow.eval()

    if "embedding_state_dict" not in ckpt:
        raise KeyError(f"No embedding_state_dict in checkpoint {ckpt_path} -- "
                        f"is this an UNCONDITIONAL run? This script needs PC-NCSF (conditional).")
    emb_sd = ckpt["embedding_state_dict"]
    weight = None
    for v in emb_sd.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            weight = v
            break
    if weight is None:
        raise KeyError(f"Could not find a 2-D weight tensor in embedding_state_dict of {ckpt_path}")
    n_proteins, emb_dim = weight.shape
    embedding = nn.Embedding(n_proteins, emb_dim).to(device)
    embedding.load_state_dict({"weight": weight})
    embedding.eval()

    return flow, embedding, config


def find_checkpoint(runs_dir, tier, explicit_path=None):
    if explicit_path is not None:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"--ckpt path for {tier} does not exist: {explicit_path}")
        return explicit_path
    candidates = sorted(glob.glob(os.path.join(runs_dir, tier, "*", "best_flow.pt")))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No best_flow.pt found under {os.path.join(runs_dir, tier)}/*/. "
            f"Pass the exact path with --ckpt {tier}=/path/to/best_flow.pt")
    raise RuntimeError(
        f"Multiple checkpoints found for {tier}, please disambiguate with "
        f"--ckpt {tier}=<one of these>:\n  " + "\n  ".join(candidates))


# ---------------------------------------------------------------------------
# Protein label <-> SCOP category mapping (mirrors report_generator.py exactly)
# ---------------------------------------------------------------------------
def protein_category_mapping(scop_csv, expected_n=None):
    """Picks the identifier column by EVIDENCE rather than a fixed priority:
    if expected_n (the checkpoint's embedding row count) is given, try
    protein_name / domain_id / pdb_id in that order and use the first whose
    number of unique values matches expected_n exactly. This matters because
    protein_name can collide across genuinely different domains (e.g. many
    unrelated PDB entries named "LYSOZYME"), which report_generator.py's
    column priority does not account for. If none match, falls back to the
    old priority order and prints all three counts so the mismatch is visible
    rather than silently trusted."""
    df = pd.read_csv(scop_csv)
    candidates = [c for c in ("protein_name", "domain_id", "pdb_id") if c in df.columns]
    if not candidates:
        id_cols = [c for c in df.columns if "protein" in c.lower() or "id" in c.lower()]
        if not id_cols:
            raise ValueError(f"No protein identifier column found in {scop_csv}")
        candidates = id_cols

    counts = {c: df[c].nunique() for c in candidates}
    print(f"  candidate id columns and unique counts: {counts}"
          + (f"  (checkpoint expects {expected_n})" if expected_n is not None else ""))

    prot_col = None
    if expected_n is not None:
        for c in candidates:
            if counts[c] == expected_n:
                prot_col = c
                break
    if prot_col is None:
        prot_col = candidates[0]
        if expected_n is not None:
            print(f"  WARNING: no candidate column has exactly {expected_n} unique values; "
                  f"defaulting to '{prot_col}' ({counts[prot_col]} unique) -- verify this manually.")
    else:
        print(f"  using '{prot_col}' as the protein identifier (matches checkpoint exactly)")

    codes, uniques = pd.factorize(df[prot_col])
    categories = []
    ambiguous = []
    for i in range(len(uniques)):
        mask = codes == i
        if "category" in df.columns:
            cat_values = df.loc[mask, "category"]
        elif "class" in df.columns:
            cat_values = df.loc[mask, "class"]
        else:
            cat_values = pd.Series([""])
        if cat_values.nunique() > 1:
            ambiguous.append((uniques[i], sorted(cat_values.unique().tolist())))
        categories.append(str(cat_values.iloc[0]))

    if ambiguous:
        print(f"  WARNING: {len(ambiguous)} value(s) of '{prot_col}' map to more than one category "
              f"(ground truth is ambiguous for these -- using the first-seen category). "
              f"e.g. {ambiguous[:3]}")
    return categories, prot_col


# ---------------------------------------------------------------------------
# Per-protein density on a fixed grid, then Hellinger distance + Ward clustering
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_protein_density_grid(flow, embedding, n_proteins, device, grid_size, batch_proteins):
    lo, hi = -math.pi, math.pi
    step = (hi - lo) / grid_size
    g = torch.linspace(lo, hi, grid_size + 1)[:-1] + step / 2
    G1, G2 = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([G1.reshape(-1), G2.reshape(-1)], dim=-1).to(device)
    n_grid = grid.shape[0]
    cell_area = step ** 2

    densities = torch.zeros(n_proteins, n_grid)
    for start in range(0, n_proteins, batch_proteins):
        end = min(start + batch_proteins, n_proteins)
        b = end - start
        labels = torch.arange(start, end, device=device)
        c = embedding(labels)
        c_exp = c.unsqueeze(1).expand(b, n_grid, c.shape[-1]).reshape(b * n_grid, -1)
        x_exp = grid.unsqueeze(0).expand(b, n_grid, 2).reshape(b * n_grid, 2)
        log_p = flow(c_exp).log_prob(x_exp).reshape(b, n_grid)
        densities[start:end] = torch.exp(log_p).cpu()
    return densities, cell_area


def hellinger_distance_matrix(densities, cell_area):
    """Eq. (26) of the paper: H(p,q) = sqrt(0.5 * sum((sqrt(p)-sqrt(q))^2 * cell_area)).
    Computed via pdist on pre-scaled sqrt-densities so ordinary Euclidean
    distance IS the Hellinger distance (avoids an O(n^2 * n_grid) Python loop)."""
    sqrt_p = torch.sqrt(torch.clamp(densities, min=0.0)).numpy()
    scaled = sqrt_p * math.sqrt(0.5 * cell_area)
    condensed = pdist(scaled, metric="euclidean")
    return condensed  # condensed form, for scipy.linkage directly


def cluster_and_score(condensed_dist, categories, k, method="ward"):
    Z = linkage(condensed_dist, method=method)
    pred = fcluster(Z, t=k, criterion="maxclust")
    cat_to_id = {c: i for i, c in enumerate(sorted(set(categories)))}
    true_ids = [cat_to_id[c] for c in categories]
    ari = adjusted_rand_score(true_ids, pred)
    nmi = normalized_mutual_info_score(true_ids, pred)
    return ari, nmi, pred


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv=None):
    """argv=None reads sys.argv (normal command-line use). Pass an explicit list
    of strings, e.g. main(["--tiers", "scop_easy"]), to call this from a Jupyter
    cell without argparse trying (and failing) to parse the kernel's own argv."""
    ap = argparse.ArgumentParser(description="Real-SCOP PC-NCSF density-based clustering (ARI/NMI)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--scop-root", default=".", help="Directory containing <scop-folder-name>/<tier>/data.csv")
    ap.add_argument("--scop-folder-name", default="Scop_data",
                    help="Name of the folder holding the per-tier SCOP subfolders (default: Scop_data)")
    ap.add_argument("--tiers", nargs="+", default=["scop_easy", "scop_moderate", "scop_hard", "scop_challenging"])
    ap.add_argument("--ckpt", action="append", default=[],
                    help="Explicit override 'tier=path/to/best_flow.pt' (repeatable)")
    ap.add_argument("--grid-size", type=int, default=100)
    ap.add_argument("--batch-proteins", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-csv", default="pcncsf_scop_clustering.csv")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.ckpt:
        if "=" not in item:
            print(f"--ckpt must be 'tier=path', got: {item}", file=sys.stderr)
            sys.exit(1)
        t, p = item.split("=", 1)
        overrides[t] = p

    results = []
    for tier in args.tiers:
        print(f"\n=== {tier} ===")
        subset = tier.split("scop_")[-1] if tier.startswith("scop_") else tier
        scop_csv = os.path.join(args.scop_root, args.scop_folder_name, subset, "data.csv")
        if not os.path.exists(scop_csv):
            print(f"  SCOP CSV not found: {scop_csv} -- skipping")
            continue

        try:
            ckpt_path = find_checkpoint(args.runs_dir, tier, overrides.get(tier))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  {e}\n  skipping {tier}")
            continue
        print(f"  checkpoint: {ckpt_path}")

        flow, embedding, config = load_pcncsf_checkpoint(ckpt_path, args.device)
        n_proteins = embedding.num_embeddings
        categories, prot_col = protein_category_mapping(scop_csv, expected_n=n_proteins)

        if len(categories) != n_proteins:
            print(f"  WARNING: {len(categories)} unique '{prot_col}' values in the CSV but the "
                  f"checkpoint's embedding has {n_proteins} rows -- the mapping below is UNRELIABLE. "
                  f"Do not trust this tier's ARI/NMI until this is resolved.")
            continue

        k = len(set(categories))
        m = n_proteins
        print(f"  m (unique proteins) = {m}, k (categories) = {k}")

        densities, cell_area = per_protein_density_grid(
            flow, embedding, n_proteins, args.device, args.grid_size, args.batch_proteins)
        condensed_dist = hellinger_distance_matrix(densities, cell_area)
        row = {"tier": tier, "m": m, "k": k, "checkpoint": ckpt_path}
        for method in ("ward", "average", "complete"):
            ari, nmi, pred = cluster_and_score(condensed_dist, categories, k, method=method)
            sizes = sorted(pd.Series(pred).value_counts().tolist(), reverse=True)
            print(f"  [{method:8s}] ARI = {ari:.3f}   NMI = {nmi:.3f}   cluster sizes = {sizes}")
            row[f"ARI_{method}"] = ari
            row[f"NMI_{method}"] = nmi

        results.append(row)

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved summary to {args.output_csv}")
        print(out_df.to_string(index=False))
    else:
        print("\nNo tier produced a result -- nothing saved.")


if __name__ == "__main__":
    main()

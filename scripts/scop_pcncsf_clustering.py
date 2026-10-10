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
        sd = ckpt["flow_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        raise KeyError(f"No flow state_dict found in checkpoint: {list(ckpt.keys())}")
    # zuko renamed the Uniform base buffers: base._0/base._1 -> base.lower/base.upper.
    # Checkpoints trained with the older zuko (Windows run) still carry the old
    # names; remap so the script loads on this machine's zuko 1.6.0. Both
    # directions are handled so the script also runs on a fresh checkout with
    # an old zuko.
    sd = dict(sd)
    if "base._0" in sd and "base.lower" not in sd:
        sd["base.lower"] = sd.pop("base._0")
        sd["base.upper"] = sd.pop("base._1")
    elif "base.lower" in sd and "base._0" not in sd:
        sd["base._0"] = sd.pop("base.lower")
        sd["base._1"] = sd.pop("base.upper")
    flow.load_state_dict(sd)
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
    candidates = sorted(glob.glob(os.path.join(runs_dir, tier, "**", "best_flow.pt"), recursive=True))
    # The training layout nests checkpoints as runs/<tier>/<exp>/cond/best_flow.pt
    # (conditional PC-NCSF) and .../uncond/best_flow.pt (unconditional baseline).
    # This script REQUIRES the conditional run (it needs embedding_state_dict),
    # so prefer checkpoints under a directory literally named 'cond'.
    cond = [c for c in candidates if "cond" in os.path.normpath(c).split(os.sep)
            and "uncond" not in os.path.normpath(c).split(os.sep)]
    if len(cond) == 1:
        return cond[0]
    if len(cond) > 1:
        # Deterministic default: prefer the primary run shared by every tier
        # (runs/<tier>/ep20-bs512/cond/best_flow.pt), so the script runs
        # unattended. Override with --ckpt <tier>=<path> to use another run.
        preferred = [c for c in cond if os.sep + "ep20-bs512" + os.sep in os.sep + os.path.normpath(c) + os.sep]
        if len(preferred) == 1:
            others = [c for c in cond if c not in preferred]
            print(f"  note: multiple conditional checkpoints for {tier}; defaulting to "
                  f"{preferred[0]}\n        (override with --ckpt {tier}=<path>; other option(s): "
                  f"{', '.join(others)})")
            return preferred[0]
        raise RuntimeError(
            f"Multiple conditional checkpoints found for {tier}, please disambiguate with "
            f"--ckpt {tier}=<one of these>:\n  " + "\n  ".join(cond))
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
# Protein label <-> SCOP category mapping (must mirror fff.data.scop.get_scop_dataset)
# ---------------------------------------------------------------------------
def protein_category_mapping(scop_csv, expected_n=None):
    """Return categories in the *same order* as the loader's integer conditions.

    Training calls load_dataset(..., condition_on="residue"). The SCOP loader
    therefore uses a CSV column named 'residue' if present, otherwise pdb_id,
    otherwise domain_id (see fff/data/scop.py). It drops rows with missing
    angles BEFORE factorizing identifiers. Do not infer the identifier column
    from unique counts: equal counts do not prove equal identities/order.

    IDs that map to more than one SCOP category are returned in
    ambiguous_indices and must be excluded from label-based scoring.
    """
    df = pd.read_csv(scop_csv)
    if "theta" in df.columns and "tau" in df.columns:
        angle_cols = ["theta", "tau"]
    elif "phi" in df.columns and "psi" in df.columns:
        angle_cols = ["phi", "psi"]
    else:
        angle_cols = [c for c in df.columns
                      if any(k in c.lower() for k in ("phi", "psi", "theta", "tau"))][:2]
        if len(angle_cols) != 2:
            raise ValueError(f"Could not identify two angle columns in {scop_csv}")
    df = df.dropna(subset=angle_cols).reset_index(drop=True)

    # Exact identifier-selection logic from get_scop_dataset(...,
    # condition_on="residue").
    if "residue" in df.columns:
        prot_col = "residue"
    elif "pdb_id" in df.columns:
        prot_col = "pdb_id"
    elif "domain_id" in df.columns:
        prot_col = "domain_id"
    else:
        id_cols = [c for c in df.columns if "protein" in c.lower() or "id" in c.lower()]
        if not id_cols:
            raise ValueError(f"No protein identifier column found in {scop_csv}")
        prot_col = id_cols[0]
    codes, uniques = pd.factorize(df[prot_col])
    counts = np.bincount(codes, minlength=len(uniques))
    if expected_n is not None and len(uniques) != expected_n:
        print(f"  ERROR: loader-compatible '{prot_col}' has {len(uniques)} IDs, "
              f"but checkpoint embedding has {expected_n} rows.")

    cat_col = "category" if "category" in df.columns else "class"
    categories = []
    ambiguous_indices = []
    ambiguous = []
    for i in range(len(uniques)):
        mask = codes == i
        cat_values = df.loc[mask, cat_col].dropna().astype(str) if cat_col in df.columns else pd.Series([""])
        if cat_values.nunique() > 1:
            ambiguous_indices.append(i)
            ambiguous.append((uniques[i], sorted(cat_values.unique().tolist())))
        categories.append(str(cat_values.iloc[0]) if len(cat_values) else "UNKNOWN")

    if ambiguous:
        print(f"  WARNING: {len(ambiguous)} '{prot_col}' IDs map to multiple SCOP categories; "
              f"these IDs will be excluded from ARI/NMI. Examples: {ambiguous[:3]}")
    return categories, prot_col, ambiguous_indices


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
    ap.add_argument("--scop-folder-name", default="SCOP",
                    help="Name of the folder holding the per-tier SCOP subfolders (default: SCOP)")
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
        categories, prot_col, ambiguous_indices = protein_category_mapping(scop_csv, expected_n=n_proteins)

        if len(categories) != n_proteins:
            print(f"  ERROR: {len(categories)} loader-compatible '{prot_col}' IDs but "
                  f"{n_proteins} checkpoint embedding rows. Skipping this tier.")
            continue

        keep_indices = [i for i in range(n_proteins) if i not in set(ambiguous_indices)]
        if len(keep_indices) < 2:
            print("  Fewer than two unambiguous proteins remain; skipping.")
            continue
        categories = [categories[i] for i in keep_indices]
        if ambiguous_indices:
            print(f"  Scoring {len(keep_indices)} unambiguous proteins; "
                  f"excluded {len(ambiguous_indices)} ambiguous IDs.")

        k = len(set(categories))
        m = len(categories)
        print(f"  m (unique proteins) = {m}, k (categories) = {k}")

        densities, cell_area = per_protein_density_grid(
            flow, embedding, n_proteins, args.device, args.grid_size, args.batch_proteins)
        condensed_dist = hellinger_distance_matrix(densities, cell_area)
        full_dist = squareform(condensed_dist)
        condensed_dist = squareform(full_dist[np.ix_(keep_indices, keep_indices)], checks=False)
        row = {"tier": tier, "m": m, "k": k, "checkpoint": ckpt_path}
        row["n_ambiguous_excluded"] = len(ambiguous_indices)
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

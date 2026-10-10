"""
scop_clustering_controls.py

Control analyses for the real-SCOP density-based clustering result
(scop_pcncsf_clustering.py). They answer, with the SAME trained PC-NCSF
checkpoints and the SAME data:

  0. Is the embedding-label -> SCOP-category alignment actually correct?
     (If it were wrong, ARI ~ 0 would be a bookkeeping artefact, not a finding.)
  1. Is each tier's ARI / NMI better than chance?  Permutation test: the true
     labels are shuffled (class sizes kept) against the FIXED predicted
     clusters; p = (1 + #{null >= observed}) / (1 + n_perm).
  2. 1-nearest-neighbour accuracy on the Hellinger distance matrix. It does not
     depend on Ward linkage (so it is not affected by one giant cluster).
     Chance level = sum_c n_c (n_c - 1) / (n (n - 1)); p-value by permutation.
  3. A representation-free baseline: each protein's density is a circular
     Gaussian KDE of its OWN (theta, tau) residues (same angle conversion the
     model was trained on), then the identical Hellinger -> Ward -> ARI/NMI
     pipeline. If the baseline is also near chance, the limitation is in the
     (theta, tau) representation / labels, not in PC-NCSF. If the baseline is
     better, that is reported as such.

Choices fixed IN ADVANCE (they were not tuned on the outcome):
  * KDE bandwidth per coordinate d and protein i:  h = n_i^(-1/6) * s_d * m,
    where s_d is the pooled circular standard deviation sqrt(-2 ln R_d) of
    coordinate d (Scott's rule for 2-D data) and m = 1. The multipliers 0.5 and
    2 are reported as a sensitivity check; none is selected after the fact.
  * Ward linkage, k = number of true categories, grid 100 x 100 (as in
    scop_pcncsf_clustering.py).
  * Every row of the output table is reported, whatever it shows.

The KDE baseline uses ALL residues of each protein (train + validation). The
training split seed is not stored explicitly in the checkpoint, so exact train-only
KDE reconstruction is not claimed; treat this as a transductive baseline, not a
strictly data-matched comparison with PC-NCSF.

Usage (command line):
    python scop_clustering_controls.py --runs-dir runs --scop-root . \
        --scop-folder-name SCOP --loader-root . \
        --tiers scop_easy scop_moderate scop_hard scop_challenging

Usage (Jupyter): from scop_clustering_controls import main; main([...same flags...])
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scop_pcncsf_clustering import (
    find_checkpoint,
    load_pcncsf_checkpoint,
    per_protein_density_grid,
    hellinger_distance_matrix,
    cluster_and_score,
)

TWO_PI = 2 * math.pi


def wrap_to_pi(x):
    return (x + math.pi) % TWO_PI - math.pi


# ---------------------------------------------------------------------------
# Data: exactly what the training script fed to the model
# ---------------------------------------------------------------------------
def get_loader_arrays(tier, loader_root="."):
    """Residue angles in the model's input domain and the model's integer protein
    labels, for train + validation together. Uses the same loader and the same
    convert_to_angles call as circularspline_protein_embedding_c.py."""
    from fff.data import load_dataset
    from fff.evaluate.tori import convert_to_angles

    datasets = load_dataset(tier, root=loader_root, condition_on="residue")
    xs, cs = [], []
    for ds in datasets:
        x, c = ds[:][0], ds[:][1]
        xs.append(convert_to_angles(x).detach().cpu().float())
        cs.append(c.detach().cpu().long())
    return torch.cat(xs), torch.cat(cs)


# ---------------------------------------------------------------------------
# 0. Label alignment check (label index -> SCOP category)
# ---------------------------------------------------------------------------
def build_label_mapping(csv_path, loader_labels, min_match=0.9):
    """Reconstruct the exact identifier ordering used by fff.data.scop.

    Training passes condition_on='residue'; the SCOP loader uses that column
    only if it exists, otherwise pdb_id, then domain_id. It drops missing-angle
    rows before factorizing, so this function must do the same. Ambiguous IDs
    are returned separately and excluded from label-based scoring.
    """
    df = pd.read_csv(csv_path)
    if "theta" in df.columns and "tau" in df.columns:
        angle_cols = ["theta", "tau"]
    elif "phi" in df.columns and "psi" in df.columns:
        angle_cols = ["phi", "psi"]
    else:
        angle_cols = [c for c in df.columns
                      if any(k in c.lower() for k in ("phi", "psi", "theta", "tau"))][:2]
        if len(angle_cols) != 2:
            raise ValueError(f"Could not identify two angle columns in {csv_path}")
    df = df.dropna(subset=angle_cols).reset_index(drop=True)
    if "residue" in df.columns:
        col = "residue"
    elif "pdb_id" in df.columns:
        col = "pdb_id"
    elif "domain_id" in df.columns:
        col = "domain_id"
    else:
        id_cols = [c for c in df.columns if "protein" in c.lower() or "id" in c.lower()]
        if not id_cols:
            raise ValueError(f"No protein identifier column found in {csv_path}")
        col = id_cols[0]
    n_labels = int(loader_labels.max()) + 1
    counts_loader = np.bincount(loader_labels.numpy(), minlength=n_labels)
    codes, uniques = pd.factorize(df[col])
    counts_csv = np.bincount(codes, minlength=len(uniques))
    frac = float(np.mean(counts_csv == counts_loader)) if len(uniques) == n_labels else 0.0
    info = {"candidates": [(col, "first-appearance", len(uniques), frac)],
            "n_labels": n_labels, "accepted": None}
    if len(uniques) != n_labels or not np.array_equal(counts_csv, counts_loader):
        info["accepted"] = None
        return None, info, []

    cat_col = "category" if "category" in df.columns else "class"
    categories, ambiguous_indices = [], []
    for i in range(n_labels):
        vals = df.loc[codes == i, cat_col].dropna().astype(str) if cat_col in df.columns else pd.Series(["UNKNOWN"])
        if vals.nunique() > 1:
            ambiguous_indices.append(i)
        categories.append(str(vals.iloc[0]) if len(vals) else "UNKNOWN")
    info.update({"accepted": (col, "first-appearance"), "match": frac,
                 "ambiguous": len(ambiguous_indices)})
    return categories, info, ambiguous_indices


def subset_condensed_distance(condensed, keep_indices):
    """Subset a SciPy condensed distance vector by retained row indices."""
    full = squareform(condensed)
    return squareform(full[np.ix_(keep_indices, keep_indices)], checks=False)


# ---------------------------------------------------------------------------
# 3. Empirical KDE baseline
# ---------------------------------------------------------------------------
def pooled_circular_std(x):
    R = torch.sqrt(torch.cos(x).mean(0) ** 2 + torch.sin(x).mean(0) ** 2)
    return torch.sqrt(-2 * torch.log(torch.clamp(R, 1e-6, 1.0))).clamp(max=2.0)


@torch.no_grad()
def kde_densities(angles, labels, n_proteins, grid_size, bw_mult):
    """Circular Gaussian KDE of each protein's own residues on the same grid used
    for the model. Returns (densities (n_proteins, grid_size**2), cell_area)."""
    step = TWO_PI / grid_size
    g = torch.linspace(-math.pi, math.pi, grid_size + 1)[:-1] + step / 2
    s = pooled_circular_std(angles)
    order = torch.argsort(labels)
    sorted_angles = angles[order]
    counts = torch.bincount(labels, minlength=n_proteins)
    chunks = torch.split(sorted_angles, counts.tolist())
    dens = torch.zeros(n_proteins, grid_size * grid_size)
    for i, xi in enumerate(chunks):
        n = xi.shape[0]
        if n == 0:
            dens[i] = 1.0 / TWO_PI ** 2
            continue
        h = ((n ** (-1.0 / 6.0)) * s * bw_mult).clamp(min=0.05)
        d1 = wrap_to_pi(g[:, None] - xi[None, :, 0])
        d2 = wrap_to_pi(g[:, None] - xi[None, :, 1])
        K1 = torch.exp(-0.5 * (d1 / h[0]) ** 2) / (math.sqrt(TWO_PI) * h[0])
        K2 = torch.exp(-0.5 * (d2 / h[1]) ** 2) / (math.sqrt(TWO_PI) * h[1])
        D = (K1 @ K2.T) / n
        D = D / (D.sum() * step ** 2)  # remove truncation error: integrates to 1
        dens[i] = D.reshape(-1)
    return dens, step ** 2


# ---------------------------------------------------------------------------
# 1 + 2. Permutation tests and 1-NN
# ---------------------------------------------------------------------------
def permutation_test_ari_nmi(true_ids, pred, n_perm, rng):
    ari = adjusted_rand_score(true_ids, pred)
    nmi = normalized_mutual_info_score(true_ids, pred)
    ari_null = np.empty(n_perm)
    nmi_null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(true_ids)
        ari_null[b] = adjusted_rand_score(p, pred)
        nmi_null[b] = normalized_mutual_info_score(p, pred)
    return {
        "ARI": ari, "ARI_p": (1 + (ari_null >= ari).sum()) / (n_perm + 1),
        "ARI_null_mean": ari_null.mean(), "ARI_null_p95": np.quantile(ari_null, 0.95),
        "NMI": nmi, "NMI_p": (1 + (nmi_null >= nmi).sum()) / (n_perm + 1),
        "NMI_null_mean": nmi_null.mean(), "NMI_null_p95": np.quantile(nmi_null, 0.95),
    }


def one_nn_test(condensed, true_ids, n_perm, rng):
    D = squareform(condensed)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    acc = float((true_ids[nn] == true_ids).mean())
    n = len(true_ids)
    cnt = np.bincount(true_ids)
    chance = float((cnt * (cnt - 1)).sum() / (n * (n - 1)))
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(true_ids)
        null[b] = (p[nn] == p).mean()
    return {"NN_acc": acc, "NN_chance": chance, "NN_p": (1 + (null >= acc).sum()) / (n_perm + 1)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Control analyses for real-SCOP PC-NCSF clustering")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--scop-root", default=".")
    ap.add_argument("--scop-folder-name", default="SCOP")
    ap.add_argument("--loader-root", default=".", help="root passed to fff.data.load_dataset (as in training)")
    ap.add_argument("--tiers", nargs="+", default=["scop_easy", "scop_moderate", "scop_hard", "scop_challenging"])
    ap.add_argument("--ckpt", action="append", default=[], help="'tier=path/to/best_flow.pt' (repeatable)")
    ap.add_argument("--grid-size", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--bw-mults", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-csv", default="scop_clustering_controls.csv")
    args = ap.parse_args(argv)

    overrides = dict(item.split("=", 1) for item in args.ckpt)
    rng = np.random.default_rng(args.seed)
    results = []

    for tier in args.tiers:
        print(f"\n=== {tier} ===")
        subset = tier.split("scop_")[-1] if tier.startswith("scop_") else tier
        csv_path = os.path.join(args.scop_root, args.scop_folder_name, subset, "data.csv")
        if not os.path.exists(csv_path):
            print(f"  SCOP CSV not found: {csv_path} -- skipping")
            continue
        try:
            ckpt_path = find_checkpoint(args.runs_dir, tier, overrides.get(tier))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  {e}\n  skipping {tier}")
            continue

        flow, embedding, _ = load_pcncsf_checkpoint(ckpt_path, args.device)
        n_proteins = embedding.num_embeddings
        angles, labels = get_loader_arrays(tier, args.loader_root)

        # 0. label alignment
        categories, info, ambiguous_indices = build_label_mapping(csv_path, labels)
        print("  exact loader-compatible label alignment:")
        for col, order, n_unique, frac in info["candidates"]:
            share = "n/a (size differs)" if math.isnan(frac) else f"{100 * frac:.1f}%"
            print(f"    {col:13s} {order:17s} unique={n_unique:5d}  {share}")
        if categories is None:
            print("  NOT ACCEPTED: loader-compatible ID counts/order do not match the training loader. "
                  "Skipping this tier rather than reporting potentially invalid scores.")
            continue
        if info["n_labels"] != n_proteins:
            print(f"  WARNING: loader has {info['n_labels']} labels but the checkpoint embedding has "
                  f"{n_proteins} rows. Skipping.")
            continue
        col, order = info["accepted"]
        print(f"  ACCEPTED: '{col}', {order}; exact residue-count alignment verified.")

        keep_indices = [i for i in range(n_proteins) if i not in set(ambiguous_indices)]
        if len(keep_indices) < 2:
            print("  Fewer than two unambiguous proteins remain; skipping.")
            continue
        categories = [categories[i] for i in keep_indices]
        if ambiguous_indices:
            print(f"  Excluding {len(ambiguous_indices)} proteins with multiple SCOP categories; "
                  f"scoring {len(keep_indices)} unambiguous proteins.")

        cats_sorted = sorted(set(categories))
        cat_to_id = {c: i for i, c in enumerate(cats_sorted)}
        true_ids = np.array([cat_to_id[c] for c in categories])
        k = len(cats_sorted)
        print(f"  m = {len(keep_indices)} proteins, k = {k} categories, class sizes = {np.bincount(true_ids).tolist()}")

        reps = []
        dens, cell = per_protein_density_grid(flow, embedding, n_proteins, args.device,
                                              args.grid_size, batch_proteins=32)
        reps.append(("PC-NCSF", subset_condensed_distance(
            hellinger_distance_matrix(dens, cell), keep_indices)))
        for m in args.bw_mults:
            d, c = kde_densities(angles, labels, n_proteins, args.grid_size, m)
            reps.append((f"KDE x{m:g}" + (" (primary)" if m == 1.0 else " (sensitivity)"),
                         subset_condensed_distance(hellinger_distance_matrix(d, c), keep_indices)))

        header = f"  {'representation':24s} {'ARI':>7s} {'p':>7s} {'NMI':>7s} {'p':>7s} {'1-NN':>7s} {'chance':>7s} {'p':>7s}"
        print(header)
        for name, condensed in reps:
            _, _, pred = cluster_and_score(condensed, categories, k, method="ward")
            res = {"tier": tier, "m": len(keep_indices), "k": k,
                   "n_ambiguous_excluded": len(ambiguous_indices), "representation": name}
            res.update(permutation_test_ari_nmi(true_ids, pred, args.n_perm, rng))
            res.update(one_nn_test(condensed, true_ids, args.n_perm, rng))
            results.append(res)
            print(f"  {name:24s} {res['ARI']:7.3f} {res['ARI_p']:7.4f} {res['NMI']:7.3f} {res['NMI_p']:7.4f} "
                  f"{res['NN_acc']:7.3f} {res['NN_chance']:7.3f} {res['NN_p']:7.4f}")
        print(f"  (p-values: permutation, {args.n_perm} shuffles, smallest possible = "
              f"{1 / (args.n_perm + 1):.4f}; KDE uses train+validation residues, the model saw only training)")

    if results:
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        print(f"\nSaved {len(results)} rows to {args.output_csv}")
    else:
        print("\nNo tier produced results -- nothing saved.")
    return results


if __name__ == "__main__":
    main()

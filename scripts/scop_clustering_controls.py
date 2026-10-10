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

The KDE baseline uses ALL residues of each protein (train + validation), because
the train/validation split cannot be reproduced from here; the model saw only
the training part. This slightly favours the baseline and is noted in the output.

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
    """The training script indexes the embedding by an integer label produced
    inside fff.data. The ground-truth category of label i is only known if we
    know which identifier that label stands for. Instead of assuming, test every
    candidate (identifier column x ordering) against the loader's own residue
    counts per label: the right candidate reproduces the counts for (almost) every
    label, a wrong one does not. Returns (categories_by_label, info) or (None, info).
    """
    df = pd.read_csv(csv_path)
    valid = df[["theta", "tau"]].notna().all(axis=1).values
    n_labels = int(loader_labels.max()) + 1
    counts_loader = np.bincount(loader_labels.numpy(), minlength=n_labels)
    cat_col = "category" if "category" in df.columns else "class"

    rows, best = [], None
    for col in [c for c in ("pdb_id", "domain_id", "protein_name") if c in df.columns]:
        for order in ("first-appearance", "sorted"):
            codes, uniques = pd.factorize(df[col], sort=(order == "sorted"))
            if len(uniques) != n_labels:
                rows.append((col, order, len(uniques), float("nan")))
                continue
            counts_csv = np.bincount(codes[valid], minlength=n_labels)
            frac = float((counts_csv == counts_loader).mean())
            rows.append((col, order, len(uniques), frac))
            if best is None or frac > best[0]:
                best = (frac, col, order, codes)

    info = {"candidates": rows, "n_labels": n_labels}
    if best is None or best[0] < min_match:
        info["accepted"] = None
        return None, info

    frac, col, order, codes = best
    grouped = pd.Series(df[cat_col].values).groupby(codes)
    categories = [str(v) for v in grouped.first().reindex(range(n_labels)).values]
    info.update({"accepted": (col, order), "match": frac,
                 "ambiguous": int((grouped.nunique() > 1).sum())})
    return categories, info


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
    ap.add_argument("--scop-folder-name", default="Scop_data")
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
        categories, info = build_label_mapping(csv_path, labels)
        print("  label-alignment check (identifier column x ordering -> share of labels whose "
              "residue count matches the loader):")
        for col, order, n_unique, frac in info["candidates"]:
            share = "n/a (size differs)" if math.isnan(frac) else f"{100 * frac:.1f}%"
            print(f"    {col:13s} {order:17s} unique={n_unique:5d}  {share}")
        if categories is None:
            print("  NOT ACCEPTED: no candidate reproduces the loader's per-label residue counts, so the "
                  "label -> category alignment cannot be verified. Any ARI/NMI for this tier would be "
                  "unreliable. Skipping; please send this output.")
            continue
        if info["n_labels"] != n_proteins:
            print(f"  WARNING: loader has {info['n_labels']} labels but the checkpoint embedding has "
                  f"{n_proteins} rows. Skipping.")
            continue
        col, order = info["accepted"]
        print(f"  ACCEPTED: '{col}', {order}  ({100 * info['match']:.1f}% of labels match; "
              f"{info['ambiguous']} label(s) with more than one category)")

        cats_sorted = sorted(set(categories))
        cat_to_id = {c: i for i, c in enumerate(cats_sorted)}
        true_ids = np.array([cat_to_id[c] for c in categories])
        k = len(cats_sorted)
        print(f"  m = {n_proteins} proteins, k = {k} categories, class sizes = {np.bincount(true_ids).tolist()}")

        reps = []
        dens, cell = per_protein_density_grid(flow, embedding, n_proteins, args.device,
                                              args.grid_size, batch_proteins=32)
        reps.append(("PC-NCSF", hellinger_distance_matrix(dens, cell)))
        for m in args.bw_mults:
            d, c = kde_densities(angles, labels, n_proteins, args.grid_size, m)
            reps.append((f"KDE x{m:g}" + (" (primary)" if m == 1.0 else " (sensitivity)"),
                         hellinger_distance_matrix(d, c)))

        header = f"  {'representation':24s} {'ARI':>7s} {'p':>7s} {'NMI':>7s} {'p':>7s} {'1-NN':>7s} {'chance':>7s} {'p':>7s}"
        print(header)
        for name, condensed in reps:
            _, _, pred = cluster_and_score(condensed, categories, k, method="ward")
            res = {"tier": tier, "m": n_proteins, "k": k, "representation": name}
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

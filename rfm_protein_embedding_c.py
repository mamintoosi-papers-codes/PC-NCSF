"""
Protein-Conditional Riemannian Flow Matching (PC-RFM).

It mirrors the structure
of circularspline_protein_embedding_c.py (PC-NCSF) so the two can be compared directly:
same CLI conventions, same dataset loader, same embedding-conditioning mechanism, same
checkpoint/metrics saving utilities.

The pure math (vector field, flow-matching loss, ODE-based NLL) lives in rfm_core.py so
it can be unit-tested independently (see test_rfm_core.py) -- this file is only the CLI,
data loading, and training/evaluation driver.

KEY CONCEPTUAL DIFFERENCE FROM PC-NCSF:
    PC-NCSF is a *discrete* normalizing flow: one invertible transform with a closed-form
    Jacobian, so the training loss IS the exact NLL (-log_prob).
    PC-RFM is a *continuous* normalizing flow trained by Flow Matching: the training loss
    (cfm_loss) is a simulation-free vector-field-regression loss, NOT the NLL. It only
    tells you the model is learning *a* velocity field, not how good the resulting density
    is. To get a number comparable to PC-NCSF's Table 2 (validation NLL), you must
    separately integrate the learned ODE (compute_nll) after training.

"""

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import math
import random
import numpy as np

from rfm_core import (
    TWO_PI,
    VectorField,
    cfm_loss,
    compute_nll,
    aggregate_per_protein_nll,
)


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# CLI (kept as close as possible to circularspline_protein_embedding_c.py)
# ---------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description="Train conditional Riemannian Flow Matching (PC-RFM) on protein torsion angles.")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--hidden-layers", type=int, default=3)
parser.add_argument("--embedding-dim", type=int, default=16)
parser.add_argument("--save-dir", type=str, default="runs")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default="torus_protein")
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--nll-ode-steps", type=int, default=50,
                     help="Fixed RK4 steps for the ODE solve used to evaluate NLL (eval-time only).")
args = parser.parse_args()

import sys, io, contextlib
@contextlib.contextmanager
def suppress_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout

set_seed(args.seed)

run_name = args.tag if args.tag else f"rfm_bs{args.batch_size}_ep{args.epochs}_ed{args.embedding_dim}_hd{args.hidden_dim}"
save_dir = os.path.join(args.save_dir, args.dataset, run_name)
os.makedirs(save_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Data (identical loader to the PC-NCSF script, so the split is directly comparable)
# ---------------------------------------------------------------------------
from fff.data import load_dataset
from fff.evaluate.tori import convert_to_angles

dataset_name = args.dataset
dataset_root = "./fff/data"
if dataset_name.startswith("scop"):
    dataset_root = "."

with suppress_stdout(args.quiet):
    protein_dataset = load_dataset(dataset_name, root=dataset_root, condition_on="residue")
trainset, valset = [convert_to_angles(ds[:][0].to(device)) for ds in protein_dataset]
traincond, valcond = [ds[:][1].to(device) for ds in protein_dataset]
n_cond = len(traincond.cpu().unique())

from torch.utils.data import TensorDataset
config = {
    "lr": args.lr,
    "epochs": args.epochs,
    "network": {"hidden_features": [args.hidden_dim] * args.hidden_layers},
    "batch_size": args.batch_size,
    "patience": args.patience,
    "seed": args.seed,
    "embedding_dim": args.embedding_dim,
    "dataset": args.dataset,
    "tag": args.tag,
    "model": "PC-RFM",
    "nll_ode_steps": args.nll_ode_steps,
}
trainloader = torch.utils.data.DataLoader(TensorDataset(trainset, traincond), batch_size=config["batch_size"], shuffle=True)
# shuffle=False here on purpose: together with the fixed noise cache below, this makes
# val_loss fully reproducible epoch-to-epoch (see "VALIDATION MONTE CARLO STABILIZATION").
valloader = torch.utils.data.DataLoader(TensorDataset(valset, valcond), batch_size=config["batch_size"], shuffle=False)


# ---------------------------------------------------------------------------
# Fixed validation Monte Carlo noise (drawn once, reused every epoch)
# ---------------------------------------------------------------------------
val_noise_gen = torch.Generator(device="cpu").manual_seed(args.seed + 12345)
val_x0_all = (torch.rand(len(valset), 2, generator=val_noise_gen) * TWO_PI).to(device)
val_t_all = torch.rand(len(valset), 1, generator=val_noise_gen).to(device)


def run_validation_epoch(vf, embedding):
    """One deterministic pass over valloader using the fixed noise cache above."""
    vf.eval()
    embedding.eval()
    total_loss = 0.0
    offset = 0
    with torch.no_grad():
        for x, c_labels in valloader:
            bs = x.shape[0]
            c_labels = c_labels.long().to(device)
            x0_fixed = val_x0_all[offset:offset + bs]
            t_fixed = val_t_all[offset:offset + bs]
            offset += bs
            total_loss += cfm_loss(vf, embedding, x, c_labels, t=t_fixed, x0=x0_fixed).item()
    return total_loss / len(valloader)


# ---------------------------------------------------------------------------
# Protein-ID mapping (best-effort): lets per_protein_nll.csv report the PDB id
# instead of just the raw integer label, using the same convention as
# report_generator.py ("helper used in the torus dataset loader: extract
# first token before ':'", sorted-unique for a stable, run-independent index).
# ---------------------------------------------------------------------------
def load_protein_id_table(dataset_root, dataset_name):
    if not dataset_name.startswith("torus"):
        return None
    torus_tsv = os.path.join(dataset_root, "raw_data", "torus", "protein.tsv")
    if not os.path.exists(torus_tsv):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(torus_tsv, delimiter="\t", header=None)
        df.columns = ["name", "phi", "psi", "subtype"]

        def _extract_res(name):
            return str(name).split(":")[0].upper()

        classes = sorted(set(_extract_res(n) for n in df["name"].values))
        return classes  # classes[label_id] -> pdb id
    except Exception as e:
        if not args.quiet:
            print(f"Warning: could not build protein-ID table ({e}); "
                  f"per_protein_nll.csv will use raw integer labels only.")
        return None


protein_id_table = load_protein_id_table(dataset_root, dataset_name)


# ---------------------------------------------------------------------------
# Model, optimizer, scheduler (mirrors circularspline_protein_embedding_c.py)
# ---------------------------------------------------------------------------
embedding = nn.Embedding(num_embeddings=n_cond, embedding_dim=config["embedding_dim"]).to(device)
vf = VectorField(config["embedding_dim"], args.hidden_dim, args.hidden_layers).to(device)

if not args.quiet:
    print(f"Vector field parameters: {sum(p.numel() for p in vf.parameters())}")
    print(f"Embedding parameters: {sum(p.numel() for p in embedding.parameters())}")
    print(f"Number of classes: {n_cond}")

optimizer = torch.optim.Adam(
    [{'params': vf.parameters()}, {'params': embedding.parameters()}],
    lr=config["lr"]
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=config["lr"], epochs=config["epochs"], steps_per_epoch=len(trainloader)
)

from tqdm import trange
pbar = trange(config["epochs"], disable=args.quiet)

best_val = float("inf")
epochs_no_improve = 0
train_losses, val_losses = [], []  # CFM regression loss curves (NOT NLL -- for early stopping only)
best_ckpt_path = os.path.join(save_dir, "best_flow.pt")

for epoch in pbar:
    vf.train()
    embedding.train()
    train_loss = 0.0
    for x, c_labels in trainloader:
        optimizer.zero_grad(set_to_none=True)
        c_labels = c_labels.long().to(device)
        loss = cfm_loss(vf, embedding, x, c_labels)  # fresh noise: correct for training
        loss.backward()
        train_loss += loss.detach().item()
        optimizer.step()
        scheduler.step()
    train_loss /= len(trainloader)

    val_loss = run_validation_epoch(vf, embedding)  # fixed noise: correct for early stopping

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save({
            "vf_state_dict": vf.state_dict(),
            "embedding_state_dict": embedding.state_dict(),
            "config": config,
            "cond_dim": config["embedding_dim"],
        }, best_ckpt_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= config["patience"]:
            break

# ---------------------------------------------------------------------------
# Reload best checkpoint, then evaluate the metrics that are actually
# comparable to the paper's tables (see the module docstring for which
# convention applies where).
# ---------------------------------------------------------------------------
ckpt = torch.load(best_ckpt_path, map_location=device)
vf.load_state_dict(ckpt["vf_state_dict"])
embedding.load_state_dict(ckpt["embedding_state_dict"])
vf.eval()
embedding.eval()

# --- Table 2 -style flat aggregate NLL over all validation residues ---
val_nlls, val_labels = [], []
for x, c_labels in valloader:
    c_labels = c_labels.long().to(device)
    nll_batch = compute_nll(vf, embedding, x, c_labels, n_steps=args.nll_ode_steps)
    val_nlls.append(nll_batch)
    val_labels.append(c_labels)
val_nlls = torch.cat(val_nlls)
val_labels = torch.cat(val_labels)
val_nll_mean = val_nlls.mean().item()

# --- Table 4/5 -style per-protein NLL (inner stage: mean within each protein) ---
per_protein = aggregate_per_protein_nll(val_nlls, val_labels)
per_protein_csv_path = os.path.join(save_dir, "per_protein_nll.csv")
with open(per_protein_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label_id", "pdb_id", "mean_nll", "n_val_residues"])
    for label_id in sorted(per_protein.keys()):
        mean_nll, n_res = per_protein[label_id]
        pdb_id = protein_id_table[label_id] if protein_id_table is not None and label_id < len(protein_id_table) else ""
        writer.writerow([label_id, pdb_id, mean_nll, n_res])

# Unweighted mean-of-per-protein-means -- the Table 4/5 "outer" stage, reported here
# for convenience; NOT the same number as val_nll_mean above when protein lengths vary.
per_protein_mean_unweighted = float(np.mean([v[0] for v in per_protein.values()]))

if not args.quiet:
    print(f"Best validation CFM loss (early-stopping criterion, fixed MC noise): {best_val:.4f}")
    print(f"Validation NLL, flat over residues  (Table 2 -style): {val_nll_mean:.4f}")
    print(f"Validation NLL, unweighted mean of per-protein means (Table 4/5 -style): {per_protein_mean_unweighted:.4f}")
    print(f"Per-protein NLL saved to: {per_protein_csv_path}")
    print(f"Artifacts saved to: {save_dir}")

from fff.train_utils import save_config_and_checkpoint, save_metrics_csv, ensure_run_folder
run_path = ensure_run_folder(args.save_dir, args.dataset, run_name)
save_metrics_csv(run_path, train_losses, val_losses)
config["val_nll_flat"] = val_nll_mean
config["val_nll_per_protein_unweighted"] = per_protein_mean_unweighted
save_config_and_checkpoint(run_path, vf.state_dict(), config, config["embedding_dim"], embedding.state_dict())

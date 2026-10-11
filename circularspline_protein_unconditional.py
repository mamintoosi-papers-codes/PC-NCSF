"""
circularspline_protein_unconditional.py

Unconditional NCSF baseline -- for the paired statistical test.

This is circularspline_protein_embedding_c_diagnostic.py with ONE change: the
embedding table has a SINGLE row (n_cond_model = 1), and every residue --
whichever protein it actually belongs to -- is mapped to that same row before
being fed to the flow. So the flow literally cannot distinguish proteins; it
learns one shared density, exactly like the "Unconditional" row of Table 2.

Two label streams are kept deliberately separate:
  - cond_labels_model : always 0, fed to the embedding (makes the model
    unconditional).
  - cond_labels_true  : the REAL per-protein label from the dataset, used ONLY
    to group residues when writing per_protein_nll.csv at the end.
This lets the output CSV be grouped by the same real protein identity as the
conditional run's per_protein_nll.csv (same pdb_id column), so the two files
can be joined and paired (Wilcoxon / paired t-test) protein-by-protein.

"""

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import numpy as np
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import argparse
import csv
parser = argparse.ArgumentParser(description="Unconditional NCSF baseline, with per-protein NLL breakdown for pairing.")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--transforms", type=int, default=8)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--hidden-layers", type=int, default=3)
parser.add_argument("--embedding-dim", type=int, default=16,
                    help="Kept for CLI parity with the conditional script; the embedding "
                         "table always has exactly 1 row regardless of this value.")
parser.add_argument("--save-dir", type=str, default="runs")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--inspect", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--dataset", type=str, default="torus_protein")
parser.add_argument("--tag", type=str, default=None)
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

run_name = args.tag if args.tag else f"uncond_bs{args.batch_size}_ep{args.epochs}_hd{args.hidden_dim}"
save_dir = os.path.join(args.save_dir, args.dataset, run_name)
os.makedirs(save_dir, exist_ok=True)

from fff.data import load_dataset
from fff.evaluate.tori import convert_to_angles

dataset_name = args.dataset
dataset_root = "./fff/data"
if dataset_name.startswith("scop"):
    dataset_root = "."

with suppress_stdout(args.quiet):
    protein_dataset = load_dataset(dataset_name, root=dataset_root, condition_on="residue")
trainset, valset = [convert_to_angles(ds[:][0].to(device)) for ds in protein_dataset]
traincond_true, valcond_true = [ds[:][1].to(device) for ds in protein_dataset]  # REAL protein labels
n_cond_true = len(traincond_true.cpu().unique())

# --- the only real change: everyone maps to embedding row 0 -----------------
traincond_model = torch.zeros_like(traincond_true)
valcond_model = torch.zeros_like(valcond_true)
n_cond_model = 1

if not args.quiet:
    print(f"[diagnostic] Real distinct proteins in data: {n_cond_true} "
          f"(ignored by the model -- this run is UNCONDITIONAL, embedding has {n_cond_model} row)")

import zuko
config = {
    "lr": args.lr, "epochs": args.epochs,
    "network": {"hidden_features": [args.hidden_dim] * args.hidden_layers, "transforms": args.transforms},
    "batch_size": args.batch_size, "patience": args.patience, "seed": args.seed,
    "embedding_dim": args.embedding_dim, "dataset": args.dataset, "tag": args.tag,
    "model": "NCSF-unconditional",
}

from torch.utils.data import TensorDataset
trainloader = torch.utils.data.DataLoader(
    TensorDataset(trainset, traincond_model, traincond_true), batch_size=config["batch_size"], shuffle=True)
valloader = torch.utils.data.DataLoader(
    TensorDataset(valset, valcond_model, valcond_true), batch_size=config["batch_size"], shuffle=False)

embedding = nn.Embedding(num_embeddings=n_cond_model, embedding_dim=config["embedding_dim"]).to(device)
cond_dim = config["embedding_dim"]
flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)

if args.inspect and not args.quiet:
    print(f"Flow parameters: {sum(p.numel() for p in flow.parameters())}")
    print(f"Embedding parameters: {sum(p.numel() for p in embedding.parameters())} (1 row -- unconditional)")

optimizer = torch.optim.Adam([{'params': flow.parameters()}, {'params': embedding.parameters()}], lr=config["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=config["lr"], epochs=config["epochs"], steps_per_epoch=len(trainloader))
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

from tqdm import trange
pbar = trange(config["epochs"], disable=args.quiet)

best_val = float("inf")
epochs_no_improve = 0
train_losses, val_losses = [], []
best_ckpt_path = os.path.join(save_dir, "best_flow.pt")

for epoch in pbar:
    flow.train(); embedding.train()
    train_loss = 0.0
    for x, c_labels_model, _ in trainloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            c_labels_model = c_labels_model.long().to(device)
            c = embedding(c_labels_model)
            loss = -flow(c).log_prob(x)
            loss = loss.mean()
        train_loss += loss.detach().item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    train_loss /= len(trainloader)

    with torch.no_grad():
        flow.eval(); embedding.eval()
        val_loss = 0.0
        for x, c_labels_model, _ in valloader:
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                c_labels_model = c_labels_model.long().to(device)
                c = embedding(c_labels_model)
                loss = -flow(c).log_prob(x)
                loss = loss.mean()
            val_loss += loss.item()
        val_loss /= len(valloader)
    train_losses.append(train_loss); val_losses.append(val_loss)

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save({"flow_state_dict": flow.state_dict(), "embedding_state_dict": embedding.state_dict(),
                    "config": config, "cond_dim": cond_dim}, best_ckpt_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= config["patience"]:
            break

ckpt = torch.load(best_ckpt_path, map_location=device)
flow.load_state_dict(ckpt["flow_state_dict"])
embedding.load_state_dict(ckpt["embedding_state_dict"])
flow.eval(); embedding.eval()

if not args.quiet:
    print(f"Best Validation Loss (unconditional, flat over residues -- matches Table 2 convention): {best_val:.4f}")

# ---------------------------------------------------------------------------
# Per-protein NLL, grouped by the REAL protein identity (not the dummy model
# label), so this lines up with the conditional run's per_protein_nll.csv.
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
        return sorted(set(_extract_res(n) for n in df["name"].values))
    except Exception as e:
        if not args.quiet:
            print(f"Warning: could not build protein-ID table ({e})")
        return None

protein_id_table = load_protein_id_table(dataset_root, dataset_name)

val_nlls, val_labels_true = [], []
with torch.no_grad():
    for x, c_labels_model, c_labels_true in valloader:
        c_labels_model = c_labels_model.long().to(device)
        c = embedding(c_labels_model)
        nll_batch = -flow(c).log_prob(x)
        val_nlls.append(nll_batch)
        val_labels_true.append(c_labels_true.to(device))
val_nlls = torch.cat(val_nlls)
val_labels_true = torch.cat(val_labels_true)
val_nll_flat = val_nlls.mean().item()

def aggregate_per_protein_nll(nll_per_sample, cond_labels):
    out = {}
    labels_cpu = cond_labels.detach().cpu()
    nll_cpu = nll_per_sample.detach().cpu()
    for pid in labels_cpu.unique().tolist():
        mask = labels_cpu == pid
        out[int(pid)] = (nll_cpu[mask].mean().item(), int(mask.sum().item()))
    return out

per_protein = aggregate_per_protein_nll(val_nlls, val_labels_true)
per_protein_csv_path = os.path.join(save_dir, "per_protein_nll.csv")
with open(per_protein_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label_id", "pdb_id", "mean_nll", "n_val_residues"])
    for label_id in sorted(per_protein.keys()):
        mean_nll, n_res = per_protein[label_id]
        pdb_id = protein_id_table[label_id] if protein_id_table is not None and label_id < len(protein_id_table) else ""
        writer.writerow([label_id, pdb_id, mean_nll, n_res])

per_protein_mean_unweighted = float(np.mean([v[0] for v in per_protein.values()]))

if not args.quiet:
    print(f"Validation NLL, flat over residues (Table 2 -style): {val_nll_flat:.4f}")
    print(f"Validation NLL, unweighted mean of per-protein means: {per_protein_mean_unweighted:.4f}")
    print(f"Per-protein NLL (grouped by REAL protein identity) saved to: {per_protein_csv_path}")
    print(f"Artifacts saved to: {save_dir}")

from fff.train_utils import save_config_and_checkpoint, save_metrics_csv, ensure_run_folder
run_path = ensure_run_folder(args.save_dir, args.dataset, run_name)
save_metrics_csv(run_path, train_losses, val_losses)
config["val_nll_flat"] = val_nll_flat
save_config_and_checkpoint(run_path, flow.state_dict(), config, cond_dim, embedding.state_dict())

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Reproducibility utilities
import random
import numpy as np
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# CLI
import argparse
import time
import csv
parser = argparse.ArgumentParser(description="Train conditional flow on protein torsion angles.")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--transforms", type=int, default=8)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--hidden-layers", type=int, default=3)
parser.add_argument("--plot-indices", type=str, default="")
parser.add_argument("--plot-all", action="store_true")
parser.add_argument("--save-dir", type=str, default="runs")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--inspect", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--dataset", type=str, default="torus_protein",
                    help="Dataset name to load (e.g., 'torus_protein' or 'scop_easy')")
parser.add_argument("--tag", type=str, default=None,
                    help="Optional tag to group paired runs (e.g., 'exp01'). If set, tag will be used as run folder name to match paired runs.")
args = parser.parse_args()

# Utility to optionally suppress stdout (e.g., to silence dataset prints)
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

# Keep run name compact: include only batch-size, epochs and hidden-dim
if args.tag:
    run_name = f"{args.tag}"
else:
    run_name = f"uncond_bs{args.batch_size}_ep{args.epochs}_hd{args.hidden_dim}"
# Make save directory dataset-aware so results for different datasets go to separate folders
save_dir = os.path.join(args.save_dir, args.dataset, run_name)
os.makedirs(save_dir, exist_ok=True)

# Load protein data
from fff.data import load_dataset
from fff.evaluate.tori import convert_to_angles
with suppress_stdout(args.quiet):
    # Allow selecting dataset via CLI; use project root for SCOP datasets
    dataset_name = args.dataset
    dataset_root = "./fff/data"
    if dataset_name.startswith("scop"):
        dataset_root = "."
    protein_dataset = load_dataset(dataset_name, root=dataset_root) # , condition_on="residue"
trainset, valset = [convert_to_angles(ds[:][0].to(device)) for ds in protein_dataset]
# Inspect conditioning: number of classes and some samples (optional)
if args.inspect and not args.quiet:
    print("first 3 x (angles):\n", trainset[:3].cpu())

import zuko

config = {
    "lr": args.lr,
    "epochs": args.epochs,
    "network": {
        "hidden_features": [args.hidden_dim] * args.hidden_layers,
        "transforms": args.transforms,
    },
    "batch_size": args.batch_size,
    "patience": args.patience,
    "seed": args.seed,
    "dataset": args.dataset,
    "tag": args.tag,
}

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=config["batch_size"], shuffle=True)

cond_dim = 1
flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)
if args.inspect and not args.quiet:
    print(f"Parameters: {sum(p.numel() for p in flow.parameters())}")
optimizer = torch.optim.Adam(flow.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], epochs=config["epochs"], steps_per_epoch=len(trainloader))
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

from tqdm import trange

pbar = trange(config["epochs"], disable=args.quiet)

best_val = float("inf")
epochs_no_improve = 0
train_losses, val_losses = [], []
best_ckpt_path = os.path.join(save_dir, "best_flow.pt")

for epoch in pbar:
    flow.train()
    train_loss = 0.0
    for x in trainloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):            
            c = torch.zeros_like(x[:,:1])
            loss = -flow(c).log_prob(x)  # -log p(x | c)
            loss = loss.mean()
        train_loss += loss.detach().item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    train_loss /= len(trainloader)

    with torch.no_grad():
        flow.eval()
        val_loss = 0.0
        for x in valloader:
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                c = torch.zeros_like(x[:,:1])
                loss = -flow(c).log_prob(x)  # -log p(x | c)
                loss = loss.mean()
            val_loss += loss.item()
        val_loss /= len(valloader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping and checkpointing
    if val_loss < best_val - 1e-6:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save({"state_dict": flow.state_dict(), "config": config, "cond_dim": cond_dim}, best_ckpt_path)
    else:
        epochs_no_improve += 1

    pbar.set_description(f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | no-imp: {epochs_no_improve}/{config['patience']}")
    if epochs_no_improve >= config["patience"]:
        break

# Load best checkpoint and evaluate on test set
ckpt = torch.load(os.path.join(save_dir, "best_flow.pt"), map_location=device)
flow.load_state_dict(ckpt["state_dict"])


if not args.quiet:
    print(f"Best Validation Loss: {best_val:.3f}")
    # print(f"Test Loss (NLL): {test_loss:.3f}")
    print(f"Artifacts saved to: {save_dir}")

from fff.train_utils import save_config_and_checkpoint, save_metrics_csv, ensure_run_folder

# Save metrics and checkpoint using train_utils (no plotting here)
run_path = ensure_run_folder(args.save_dir, args.dataset, run_name)
save_metrics_csv(run_path, train_losses, val_losses)
save_config_and_checkpoint(run_path, flow.state_dict(), config, cond_dim)


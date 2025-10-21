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

from typing import Optional
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_model_log_densities(
    model,
    reference_data=None,
    cond_index: int = 0,
    num_grid_points: int = 200,
    levels: int = 10,
    ax: Optional[plt.Axes] = None,
    fontsizes: dict = dict(TITLESIZE=18, LABELSIZE=16, TICKSIZE=14),
):
    """
    Plot log density contours for unconditional model.
    Uses [-π, π] for φ and [0, 2π] for ψ to avoid discontinuity at ψ=0.
    
    Args:
        model: Trained unconditional flow model
        reference_data: Optional validation data to scatter plot
        cond_index: Condition index (for compatibility, not used in unconditional)
        num_grid_points: Resolution of the grid
        levels: Number of contour levels
        ax: Matplotlib axes to plot on
        fontsizes: Font size settings
    """
    
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

    # Create grid for visualization: φ in [-π, π], ψ in [0, 2π]
    range_phi = torch.linspace(-torch.pi, torch.pi, num_grid_points)
    range_psi = torch.linspace(0, 2 * torch.pi, num_grid_points)
    phi_grid, psi_grid = torch.meshgrid(range_phi, range_psi, indexing='ij')
    x = torch.stack((phi_grid, psi_grid), dim=-1).to(device).reshape(-1, 2)
    
    # For unconditional model, create dummy condition tensor
    c = torch.zeros(x.shape[0], cond_dim, device=device, dtype=x.dtype)
    c[:, 0] = cond_index
    
    # Calculate log probabilities
    log_prob = model(c).log_prob(x).cpu()

    # Create contour plot
    phi, psi = x[..., 0].cpu(), x[..., 1].cpu()
    contours = ax.tricontourf(phi, psi, log_prob, levels=levels, cmap="viridis")
    cbar = plt.colorbar(contours)
    cbar.set_label("Log density", fontsize=fontsizes.get("LABELSIZE"))
    cbar.ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    # Set plot limits: φ in [-π, π], ψ in [0, 2π]
    ax.set_xlim(-torch.pi, torch.pi)
    ax.set_ylim(0, 2 * torch.pi)

    # Set ticks for φ axis (horizontal) - [-π, π]
    ax.set_xticks(
        [-torch.pi, -torch.pi/2, 0, torch.pi/2, torch.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    
    # Set ticks for ψ axis (vertical) - [0, 2π]
    ax.set_yticks(
        [0, torch.pi/2, torch.pi, 3*torch.pi/2, 2*torch.pi],
        [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
    )
    
    ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    ax.set_xlabel(r"$\Phi$", fontsize=fontsizes.get("LABELSIZE"))
    ax.set_ylabel(r"$\Psi$", fontsize=fontsizes.get("LABELSIZE"))

    # Add reference data if provided
    if reference_data is not None:
        # Convert ψ values to [0, 2π] range while keeping φ in [-π, π]
        ref_phi = reference_data[..., 0]
        ref_psi = reference_data[..., 1] % (2 * torch.pi)  # Wrap ψ to [0, 2π]
        
        ax.scatter(
            ref_phi,
            ref_psi,
            s=1 / len(reference_data) * 2e3,
            c="black",
            alpha=0.2,
            label="validation data",
        )

    ax.set_title("Log density", fontsize=fontsizes.get("TICKSIZE"))
    return ax.figure

# Save metrics and plots
# Loss curves
fig = plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("NLL")
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
plt.close(fig)

# Save metrics CSV
with open(os.path.join(save_dir, "metrics.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for i, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
        writer.writerow([i, tr, va])

idx = 0
fig = plot_model_log_densities(flow, valset.cpu(), cond_index=idx)
fig.savefig(os.path.join(save_dir, f"contour_cond_{idx}.png"), dpi=150, bbox_inches="tight")
plt.close(fig)


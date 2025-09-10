import torch
import torch.nn as nn

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
# parser.add_argument("--cond-index", type=int, default=0)
parser.add_argument("--plot-indices", type=str, default="")
parser.add_argument("--plot-all", action="store_true")
parser.add_argument("--save-dir", type=str, default="runs")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--inspect", action="store_true")
parser.add_argument("--quiet", action="store_true")
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

run_name = f"cond_residue_{time.strftime('%Y%m%d-%H%M%S')}"
save_dir = os.path.join(args.save_dir, run_name)
os.makedirs(save_dir, exist_ok=True)

# Load protein data
from fff.data import load_dataset
from fff.evaluate.tori import convert_to_angles
with suppress_stdout(args.quiet):
    protein_dataset = load_dataset("torus_protein", root="./fff/data", condition_on="residue")
trainset, valset, testset = [convert_to_angles(ds[:][0].to(device)) for ds in protein_dataset]
traincond, valcond, testcond = [ds[:][1].to(device) for ds in protein_dataset]
labels = traincond.cpu()
n_cond = len(labels.unique())
# Inspect conditioning: number of classes and some samples (optional)
if args.inspect and not args.quiet:
    labels = traincond.cpu()
    print(len(labels), n_cond)
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
    "embedding_dim": 32  # بعد جدید برای embedding
}

from torch.utils.data import TensorDataset
trainloader = torch.utils.data.DataLoader(TensorDataset(trainset, traincond), batch_size=config["batch_size"], shuffle=True)
valloader = torch.utils.data.DataLoader(TensorDataset(valset, valcond), batch_size=config["batch_size"], shuffle=True)
testloader = torch.utils.data.DataLoader(TensorDataset(testset, testcond), batch_size=config["batch_size"], shuffle=True)

# ایجاد مدل با Embedding
embedding = nn.Embedding(n_cond, config["embedding_dim"]).to(device)

cond_dim = 1
flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)
if args.inspect and not args.quiet:
    print(f"Parameters: {sum(p.numel() for p in flow.parameters())}")

# بهینه‌ساز باید پارامترهای هر دو مدل را شامل شود
optimizer = torch.optim.Adam(
    [
        {'params': embedding.parameters()},
        {'params': flow.parameters()}
    ],
    lr=config["lr"]
)    
# optimizer = torch.optim.Adam(flow.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], epochs=config["epochs"], steps_per_epoch=len(trainloader))
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

from tqdm import trange

pbar = trange(config["epochs"], disable=args.quiet)

best_val = float("inf")
epochs_no_improve = 0
train_losses, val_losses = [], []
best_ckpt_path = os.path.join(save_dir, "best_flow.pt")


# ابتدا خارج از حلقه، Embedding Layer را تعریف کنید
embedding = nn.Embedding(num_embeddings=500, embedding_dim=cond_dim).to(device)

# بهینه‌ساز را برای هر دو مدل تنظیم کنید
optimizer = torch.optim.Adam(
    [
        {'params': flow.parameters()},
        {'params': embedding.parameters()}
    ],
    lr=config["lr"]
)


for epoch in pbar:
    flow.train()
    embedding.train()  # Embedding را هم در حالت آموزش قرار دهید
    train_loss = 0.0
    
    for x, c_labels in trainloader:  # c_labels اکنون integer هستند
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            # تبدیل labelهای integer به embedding
            c_labels = c_labels.long().to(device)
            c = embedding(c_labels)  # shape: (batch_size, cond_dim)
            
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
        embedding.eval()  # Embedding را هم در حالت ارزیابی قرار دهید
        val_loss = 0.0
        
        for x, c_labels in valloader:
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                # تبدیل labelهای integer به embedding
                c_labels = c_labels.long().to(device)
                c = embedding(c_labels)  # shape: (batch_size, cond_dim)
                
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
        # هر دو state_dict را ذخیره کنید
        torch.save({
            "flow_state_dict": flow.state_dict(),
            "embedding_state_dict": embedding.state_dict(),
            "config": config,
            "cond_dim": cond_dim
        }, best_ckpt_path)
    else:
        epochs_no_improve += 1

# Load best checkpoint and evaluate on test set
ckpt = torch.load(os.path.join(save_dir, "best_flow.pt"), map_location=device)

# بارگذاری state dictهای هر دو مدل
flow.load_state_dict(ckpt["flow_state_dict"])
embedding.load_state_dict(ckpt["embedding_state_dict"])

with torch.no_grad():
    flow.eval()
    embedding.eval()  # Embedding را هم در حالت ارزیابی قرار دهید
    
    test_loss = 0.0
    for x, c_labels in testloader:  # c_labels اکنون integer هستند
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            c_labels = c_labels.long().to(device)
            c = embedding(c_labels)  # shape: (batch_size, cond_dim)
            
            loss = -flow(c).log_prob(x)
            loss = loss.mean()
        
        test_loss += loss.item()
    
    test_loss /= len(testloader)

print(f"Test Loss: {test_loss:.4f}")

# برای نمونه‌گیری از مدل آموزش دیده:
# def generate_samples(class_id, num_samples=1000):
#     """تولید نمونه از یک کلاس خاص"""
#     with torch.no_grad():
#         flow.eval()
#         embedding.eval()
        
#         # ایجاد شرط برای کلاس مورد نظر
#         class_tensor = torch.tensor([class_id], device=device)
#         c = embedding(class_tensor)  # (1, cond_dim)
        
#         # تکرار شرط برای تعداد نمونه‌های مورد نیاز
#         c_repeated = c.repeat(num_samples, 1)  # (num_samples, cond_dim)
        
#         # نمونه‌گیری
#         samples = flow(c_repeated).sample()
        
#     return samples.cpu()

# # مثال استفاده:
# samples_class_0 = generate_samples(0, 1000)
# samples_class_123 = generate_samples(123, 1000)
# samples_class_499 = generate_samples(499, 1000)

if not args.quiet:
    print(f"Best Validation Loss: {best_val:.3f}")
    print(f"Test Loss (NLL): {test_loss:.3f}")
    print(f"Artifacts saved to: {save_dir}")

from typing import Optional
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_model_log_densities(
    model,
    embedding_layer,  # ADDED: Embedding layer parameter
    reference_data=None,
    cond_index: int = 0,
    num_grid_points: int = 200,
    levels: int = 10,
    ax: Optional[plt.Axes] = None,
    fontsizes: dict = dict(TITLESIZE=18, LABELSIZE=16, TICKSIZE=14),
):
    """
    Plot log density contours for a given conditional index.
    
    Args:
        model: The trained flow model
        embedding_layer: The trained embedding layer for condition processing
        reference_data: Optional validation data to scatter plot
        cond_index: Which condition index to visualize (0-499)
        num_grid_points: Resolution of the grid
        levels: Number of contour levels
        ax: Matplotlib axes to plot on
        fontsizes: Font size settings
    """
    
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

    # Create grid for visualization
    range_angular = torch.linspace(-torch.pi, torch.pi, num_grid_points)
    phi_grid, psi_grid = torch.meshgrid(range_angular, range_angular, indexing='ij')
    x = torch.stack((phi_grid, psi_grid), dim=-1).to(device).reshape(-1, 2)
    
    # MODIFIED: Use embedding layer instead of direct condition tensor
    # Create condition tensor with the specified index
    cond_tensor = torch.tensor([cond_index], device=device, dtype=torch.long)
    c_embedding = embedding_layer(cond_tensor)  # Get embedding for this condition
    
    # Repeat embedding for all grid points
    c = c_embedding.repeat(x.shape[0], 1)  # shape: (num_grid_points**2, cond_dim)
    
    # Calculate log probabilities
    log_prob = model(c).log_prob(x).cpu()

    # Create contour plot
    phi, psi = x[..., 0].cpu(), x[..., 1].cpu()
    contours = ax.tricontourf(phi, psi, log_prob, levels=levels, cmap="viridis")
    cbar = plt.colorbar(contours)
    cbar.set_label("Log density", fontsize=fontsizes.get("LABELSIZE"))
    cbar.ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    # Set plot limits and labels
    ax.set_xlim(-torch.pi, torch.pi)
    ax.set_ylim(-torch.pi, torch.pi)

    ax.set_xticks(
        [-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    ax.set_yticks(
        [-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    ax.set_xlabel(r"$\Phi$", fontsize=fontsizes.get("LABELSIZE"))
    ax.set_ylabel(r"$\Psi$", fontsize=fontsizes.get("LABELSIZE"))

    # Add reference data if provided
    if reference_data is not None:
        ax.scatter(
            reference_data[..., 0],
            reference_data[..., 1],
            s=1 / len(reference_data) * 2e3,
            c="black",
            alpha=0.1,
            label="validation data",
        )

    ax.set_title(f"Log density | cond_idx={cond_index}", fontsize=fontsizes.get("TITLESIZE"))
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

# Contour plots for specified indices
# MODIFIED: Use n_cond instead of cond_dim for clarity
step = 50
indices = np.arange(0, n_cond, step).tolist()  # Assuming n_cond conditions total

for idx in indices:
    # MODIFIED: Pass both flow and embedding to the plotting function
    fig = plot_model_log_densities(flow, embedding, valset.cpu(), cond_index=idx)
    fig.savefig(os.path.join(save_dir, f"contour_cond_{idx}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# Additional: Plot embeddings to visualize learned condition representations
@torch.no_grad()
def plot_embeddings(embedding_layer, save_path):
    """Visualize the learned embedding space"""
    embedding_layer.eval()
    
    # Get all embeddings
    all_indices = torch.arange(embedding_layer.num_embeddings, device=device)
    embeddings = embedding_layer(all_indices).cpu().numpy()
    
    n_samples, n_features = embeddings.shape
    
    # فقط اگر بیش از 1 کلاس داریم PCA انجام بده
    if n_samples > 1 and n_features > 1:
        try:
            from sklearn.decomposition import PCA
            n_components = min(2, n_samples, n_features)
            pca = PCA(n_components=n_components)
            embeddings_2d = pca.fit_transform(embeddings)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            if n_components == 2:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=all_indices.cpu().numpy(), cmap='viridis', alpha=0.7)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
            else:
                # اگر فقط 1 بعد داریم
                scatter = ax.scatter(embeddings_2d[:, 0], np.zeros_like(embeddings_2d[:, 0]),
                                    c=all_indices.cpu().numpy(), cmap='viridis', alpha=0.7)
                ax.set_xlabel('PC1')
                ax.set_yticks([])
            
            plt.colorbar(scatter, label='Condition Index')
            ax.set_title(f'Embedding Space Visualization (n_components={n_components})')
            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
        except Exception as e:
            print(f"PCA failed: {e}")
    else:
        print(f"Cannot perform PCA: n_samples={n_samples}, n_features={n_features}")
        # می‌توانید embeddingها را مستقیماً plot کنید اگر فقط 1 بعدی هستند
        if n_features == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(embeddings[:, 0], np.zeros_like(embeddings[:, 0]), 
                      c=all_indices.cpu().numpy(), cmap='viridis', alpha=0.7)
            ax.set_xlabel('Embedding Value')
            ax.set_yticks([])
            ax.set_title('1D Embedding Space')
            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

# Plot embedding visualization
plot_embeddings(embedding, os.path.join(save_dir, "embedding_space.png"))

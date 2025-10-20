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
parser.add_argument("--embedding-dim", type=int, default=32)
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

# Keep run name compact: include only batch-size, epochs, embedding-dim and hidden-dim
run_name = f"cond_bs{args.batch_size}_ep{args.epochs}_ed{args.embedding_dim}_hd{args.hidden_dim}"
# Make save directory dataset-aware so results for different datasets go to separate folders
save_dir = os.path.join(args.save_dir, args.dataset, run_name)
os.makedirs(save_dir, exist_ok=True)

# Load protein data
from fff.data import load_dataset
from fff.evaluate.tori import convert_to_angles

# determine root depending on dataset choice; SCOP data sits under ./SCOP
dataset_name = args.dataset
dataset_root = "./fff/data"
if dataset_name.startswith("scop"):
    dataset_root = "."

with suppress_stdout(args.quiet):
    protein_dataset = load_dataset(dataset_name, root=dataset_root, condition_on="residue")
trainset, valset = [convert_to_angles(ds[:][0].to(device)) for ds in protein_dataset]
traincond, valcond = [ds[:][1].to(device) for ds in protein_dataset]
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
    "embedding_dim": args.embedding_dim
}

from torch.utils.data import TensorDataset
trainloader = torch.utils.data.DataLoader(TensorDataset(trainset, traincond), batch_size=config["batch_size"], shuffle=True)
valloader = torch.utils.data.DataLoader(TensorDataset(valset, valcond), batch_size=config["batch_size"], shuffle=True)

# Create embedding layer
embedding = nn.Embedding(num_embeddings=n_cond, embedding_dim=config["embedding_dim"]).to(device)

# Use the same dimension as embedding output for flow condition dimension
cond_dim = config["embedding_dim"]  # This should match embedding dimension
flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)

if args.inspect and not args.quiet:
    print(f"Flow parameters: {sum(p.numel() for p in flow.parameters())}")
    print(f"Embedding parameters: {sum(p.numel() for p in embedding.parameters())}")
    print(f"Condition dimension: {cond_dim}")
    print(f"Number of classes: {n_cond}")

# Optimizer for both models
optimizer = torch.optim.Adam(
    [
        {'params': flow.parameters()},
        {'params': embedding.parameters()}
    ],
    lr=config["lr"]
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=config["lr"], 
    epochs=config["epochs"], 
    steps_per_epoch=len(trainloader)
)
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

from tqdm import trange
pbar = trange(config["epochs"], disable=args.quiet)

best_val = float("inf")
epochs_no_improve = 0
train_losses, val_losses = [], []
best_ckpt_path = os.path.join(save_dir, "best_flow.pt")

# Training loop
for epoch in pbar:
    flow.train()
    embedding.train()
    train_loss = 0.0
    
    for x, c_labels in trainloader:
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            # Convert integer labels to embeddings
            c_labels = c_labels.long().to(device)
            c = embedding(c_labels)  # shape: (batch_size, embedding_dim)
            
            # Calculate loss - flow now expects condition dimension = embedding_dim
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
        embedding.eval()  
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

flow.load_state_dict(ckpt["flow_state_dict"])
embedding.load_state_dict(ckpt["embedding_state_dict"])

def generate_samples(class_id: int, num_samples: int = 1000) -> torch.Tensor:
    """
    Generate samples from the trained conditional flow model for a specific class.
    
    This function samples from the learned probability distribution p(x|class_id)
    where the model has been trained to approximate the data distribution for
    each conditional class.
    
    Args:
        class_id: Integer identifier of the target class (0 to n_classes-1)
        num_samples: Number of samples to generate from the conditional distribution
        
    Returns:
        torch.Tensor: Generated samples with shape (num_samples, 2) containing
                     (φ, ψ) angles in the embedded space representation
                     
    Example:
        >>> # Generate samples for different protein classes
        >>> samples_class_0 = generate_samples(0, 1000)      # Class 0 samples
        >>> samples_class_123 = generate_samples(123, 1000)  # Class 123 samples
    """
    with torch.no_grad():  # Disable gradient computation for inference
        # Set model to evaluation mode
        flow.eval()
        embedding.eval()
        
        # Create condition tensor for the target class
        # Convert class_id to tensor and move to appropriate device (CPU/GPU)
        class_tensor = torch.tensor([class_id], device=device, dtype=torch.long)
        
        # Get embedding vector for the class condition
        # shape: (1, cond_dim) where cond_dim is the embedding dimension
        c = embedding(class_tensor)
        
        # Repeat the condition vector for all samples
        # shape: (num_samples, cond_dim)
        c_repeated = c.repeat(num_samples, 1)
        
        # Sample from the conditional distribution p(x|class_id)
        # The flow model transforms base distribution samples through
        # learned invertible transformations conditioned on c_repeated
        samples = flow(c_repeated).sample()
        
    # Return samples on CPU for further processing or visualization
    return samples.cpu()

if not args.quiet:
    print(f"Best Validation Loss: {best_val:.3f}")
    print(f"Artifacts saved to: {save_dir}")

from typing import Optional
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_model_log_densities(
    model,
    embedding_layer,
    reference_data=None,
    reference_cond=None,  # Condition labels for reference data
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
        reference_cond: Condition labels for the reference data (must be provided with reference_data)
        cond_index: Which condition index to visualize
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
    
    # Use embedding layer for conditioning
    cond_tensor = torch.tensor([cond_index], device=device, dtype=torch.long)
    c_embedding = embedding_layer(cond_tensor)
    
    # Repeat embedding for all grid points
    c = c_embedding.repeat(x.shape[0], 1)
    
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

    # Add reference data if provided - ONLY FOR THE CURRENT cond_index
    if reference_data is not None and reference_cond is not None:
        # Filter data points that belong to the current condition
        mask = (reference_cond == cond_index)
        filtered_data = reference_data[mask]
        
        if len(filtered_data) > 0:
            # Convert ψ values to [0, 2π] range while keeping φ in [-π, π]
            ref_phi = filtered_data[..., 0]
            ref_psi = filtered_data[..., 1] % (2 * torch.pi)  # Wrap ψ to [0, 2π]
            
            ax.scatter(
                ref_phi,
                ref_psi,
                s=7,
                c="red",
                alpha=0.6,
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
# step = 50
# indices = np.arange(0, n_cond, step).tolist()

allset = torch.cat([trainset, valset], dim=0)
allcond = torch.cat([traincond, valcond], dim=0)
n_cond = int(allcond.max().item()) + 1
indices = [100 * i for i in range(5)]
for idx in indices:
    fig = plot_model_log_densities(
        flow,
        embedding_layer=embedding,
        reference_data=allset.cpu(),
        reference_cond=allcond.cpu(),
        cond_index=idx,
    )

    fig.savefig(os.path.join(save_dir, f"contour_cond_{idx}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# Additional: Plot embeddings to visualize learned condition representations
@torch.no_grad()
def plot_embeddings(embedding_layer, save_path):
    """Visualize the learned embedding space"""
    embedding_layer.eval()
    
    # Get all embeddings
    all_indices = torch.arange(embedding_layer.num_embeddings, device=device)
    embeddings = embedding_layer(all_indices)
    
    embeddings = embeddings.cpu().numpy()
    
    n_samples, n_features = embeddings.shape
   
    # Apply PCA 
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
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # scatter plot
        scatter = ax.scatter(embeddings[:, 0], np.zeros_like(embeddings[:, 0]), 
                          c=all_indices.cpu().numpy(), cmap='viridis', alpha=0.7, s=50)
        
        for i, (x, y) in enumerate(zip(embeddings[:, 0], np.zeros_like(embeddings[:, 0]))):
            ax.annotate(f'{i}', (x, y), xytext=(0, 10), 
                       textcoords='offset points', ha='center', fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='Condition Index')
        ax.set_xlabel('Embedding Value')
        ax.set_yticks([])
        ax.set_title(f'1D Embedding Space (n_samples={n_samples})')
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

# Plot embedding visualization
plot_embeddings(embedding, os.path.join(save_dir, "embedding_space.png"))

"""
circularspline_protein_embedding_c_diagnostic.py

This is your ORIGINAL circularspline_protein_embedding_c.py (PC-NCSF), UNCHANGED in every
line that affects training: same architecture, same optimizer/scheduler, same early
stopping, same checkpointing. Nothing about how the model is trained is different.

Only two things were ADDED (each block is clearly marked below), purely for diagnosis:
  1. Unconditional printouts of n_cond and the validation set's composition (previously
     these only printed with --inspect), so we can see exactly how many distinct proteins
     end up in train vs. validation on this machine/dataset copy.
  2. A per-protein NLL CSV (per_protein_nll.csv), computed with EXACTLY the same grouping
     convention as the PC-RFM script's per_protein_nll.csv (mean NLL within each protein
     first), so the two files can be compared/joined directly on `label_id`/`pdb_id`.

Run it exactly the same way you'd run the original script (same flags, same working
directory). The goal is to remove the ambiguity from the PC-RFM run: does this dataset
copy really only put ~136 proteins in validation, or was something different about the
RFM run's data loading?
"""

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

# Keep run name compact: include only batch-size, epochs, embedding-dim and hidden-dim
if args.tag:
    run_name = f"{args.tag}"
else:
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

# =====  ADDED (1/2): unconditional diagnostics -- not gated behind --inspect  =====
n_val_unique = len(valcond.cpu().unique())
n_train_unique = n_cond
print(f"[diagnostic] Number of classes seen in TRAIN labels (n_cond): {n_train_unique}")
print(f"[diagnostic] Number of distinct proteins seen in VAL labels : {n_val_unique}")
print(f"[diagnostic] Total train residues: {len(trainset)}  |  Total val residues: {len(valset)}")
# =====  END ADDED (1/2)  =====

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
    "embedding_dim": args.embedding_dim,
    "dataset": args.dataset,
    "tag": args.tag,
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

# =====  ADDED (2/2): per-protein NLL CSV, same grouping convention as PC-RFM's  =====
# (kept self-contained / duplicated rather than imported from rfm_core.py, so this
#  diagnostic script does not depend on any other file being present)
def _aggregate_per_protein_nll(nll_per_sample, cond_labels):
    out = {}
    labels_cpu = cond_labels.detach().cpu()
    nll_cpu = nll_per_sample.detach().cpu()
    for pid in labels_cpu.unique().tolist():
        mask = labels_cpu == pid
        out[int(pid)] = (nll_cpu[mask].mean().item(), int(mask.sum().item()))
    return out


def _load_protein_id_table(dataset_root, dataset_name):
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
        print(f"[diagnostic] Warning: could not build protein-ID table ({e})")
        return None


protein_id_table = _load_protein_id_table(dataset_root, dataset_name)

flow.eval()
embedding.eval()
val_nlls_list, val_labels_list = [], []
with torch.no_grad():
    for x, c_labels in valloader:
        c_labels = c_labels.long().to(device)
        c = embedding(c_labels)
        nll_batch = -flow(c).log_prob(x)  # exact NLL per residue -- no ODE needed for NCSF
        val_nlls_list.append(nll_batch)
        val_labels_list.append(c_labels)
val_nlls_all = torch.cat(val_nlls_list)
val_labels_all = torch.cat(val_labels_list)
val_nll_flat = val_nlls_all.mean().item()

per_protein = _aggregate_per_protein_nll(val_nlls_all, val_labels_all)
per_protein_csv_path = os.path.join(save_dir, "per_protein_nll.csv")
with open(per_protein_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label_id", "pdb_id", "mean_nll", "n_val_residues"])
    for label_id in sorted(per_protein.keys()):
        mean_nll, n_res = per_protein[label_id]
        pdb_id = protein_id_table[label_id] if protein_id_table is not None and label_id < len(protein_id_table) else ""
        writer.writerow([label_id, pdb_id, mean_nll, n_res])

per_protein_mean_unweighted = sum(v[0] for v in per_protein.values()) / len(per_protein)

print(f"[diagnostic] Validation NLL, flat over residues (Table 2 -style): {val_nll_flat:.4f}  "
      f"(should equal Best Validation Loss above, up to eval/AMP rounding)")
print(f"[diagnostic] Validation NLL, unweighted mean of per-protein means (Table 4/5 -style): {per_protein_mean_unweighted:.4f}")
print(f"[diagnostic] Unique proteins present in per-protein NLL breakdown: {len(per_protein)}")
print(f"[diagnostic] Per-protein NLL saved to: {per_protein_csv_path}")
# =====  END ADDED (2/2)  =====

from fff.train_utils import save_config_and_checkpoint, save_metrics_csv, ensure_run_folder

# Save metrics and checkpoint using train_utils (no plotting here)
run_path = ensure_run_folder(args.save_dir, args.dataset, run_name)
save_metrics_csv(run_path, train_losses, val_losses)
save_config_and_checkpoint(run_path, flow.state_dict(), config, cond_dim, embedding.state_dict())

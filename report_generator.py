import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from fff.data import load_dataset
from fff.evaluate.tori import convert_to_angles
import zuko
import torch.nn as nn
from typing import Optional
import pandas as pd
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

@torch.no_grad()
def plot_model_log_densities(
    model,
    cond_dim: int,
    embedding_layer=None,
    reference_data=None,
    reference_cond=None,
    cond_index: int = 0,
    num_grid_points: int = 200,
    levels: int = 10,
    ax: Optional[plt.Axes] = None,
    fontsizes: dict = dict(TITLESIZE=18, LABELSIZE=16, TICKSIZE=14),
):
    """Plot log density contours for flow models (both unconditional and conditional)."""

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)

    # Create grid
    range_phi = torch.linspace(-torch.pi, torch.pi, num_grid_points)
    range_psi = torch.linspace(0, 2 * torch.pi, num_grid_points)
    phi_grid, psi_grid = torch.meshgrid(range_phi, range_psi, indexing="ij")
    x = torch.stack((phi_grid, psi_grid), dim=-1).to(device).reshape(-1, 2)

    # Conditioning
    if embedding_layer is not None:
        cond_tensor = torch.tensor([cond_index], device=device, dtype=torch.long)
        c_embedding = embedding_layer(cond_tensor)
        c = c_embedding.repeat(x.shape[0], 1)
    else:
        c = torch.zeros(x.shape[0], cond_dim, device=device, dtype=x.dtype)
        c[:, 0] = cond_index

    # Log prob
    log_prob = model(c).log_prob(x).cpu()

    # Plot contour
    phi, psi = x[..., 0].cpu(), x[..., 1].cpu()
    contours = ax.tricontourf(phi, psi, log_prob, levels=levels, cmap="viridis")
    cbar = plt.colorbar(contours)
    cbar.set_label("Log density", fontsize=fontsizes.get("LABELSIZE"))
    cbar.ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    # Limits and ticks
    ax.set_xlim(-torch.pi, torch.pi)
    ax.set_ylim(0, 2 * torch.pi)
    ax.set_xticks(
        [-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    ax.set_yticks(
        [0, torch.pi / 2, torch.pi, 3 * torch.pi / 2, 2 * torch.pi],
        [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
    )
    ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))
    ax.set_xlabel(r"$\Phi$", fontsize=fontsizes.get("LABELSIZE"))
    ax.set_ylabel(r"$\Psi$", fontsize=fontsizes.get("LABELSIZE"))

    # Reference scatter
    if reference_data is not None:
        if embedding_layer is None:
            ref_phi = reference_data[..., 0]
            ref_psi = reference_data[..., 1] % (2 * torch.pi)
            ax.scatter(
                ref_phi,
                ref_psi,
                s=1 / len(reference_data) * 2e3,
                c="black",
                alpha=0.2,
            )
        else:
            if reference_cond is None:
                raise ValueError("reference_cond must be provided for conditional case")
            mask = reference_cond == cond_index
            filtered = reference_data[mask]
            if len(filtered) > 0:
                ref_phi = filtered[..., 0]
                ref_psi = filtered[..., 1] % (2 * torch.pi)
                ax.scatter(ref_phi, ref_psi, s=7, c="red", alpha=0.6)

    title = "Log density"
    if embedding_layer is not None:
        title += f" | cond={cond_index}"
    ax.set_title(title, fontsize=fontsizes.get("TITLESIZE"))

    return ax.figure


# ----------- Reporting -----------
runs_dir = "runs"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# Load full dataset (train + val)
protein_dataset = load_dataset("torus_protein", root="./fff/data", condition_on="residue")
trainset, valset = [convert_to_angles(ds[:][0].to(device)) for ds in protein_dataset]
traincond, valcond = [ds[:][1].to(device) for ds in protein_dataset]

allset = torch.cat([trainset, valset], dim=0)
allcond = torch.cat([traincond, valcond], dim=0)

# Iterate over run folders
for run_folder in os.listdir(runs_dir):
    run_path = os.path.join(runs_dir, run_folder)
    if not os.path.isdir(run_path):
        continue

    ckpt_path = os.path.join(run_path, "best_flow.pt")
    if not os.path.exists(ckpt_path):
        continue

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    cond_dim = ckpt["cond_dim"]

    # ------------------ embedding loader ------------------
    embedding = None
    embedding_dim = config.get("embedding_dim", None)

    if "embedding_state_dict" in ckpt:
        emb_sd = ckpt["embedding_state_dict"]
        weight_tensor = None
        for k, v in emb_sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                weight_tensor = v
                break

        if weight_tensor is not None:
            num_embeddings_ckpt, embedding_dim_ckpt = weight_tensor.shape
            embedding = nn.Embedding(
                num_embeddings=num_embeddings_ckpt,
                embedding_dim=embedding_dim_ckpt,
            ).to(device)
            try:
                embedding.load_state_dict({"weight": weight_tensor})
            except Exception:
                embedding.load_state_dict(emb_sd)
            embedding.eval()
            embedding_dim = embedding.embedding_dim
    # ------------------ end embedding loader ------------------

    # Define file prefix (keep original naming for files)
    if embedding is None:
        file_prefix = f"uncond_bs{batch_size}_ep{epochs}"
        display_name = "FFF"  # For display purposes only
    else:
        file_prefix = f"cond_bs{batch_size}_ep{epochs}_ed{embedding_dim}"
        display_name = "PC-FFF"  # For display purposes only

    # Recreate flow
    flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)
    if "flow_state_dict" in ckpt:
        flow.load_state_dict(ckpt["flow_state_dict"])
    elif "state_dict" in ckpt:
        flow.load_state_dict(ckpt["state_dict"])
    else:
        raise KeyError(f"No flow state_dict found in checkpoint: {ckpt.keys()}")
    flow.eval()

    # ----- Plot densities -----
    if embedding is None:
        # Only one plot for unconditional model
        fig = plot_model_log_densities(
            flow,
            cond_dim,
            embedding_layer=None,
            reference_data=allset.cpu(),
        )
        save_name = f"{file_prefix}.png"
        fig.savefig(os.path.join(reports_dir, save_name), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        # Multiple plots for conditional model
        n_cond = int(allcond.max().item()) + 1
        indices = [100 * i for i in range(5)]
        for idx in indices:
            fig = plot_model_log_densities(
                flow,
                cond_dim,
                embedding_layer=embedding,
                reference_data=allset.cpu(),
                reference_cond=allcond.cpu(),
                cond_index=idx,
            )
            save_name = f"{file_prefix}_cond{idx}.png"
            fig.savefig(os.path.join(reports_dir, save_name), dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"✅ Reports generated for {file_prefix}")


# ----------- Loss plots and CSV summary -----------
all_metrics = []
run_configs = []

for run_folder in os.listdir(runs_dir):
    run_path = os.path.join(runs_dir, run_folder)
    metrics_path = os.path.join(run_path, "metrics.csv")
    ckpt_path = os.path.join(run_path, "best_flow.pt")
    
    if not os.path.exists(metrics_path) or not os.path.exists(ckpt_path):
        continue

    # Load metrics
    df = pd.read_csv(metrics_path)
    df["run"] = run_folder
    
    # Load config to get model parameters
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    
    # Determine model type for display (keep original folder names)
    embedding_dim = config.get("embedding_dim", 0)
    model_type_display = "PC-FFF" if embedding_dim > 0 else "FFF"
    hidden_features = config.get("hidden_features", None)
    num_transforms = config.get("num_transforms", None)

    df["model_type_display"] = model_type_display
    df["original_run_name"] = run_folder
    df["batch_size"] = config["batch_size"]
    df["epochs"] = config["epochs"]
    df["embedding_dim"] = embedding_dim
    df["learning_rate"] = config.get("learning_rate", "N/A")
    df["hidden_features"] = hidden_features
    df["num_transforms"] = num_transforms
    
    all_metrics.append(df)

# Create comprehensive dataframe
if all_metrics:
    full_df = pd.concat(all_metrics, ignore_index=True)
    
    # Group by model type and epoch for plotting
    grouped = full_df.groupby(["model_type_display", "epoch"]).agg(
        train_mean=("train_loss", "mean"),
        train_std=("train_loss", "std"),
        val_mean=("val_loss", "mean"),
        val_std=("val_loss", "std"),
        count=("train_loss", "count")
    ).reset_index()

    # Plot loss curves
    plt.figure(figsize=(10, 8))
    
    # Use different colors for each model type
    colors = {"FFF": "blue", "PC-FFF": "orange"}
    
    for model_type in grouped["model_type_display"].unique():
        model_data = grouped[grouped["model_type_display"] == model_type]
        
        if len(model_data) > 0:
            # Plot training loss
            sns.lineplot(x="epoch", y="train_mean", data=model_data,
                         label=f"{model_type} Train", color=colors[model_type])
            plt.fill_between(model_data["epoch"],
                             model_data["train_mean"] - model_data["train_std"],
                             model_data["train_mean"] + model_data["train_std"],
                             color=colors[model_type], alpha=0.2)
            
            # Plot validation loss
            sns.lineplot(x="epoch", y="val_mean", data=model_data,
                         label=f"{model_type} Val", color=colors[model_type], linestyle="--")
            plt.fill_between(model_data["epoch"],
                             model_data["val_mean"] - model_data["val_std"],
                             model_data["val_mean"] + model_data["val_std"],
                             color=colors[model_type], alpha=0.1)

    plt.xlabel("Epoch")
    plt.ylabel("Loss (NLL)")
    
    # Use (Mean ± Std) in title when multiple files are processed
    num_models = len(full_df["original_run_name"].unique())
    title_suffix = " (Mean ± Std)" if num_models > 2 else ""
    plt.title(f"Training and Validation Loss{title_suffix}")
    
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(reports_dir, "loss_curves_comparison.png"), dpi=150)
    plt.close()

    # Save detailed Excel with multiple sheets
    with pd.ExcelWriter(os.path.join(reports_dir, "loss_curves_summary.xlsx")) as writer:
        # Sheet 1: All data
        full_df.to_excel(writer, sheet_name="All_Models_Data", index=False)
        
        # Sheet 2: Summary statistics by model type
        summary_stats = full_df.groupby(["model_type_display", "epoch"]).agg({
            "train_loss": ["mean", "std", "count"],
            "val_loss": ["mean", "std", "count"]
        }).round(4)
        summary_stats.to_excel(writer, sheet_name="Summary_Statistics")
        
        # Sheet 3: Model configurations
        config_summary = full_df[["original_run_name", "model_type_display", "batch_size", "epochs", 
                                 "embedding_dim", "learning_rate", "hidden_features", 
                                 "num_transforms"]].drop_duplicates()
        config_summary.to_excel(writer, sheet_name="Model_Configurations", index=False)
        
        # Additional sheets for each model type
        for model_type in full_df["model_type_display"].unique():
            model_data = full_df[full_df["model_type_display"] == model_type]
            model_data.to_excel(writer, sheet_name=f"{model_type}_Data", index=False)

    print("✅ Comprehensive Excel report generated with detailed model information")
else:
    print("⚠️ No metrics files found to generate loss curves")
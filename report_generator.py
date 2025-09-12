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

# ----------- تابع رسم عمومی -----------
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


# ----------- گزارش‌گیری -----------
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

    # Define file prefix
    if embedding is None:
        file_prefix = f"uncond_bs{batch_size}_ep{epochs}"
    else:
        file_prefix = f"cond_bs{batch_size}_ep{epochs}_ed{embedding_dim}"

    # Recreate flow
    flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)
    if "flow_state_dict" in ckpt:
        flow.load_state_dict(ckpt["flow_state_dict"])
    elif "state_dict" in ckpt:
        flow.load_state_dict(ckpt["state_dict"])
    else:
        raise KeyError(f"No flow state_dict found in checkpoint: {ckpt.keys()}")
    flow.eval()

    # ----- رسم چگالی -----
    if embedding is None:
        # فقط یک پلات برای مدل بدون شرط
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
        # برای مدل شرطی چند شرط انتخابی
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


# ----------- نمودار Loss و CSV میانگین -----------
metrics_cond = []
metrics_uncond = []

for run_folder in os.listdir(runs_dir):
    run_path = os.path.join(runs_dir, run_folder)
    metrics_path = os.path.join(run_path, "metrics.csv")
    if not os.path.exists(metrics_path):
        continue

    df = pd.read_csv(metrics_path)
    df["run"] = run_folder

    if run_folder.startswith("cond"):
        metrics_cond.append(df)
    elif run_folder.startswith("uncond"):
        metrics_uncond.append(df)

cond_df = pd.concat(metrics_cond, ignore_index=True) if metrics_cond else None
uncond_df = pd.concat(metrics_uncond, ignore_index=True) if metrics_uncond else None

plt.figure(figsize=(8, 6))

out_csv = {}

if cond_df is not None:
    cond_grouped = cond_df.groupby("epoch").agg(
        train_mean=("train_loss", "mean"),
        train_std=("train_loss", "std"),
        val_mean=("val_loss", "mean"),
        val_std=("val_loss", "std"),
    ).reset_index()

    sns.lineplot(x="epoch", y="train_mean", data=cond_grouped,
                 label="Conditional Train", color="blue")
    plt.fill_between(cond_grouped["epoch"],
                     cond_grouped["train_mean"] - cond_grouped["train_std"],
                     cond_grouped["train_mean"] + cond_grouped["train_std"],
                     color="blue", alpha=0.2)

    sns.lineplot(x="epoch", y="val_mean", data=cond_grouped,
                 label="Conditional Val", color="orange")
    plt.fill_between(cond_grouped["epoch"],
                     cond_grouped["val_mean"] - cond_grouped["val_std"],
                     cond_grouped["val_mean"] + cond_grouped["val_std"],
                     color="orange", alpha=0.2)

    out_csv["conditional"] = cond_grouped

if uncond_df is not None:
    uncond_grouped = uncond_df.groupby("epoch").agg(
        train_mean=("train_loss", "mean"),
        train_std=("train_loss", "std"),
        val_mean=("val_loss", "mean"),
        val_std=("val_loss", "std"),
    ).reset_index()

    sns.lineplot(x="epoch", y="train_mean", data=uncond_grouped,
                 label="Unconditional Train", color="green")
    plt.fill_between(uncond_grouped["epoch"],
                     uncond_grouped["train_mean"] - uncond_grouped["train_std"],
                     uncond_grouped["train_mean"] + uncond_grouped["train_std"],
                     color="green", alpha=0.2)

    sns.lineplot(x="epoch", y="val_mean", data=uncond_grouped,
                 label="Unconditional Val", color="red")
    plt.fill_between(uncond_grouped["epoch"],
                     uncond_grouped["val_mean"] - uncond_grouped["val_std"],
                     uncond_grouped["val_mean"] + uncond_grouped["val_std"],
                     color="red", alpha=0.2)

    out_csv["unconditional"] = uncond_grouped

plt.xlabel("Epoch")
plt.ylabel("Loss (NLL)")
plt.title("Training and Validation Loss (Mean ± Std)")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(reports_dir, "loss_curves_comparison.png"), dpi=150)
plt.close()

# ذخیره CSV
with pd.ExcelWriter(os.path.join(reports_dir, "loss_curves_summary.xlsx")) as writer:
    for name, df in out_csv.items():
        df.to_excel(writer, sheet_name=name, index=False)

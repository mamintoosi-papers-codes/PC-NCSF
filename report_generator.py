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
        fontsizes: dict = dict(TITLESIZE=12, LABELSIZE=16, TICKSIZE=14),
    cond_label: Optional[str] = None,
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

    # Title: show only the provided cond_label (which includes protein_name and category)
    # Do not include the 'cond=XXX' prefix to keep titles shorter per user request.
    title = cond_label if cond_label is not None else ""
    ax.set_title(title, fontsize=fontsizes.get("TITLESIZE"))

    return ax.figure

# CLI
import argparse
parser = argparse.ArgumentParser(description="Reporting.")
parser.add_argument("--runs-dir", type=str, default="runs",
                    help="Base runs directory (default 'runs'). If --dataset is provided, looks under runs/<dataset> unless a full path is given.")
parser.add_argument("--reports-dir", type=str, default="reports",
                    help="Base reports directory (default 'reports'). If --dataset is provided, writes to reports/<dataset> unless a full path is given.")
parser.add_argument("--dataset", type=str, default=None,
                    help="Optional dataset name to scope runs and reports (e.g., 'scop_easy' or 'torus_protein').")
args = parser.parse_args()

# ----------- Reporting -----------
runs_dir = args.runs_dir
reports_dir = args.reports_dir
if args.dataset is not None:
    # If the user passed a specific run-folder (e.g. runs/scop_easy/exp01), don't append
    # the dataset again. Detect a specific run-folder by checking for a checkpoint file.
    is_specific_run = os.path.isdir(runs_dir) and os.path.exists(os.path.join(runs_dir, "best_flow.pt"))

    # Only scope runs_dir by dataset when the provided runs_dir is a base directory
    # (for example the default 'runs' or another directory that contains a subfolder
    # named after the dataset).
    if not is_specific_run:
        candidate = os.path.join(runs_dir, args.dataset)
        if os.path.isdir(candidate):
            runs_dir = candidate

    # For reports_dir: prefer a reports/<dataset> folder when it exists or when the
    # user left the default 'reports' base. Otherwise keep the provided reports_dir.
    candidate_reports = os.path.join(reports_dir, args.dataset)
    if reports_dir == "reports" or os.path.isdir(candidate_reports):
        reports_dir = candidate_reports

os.makedirs(reports_dir, exist_ok=True)

def _is_run_folder(path: str) -> bool:
    """Return True if path looks like a single run folder (contains best_flow.pt and metrics.csv)."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "best_flow.pt"))


# If runs_dir points to a single run folder, process just that; otherwise iterate its subfolders
run_paths = []
if _is_run_folder(runs_dir):
    run_paths = [runs_dir]
else:
    run_paths = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)]

for run_path in run_paths:
    if not _is_run_folder(run_path):
        continue

    ckpt_path = os.path.join(run_path, "best_flow.pt")
    if not os.path.exists(ckpt_path):
        continue

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    cond_dim = ckpt["cond_dim"]

    # Load the dataset specified in the checkpoint config if present, else fall back to args.dataset or torus_protein
    ds_name = config.get("dataset", args.dataset if args.dataset is not None else "torus_protein")
    ds_root = "./fff/data"
    if ds_name.startswith("scop"):
        ds_root = "."
    protein_dataset = load_dataset(ds_name, root=ds_root, condition_on="residue")
    trainset, valset = [convert_to_angles(ds[:][0].to(device)) for ds in protein_dataset]
    traincond, valcond = [ds[:][1].to(device) for ds in protein_dataset]

    allset = torch.cat([trainset, valset], dim=0)
    allcond = torch.cat([traincond, valcond], dim=0)

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

    # Define file prefix to match simplified training run naming convention
    # We include only batch-size, epochs, embedding-dim (if conditional) and hidden-dim
    net_cfg = config.get("network", {})
    hidden_features = net_cfg.get("hidden_features", None)
    if isinstance(hidden_features, list) and len(hidden_features) > 0:
        hidden_dim_ckpt = hidden_features[0]
    else:
        hidden_dim_ckpt = config.get("hidden_dim", 0)

    if embedding is None:
        file_prefix = f"uncond_bs{batch_size}_ep{epochs}_hd{hidden_dim_ckpt}"
        display_name = "FFF"
    else:
        file_prefix = f"cond_bs{batch_size}_ep{epochs}_ed{embedding_dim}_hd{hidden_dim_ckpt}"
        display_name = "PC-FFF"

    # Recreate flow
    flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)
    if "flow_state_dict" in ckpt:
        flow.load_state_dict(ckpt["flow_state_dict"])
    elif "state_dict" in ckpt:
        flow.load_state_dict(ckpt["state_dict"])
    else:
        raise KeyError(f"No flow state_dict found in checkpoint: {ckpt.keys()}")
    flow.eval()

    # ----- Prepare mapping for datasets with condition labels (SCOP, TORUS) -----
    scop_mapping = None
    if embedding is not None and ds_name is not None:
        try:
            # SCOP: mapping comes from SCOP CSV under project root
            if ds_name.startswith("scop"):
                subset = ds_name.split("scop_")[-1] if ds_name != "scop" else "easy"
                scop_csv = os.path.join(ds_root, "SCOP", subset, "data.csv")
                if os.path.exists(scop_csv):
                    df_scop = pd.read_csv(scop_csv)
                    # choose protein identifier column
                    if "protein_name" in df_scop.columns:
                        prot_col = "protein_name"
                    elif "pdb_id" in df_scop.columns:
                        prot_col = "pdb_id"
                    elif "domain_id" in df_scop.columns:
                        prot_col = "domain_id"
                    else:
                        id_cols = [c for c in df_scop.columns if "protein" in c.lower() or "id" in c.lower()]
                        prot_col = id_cols[0] if id_cols else None

                    if prot_col is not None:
                        codes, uniques = pd.factorize(df_scop[prot_col])
                        # build category list and protein_name for each unique (take first occurrence)
                        categories = []
                        names = []
                        for i, u in enumerate(uniques):
                            mask = codes == i
                            # prefer 'category' column for labeling, fall back to 'class'
                            if "category" in df_scop.columns:
                                cat = df_scop.loc[mask, "category"].iloc[0]
                            elif "class" in df_scop.columns:
                                cat = df_scop.loc[mask, "class"].iloc[0]
                            else:
                                cat = ""
                            categories.append(str(cat))

                            # prefer to show 'protein_name' in titles; fall back to the unique value
                            if "protein_name" in df_scop.columns:
                                nm = df_scop.loc[mask, "protein_name"].iloc[0]
                            else:
                                nm = u
                            names.append(str(nm))

                        scop_mapping = {i: {"name": names[i], "category": categories[i]} for i in range(len(uniques))}

            # TORUS: mapping comes from the raw TSV under fff/data/raw_data/torus
            elif ds_name.startswith("torus"):
                torus_tsv = os.path.join(ds_root, "raw_data", "torus", "protein.tsv")
                if os.path.exists(torus_tsv):
                    df_torus = pd.read_csv(torus_tsv, delimiter="\t", header=None)
                    # expected columns: name, phi, psi, subtype
                    df_torus.columns = ["name", "phi", "psi", "subtype"]

                    # helper used in the torus dataset loader: extract first token before ':'
                    def _extract_res(name: str) -> str:
                        return str(name).split(":")[0].upper()

                    # Determine classes used when condition_on == 'residue'
                    classes = sorted(set(_extract_res(n) for n in df_torus["name"].values))
                    # Build mapping: for each class, take the first matching row's subtype as category
                    categories = []
                    names = []
                    for i, cls in enumerate(classes):
                        mask = df_torus["name"].apply(lambda n: _extract_res(n) == cls)
                        names.append(cls)
                        if mask.any() and "subtype" in df_torus.columns:
                            # take first subtype occurrence for this class
                            categories.append(str(df_torus.loc[mask, "subtype"].iloc[0]))
                        else:
                            categories.append("")

                    scop_mapping = {i: {"name": names[i], "category": categories[i]} for i in range(len(classes))}
        except Exception as e:
            print(f"Warning: Failed to create SCOP/torus mapping: {e}")
            scop_mapping = None
            # Reuse any existing `scop_mapping` created earlier. The mapping was prepared
            # above (if applicable) to avoid reading the SCOP CSV multiple times.
            # If no mapping is available, `scop_mapping` remains whichever value was
            # set earlier (possibly None) and the code below will fall back to
            # deriving condition count from the condition tensor.

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
        # Multiple plots for conditional model: pick up to 5 evenly spaced condition indices
        # For SCOP datasets, use number of unique proteins from mapping
        if scop_mapping is not None:
            n_cond = len(scop_mapping)
        else:
            n_cond = int(allcond.max().item()) + 1

        num_to_plot = min(5, n_cond)
        import numpy as _np
        indices = _np.linspace(0, n_cond - 1, num=num_to_plot, dtype=int).tolist()

        # The `scop_mapping` (if needed) is created earlier in the "Prepare SCOP mapping"
        # block above. We avoid re-reading the SCOP CSV here to prevent duplication
        # and potential inconsistencies — just reuse `scop_mapping` as populated
        # above (or None if unavailable).

        for idx in indices:
            # determine cond_label when available for SCOP
            cond_label = None
            if scop_mapping is not None and int(idx) in scop_mapping:
                info = scop_mapping[int(idx)]
                # sanitize strings for title/filename
                def _safe(s):
                    return str(s).replace(" ", "_").replace("/", "_").replace("\\\\", "_")

                cond_label = f"name={info['name']} | category={info.get('category', '')}"

            fig = plot_model_log_densities(
                flow,
                cond_dim,
                embedding_layer=embedding,
                reference_data=allset.cpu(),
                reference_cond=allcond.cpu(),
                cond_index=int(idx),
                cond_label=cond_label,
            )
            # Always keep filenames simple: include only cond index to avoid filesystem issues
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
    # Determine model type from checkpoint contents: conditional models include an embedding_state_dict
    model_type_display = "PC-FFF" if "embedding_state_dict" in ckpt else "FFF"
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

    # Ensure numeric stds (NaN -> 0) so fill_between works when only one sample/epoch exists
    grouped["train_std"] = grouped["train_std"].fillna(0.0)
    grouped["val_std"] = grouped["val_std"].fillna(0.0)

    # Prepare epochs axis (sorted unique epochs across all runs)
    epoch_axis = sorted(full_df["epoch"].unique())

    # Extract series for unconditional (FFF) and conditional (PC-FFF)
    cond_group = grouped[grouped["model_type_display"] == "PC-FFF"].set_index("epoch")
    uncond_group = grouped[grouped["model_type_display"] == "FFF"].set_index("epoch")

    # Helper to build arrays aligned to epoch_axis
    def _arr_for(group_df, col):
        return [group_df[col].get(e, float('nan')) for e in epoch_axis]

    train_uncond = _arr_for(uncond_group, "train_mean")
    val_uncond = _arr_for(uncond_group, "val_mean")
    train_cond = _arr_for(cond_group, "train_mean")
    val_cond = _arr_for(cond_group, "val_mean")

    train_uncond_std = _arr_for(uncond_group, "train_std")
    val_uncond_std = _arr_for(uncond_group, "val_std")
    train_cond_std = _arr_for(cond_group, "train_std")
    val_cond_std = _arr_for(cond_group, "val_std")

    # Plot single figure with four series
    plt.figure(figsize=(12, 9))
    plt.rcParams['font.size'] = 16
    colors = {"uncond": "red", "cond": "blue"}

    epoch_arr = epoch_axis

    # Plot Unconditional Train/Val
    plt.plot(epoch_arr, train_uncond, label="Unconditional Train", color=colors['uncond'], marker='o', linewidth=2.5)
    plt.fill_between(epoch_arr, [a - b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(train_uncond, train_uncond_std)],
                     [a + b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(train_uncond, train_uncond_std)],
                     color=colors['uncond'], alpha=0.15)
    plt.plot(epoch_arr, val_uncond, label="Unconditional Val", color=colors['uncond'], linestyle='--', marker='s', linewidth=2.5)
    plt.fill_between(epoch_arr, [a - b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(val_uncond, val_uncond_std)],
                     [a + b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(val_uncond, val_uncond_std)],
                     color=colors['uncond'], alpha=0.08)

    # Plot Conditional Train/Val
    plt.plot(epoch_arr, train_cond, label="Conditional Train", color=colors['cond'], marker='o', linewidth=2.5)
    plt.fill_between(epoch_arr, [a - b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(train_cond, train_cond_std)],
                     [a + b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(train_cond, train_cond_std)],
                     color=colors['cond'], alpha=0.15)
    plt.plot(epoch_arr, val_cond, label="Conditional Val", color=colors['cond'], linestyle='--', marker='s', linewidth=2.5)
    plt.fill_between(epoch_arr, [a - b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(val_cond, val_cond_std)],
                     [a + b if not (isinstance(a, float) and np.isnan(a)) else np.nan for a, b in zip(val_cond, val_cond_std)],
                     color=colors['cond'], alpha=0.08)

    plt.xlabel("Epoch", fontsize=20, fontweight='bold')
    plt.ylabel("Loss (NLL)", fontsize=20, fontweight='bold')
    plt.title("Training and Validation Loss: Unconditional vs Conditional", fontsize=22, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "loss_curves_comparison.png"), dpi=300)
    plt.savefig(os.path.join(reports_dir, "loss_curves_comparison.pdf"), bbox_inches='tight')
    plt.close()

    # (Plot already labeled and saved above.)

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
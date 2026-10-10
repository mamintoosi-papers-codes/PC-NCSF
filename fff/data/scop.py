import os
import random
from typing import Optional

import pandas as pd
import torch
try:
    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.geometry.product_manifold import ProductManifold
except ImportError:
    # geomstats (or one of its deps) is unavailable -- e.g. geomstats 2.8
    # breaks on numpy>=2 because it imports the removed numpy.trapz. The
    # manifold is only stored on the dataset (see ManifoldDataset) and is
    # never used in the data path, so a None sentinel is a safe fallback.
    Hypersphere = None
    ProductManifold = None
from torch import Tensor
from torch.utils.data import TensorDataset

from .utils import split_dataset
from .manifold import ManifoldDataset


def embed_angle_in_2d(angle: Tensor) -> Tensor:
    x, y = torch.cos(angle), torch.sin(angle)
    return torch.stack([x, y], dim=-1)


def get_scop_dataset(
    subset: str = "easy",
    seed: int = random.randint(0, 2 ** 32 - 1),
    root: Optional[str] = None,
    condition_on: Optional[str] = "protein",
):
    """Load SCOP angular data and return (train, val[, test]) ManifoldDataset tuples.

    Args:
        subset: one of the SCOP subfolders (e.g. 'easy', 'moderate', ...).
        seed: RNG seed used for splitting.
        root: base path where the `SCOP` folder lives. If None, uses project root '.'
        condition_on: which column to use for conditioning; default tries to use a protein id column.

    The function expects a CSV at <root>/SCOP/<subset>/data.csv with two angle columns
    (theta/tau or phi/psi) and an identifier column like 'pdb_id' or 'domain_id'.
    """
    if root is None:
        root = "."

    file_path = os.path.join(root, "SCOP", subset, "data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SCOP data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Determine angle columns
    if "theta" in df.columns and "tau" in df.columns:
        phi_col, psi_col = "theta", "tau"
    elif "phi" in df.columns and "psi" in df.columns:
        phi_col, psi_col = "phi", "psi"
    else:
        # try to find two columns that look like angles
        candidates = [c for c in df.columns if any(k in c.lower() for k in ["phi", "psi", "theta", "tau"]) ]
        if len(candidates) >= 2:
            phi_col, psi_col = candidates[0], candidates[1]
        else:
            raise ValueError("Could not find angle columns (expected 'theta'/'tau' or 'phi'/'psi') in SCOP CSV")

    # Determine protein identifier column
    if condition_on and condition_on in df.columns:
        protein_col = condition_on
    elif "pdb_id" in df.columns:
        protein_col = "pdb_id"
    elif "domain_id" in df.columns:
        protein_col = "domain_id"
    else:
        id_cols = [c for c in df.columns if "protein" in c.lower() or "id" in c.lower()]
        if not id_cols:
            raise ValueError("No protein identifier column found in SCOP CSV")
        protein_col = id_cols[0]

    # Drop rows with missing angle values to avoid NaNs propagating through embeddings
    initial_len = len(df)
    df = df.dropna(subset=[phi_col, psi_col]).reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"get_scop_dataset: dropped {dropped} rows with missing angles from {file_path}")

    # Extract angles and encode protein IDs
    angles = torch.tensor(df[[phi_col, psi_col]].values, dtype=torch.float32)
    # Convert degrees to radians
    angles = angles * 2 * torch.pi / 360.0

    # embed each angle in 2D (cos,sin) and build torus embedding
    phi_embedding = embed_angle_in_2d(angles[:, 0]).unsqueeze(-1)
    psi_embedding = embed_angle_in_2d(angles[:, 1]).unsqueeze(-1)
    data = torch.cat([phi_embedding, psi_embedding], dim=-1).transpose(1, 2)

    idx = torch.tensor(pd.factorize(df[protein_col])[0], dtype=torch.long)

    # Create condition vector as integer class labels (not one-hot)
    cond = idx

    # Split data using provided utility so splits align
    if cond is None:
        train_data, val_data = split_dataset(data, seed=seed)
        train_ds = TensorDataset(train_data)
        val_ds = TensorDataset(val_data)
    else:
        train_data, val_data = split_dataset(data, seed=seed)
        train_cond, val_cond = split_dataset(cond, seed=seed)
        train_ds = TensorDataset(train_data, train_cond)
        val_ds = TensorDataset(val_data, val_cond)

    manifold = (ProductManifold([Hypersphere(1), Hypersphere(1)])
                if ProductManifold is not None else None)
    return (
        ManifoldDataset(train_ds, manifold),
        ManifoldDataset(val_ds, manifold),
    )

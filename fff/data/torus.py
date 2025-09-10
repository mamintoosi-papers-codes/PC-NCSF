import os

import random
import numpy as np
import pandas as pd
import torch
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from .utils import split_dataset
from .manifold import ManifoldDataset


def embed_angle_in_2d(angle: Tensor) -> Tensor:
    x, y = torch.cos(angle), torch.sin(angle)
    return torch.stack([x, y], dim=-1)


def get_torus_protein_dataset(
    subtype: str | None = None,
    seed: int = random.randint(0, 2**32 - 1),
    root: str = "./fff/data",
    condition_on: str | None = None,
):
    # print(f"Dataset seed: {seed}")
    file_path = os.path.join(root, "raw_data", "torus", "protein.tsv")

    raw_data = pd.read_csv(file_path, delimiter="\t", header=None)
    raw_data.columns = ["name", "phi", "psi", "subtype"]

    # Select rows according to conditioning and/or subtype
    if condition_on is None:
        if subtype is None:
            used = raw_data
        else:
            used = raw_data[raw_data["subtype"] == subtype]
    else:
        used = raw_data

    # Embed angles on the torus (S1 x S1)
    phi_embedding = embed_angle_in_2d(
        torch.tensor(used["phi"].values) * 2 * torch.pi / 360
    ).unsqueeze(-1)
    psi_embedding = embed_angle_in_2d(
        torch.tensor(used["psi"].values) * 2 * torch.pi / 360
    ).unsqueeze(-1)
    data = torch.cat([phi_embedding, psi_embedding], dim=-1).transpose(1, 2)

    # Optional condition vector
    cond = None
    if condition_on is not None:
        if condition_on == "residue":
            # Extract three-letter residue code (e.g., GLY, PRO) from the name field
            def _extract_res(name: str) -> str:
                tail = str(name).split(":")[-1]
                res = "".join([ch for ch in tail if ch.isalpha()])
                return res.upper()

            classes = sorted(set(_extract_res(n) for n in used["name"].values))
            
            idx = torch.tensor([classes.index(_extract_res(n)) for n in used["name"].values], dtype=torch.long)
        elif condition_on == "subtype":
            classes = sorted(set(used["subtype"].values))
            idx = torch.tensor([classes.index(s) for s in used["subtype"].values], dtype=torch.long)
        else:
            raise ValueError(f"Unknown condition_on={condition_on!r}. Use 'residue' or 'subtype'.")

        # cond = F.one_hot(idx, num_classes=len(classes)).to(dtype=data.dtype)
        # print(len(classes), len(idx), len(cond))
        # print(classes[:5], idx[:5], cond[:5])

        # به جای یک داغ همان شماره کلاس را در نظر می‌گیرم
        cond = idx

    # Split data (and conditions, if present) with identical seeds to align splits
    if cond is None:
        train_data, val_data, test_data = split_dataset(data, seed=seed)
        train_ds = TensorDataset(train_data)
        val_ds = TensorDataset(val_data)
        test_ds = TensorDataset(test_data)
    else:
        train_data, val_data, test_data = split_dataset(data, seed=seed)
        train_cond, val_cond, test_cond = split_dataset(cond, seed=seed)
        train_ds = TensorDataset(train_data, train_cond)
        val_ds = TensorDataset(val_data, val_cond)
        test_ds = TensorDataset(test_data, test_cond)

    manifold = ProductManifold([Hypersphere(1), Hypersphere(1)])
    return (
        ManifoldDataset(train_ds, manifold),
        ManifoldDataset(val_ds, manifold),
        ManifoldDataset(test_ds, manifold),
    )


def get_torus_rna_dataset(
    root: str = "./fff/data", seed: int = random.randint(0, 2**32 - 1)
):
    print(f"Dataset seed: {seed}")
    file_path = os.path.join(root, "raw_data", "torus", "rna.tsv")

    raw_data = pd.read_csv(file_path, delimiter="\t", header=None)
    raw_data.columns = [
        "pdb_id",
        "resname",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "chi",
    ]

    embeddings = []
    for angle in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]:
        raw_data[angle] = raw_data[angle] * 2 * torch.pi / 360
        embeddings.append(
            embed_angle_in_2d(torch.tensor(raw_data[angle].values)).unsqueeze(-1)
        )
    data = torch.cat(embeddings, dim=-1).transpose(1, 2)

    train_data, val_data, test_data = split_dataset(data, seed=seed)

    manifold = ProductManifold([Hypersphere(1) for _ in range(7)])
    return (
        ManifoldDataset(TensorDataset(train_data), manifold),
        ManifoldDataset(TensorDataset(val_data), manifold),
        ManifoldDataset(TensorDataset(test_data), manifold),
    )

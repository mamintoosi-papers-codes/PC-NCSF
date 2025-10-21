import os
import json
import torch
import numpy as np
import random

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_run_folder(save_dir: str, dataset: str, run_name: str):
    path = os.path.join(save_dir, dataset, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_config_and_checkpoint(path: str, flow_state: dict, config: dict, cond_dim: int = 1, embedding_state: dict = None):
    ckpt = {"config": config, "cond_dim": cond_dim}
    # prefer standardized keys
    ckpt["flow_state_dict"] = flow_state
    if embedding_state is not None:
        ckpt["embedding_state_dict"] = embedding_state
    torch.save(ckpt, os.path.join(path, "best_flow.pt"))

    # save a json copy of config for easy inspection
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)


def save_metrics_csv(path: str, train_losses: list, val_losses: list):
    import csv
    metrics_path = os.path.join(path, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow([i, tr, va])

    return metrics_path

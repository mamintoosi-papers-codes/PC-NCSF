#!/usr/bin/env python
import os
import argparse
import torch
from fff.data import load_dataset
from fff.evaluate.tori import convert_to_angles
import zuko
import torch.nn as nn
from fff.train_utils import ensure_run_folder, save_metrics_csv, save_config_and_checkpoint, set_seed


def train_uncond(trainset, valset, config, run_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_uncond] device={device}, run_path={run_path}")
    from contextlib import nullcontext
    autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=config["batch_size"], shuffle=False)
    print(f"[train_uncond] train batches={len(trainloader)}, val batches={len(valloader)}")

    cond_dim = 1
    flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], epochs=config["epochs"], steps_per_epoch=max(1, len(trainloader)))
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_val = float("inf")
    epochs_no_imp = 0
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        flow.train()
        t_loss = 0.0
        for x in trainloader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                with autocast_ctx():
                    c = torch.zeros_like(x[:, :1])
                    loss = -flow(c).log_prob(x).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU path
                c = torch.zeros_like(x[:, :1])
                loss = -flow(c).log_prob(x).mean()
                loss.backward()
                optimizer.step()
            if len(trainloader) > 0:
                try:
                    scheduler.step()
                except Exception:
                    pass
            t_loss += loss.item()
        t_loss = t_loss / max(1, len(trainloader))
        train_losses.append(t_loss)

        # validation
        flow.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x in valloader:
                x = x.to(device)
                c = torch.zeros_like(x[:, :1])
                loss = -flow(c).log_prob(x).mean()
                v_loss += loss.item()
        v_loss = v_loss / max(1, len(valloader))
        val_losses.append(v_loss)

        if v_loss < best_val - 1e-6:
            best_val = v_loss
            epochs_no_imp = 0
            save_config_and_checkpoint(run_path, flow.state_dict(), config, cond_dim)
            print("Saved checkpoint to", os.path.join(run_path, "best_flow.pt"))
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= config.get("patience", 10):
            break

    # final save (ensure checkpoint exists)
    save_metrics_csv(run_path, train_losses, val_losses)
    print("Saved metrics to", os.path.join(run_path, "metrics.csv"))
    save_config_and_checkpoint(run_path, flow.state_dict(), config, cond_dim)
    print("Saved final checkpoint to", os.path.join(run_path, "best_flow.pt"))
    return


def train_cond(trainset, valset, traincond, valcond, config, run_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_cond] device={device}, run_path={run_path}")
    from torch.utils.data import TensorDataset
    trainloader = torch.utils.data.DataLoader(TensorDataset(trainset, traincond), batch_size=config["batch_size"], shuffle=True)
    valloader = torch.utils.data.DataLoader(TensorDataset(valset, valcond), batch_size=config["batch_size"], shuffle=False)
    print(f"[train_cond] train batches={len(trainloader)}, val batches={len(valloader)}, n_cond={int(traincond.max().item())+1}")

    n_cond = int(traincond.max().item()) + 1
    embedding = nn.Embedding(num_embeddings=n_cond, embedding_dim=config["embedding_dim"]).to(device)
    cond_dim = config["embedding_dim"]
    flow = zuko.flows.NCSF(2, cond_dim, **config["network"]).to(device)

    optimizer = torch.optim.Adam([{'params': flow.parameters()}, {'params': embedding.parameters()}], lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], epochs=config["epochs"], steps_per_epoch=max(1, len(trainloader)))
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_val = float("inf")
    epochs_no_imp = 0
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        flow.train(); embedding.train()
        t_loss = 0.0
        for x, c_labels in trainloader:
            x = x.to(device)
            c_labels = c_labels.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    c = embedding(c_labels)
                    loss = -flow(c).log_prob(x).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                c = embedding(c_labels)
                loss = -flow(c).log_prob(x).mean()
                loss.backward()
                optimizer.step()
            if len(trainloader) > 0:
                try:
                    scheduler.step()
                except Exception:
                    pass
            t_loss += loss.item()
        t_loss = t_loss / max(1, len(trainloader))
        train_losses.append(t_loss)

        flow.eval(); embedding.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x, c_labels in valloader:
                x = x.to(device)
                c_labels = c_labels.long().to(device)
                c = embedding(c_labels)
                loss = -flow(c).log_prob(x).mean()
                v_loss += loss.item()
        v_loss = v_loss / max(1, len(valloader))
        val_losses.append(v_loss)

        if v_loss < best_val - 1e-6:
            best_val = v_loss
            epochs_no_imp = 0
            save_config_and_checkpoint(run_path, flow.state_dict(), config, cond_dim, embedding.state_dict())
            print("Saved checkpoint to", os.path.join(run_path, "best_flow.pt"))
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= config.get("patience", 10):
            break

    save_metrics_csv(run_path, train_losses, val_losses)
    print("Saved metrics to", os.path.join(run_path, "metrics.csv"))
    save_config_and_checkpoint(run_path, flow.state_dict(), config, cond_dim, embedding.state_dict())
    print("Saved final checkpoint to", os.path.join(run_path, "best_flow.pt"))
    return


def main():
    parser = argparse.ArgumentParser(description="Unified trainer for uncond/cond flows")
    parser.add_argument("--mode", choices=["uncond", "cond", "both"], default="both")
    parser.add_argument("--dataset", type=str, default="torus_protein")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--transforms", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    network = {"hidden_features": [args.hidden_dim] * args.hidden_layers, "transforms": args.transforms}
    config = {"lr": args.lr, "epochs": args.epochs, "network": network, "batch_size": args.batch_size, "patience": args.patience, "seed": args.seed, "dataset": args.dataset, "tag": args.tag, "embedding_dim": args.embedding_dim}

    # Determine run name
    if args.tag:
        run_name = args.tag
    else:
        run_name = f"bs{args.batch_size}_ep{args.epochs}_hd{args.hidden_dim}"

    # Load dataset
    ds_root = "./fff/data"
    if args.dataset.startswith("scop"):
        ds_root = "."
    dataset = load_dataset(args.dataset, root=ds_root, condition_on="residue" if args.mode != "uncond" else None)
    if args.mode == "uncond":
        trainset, valset = [convert_to_angles(ds[:][0].to("cpu")) for ds in dataset]
        print(f"[main] Loaded uncond dataset: train={len(trainset)}, val={len(valset)}")
    else:
        # conditional and both: dataset returns ManifoldDataset objects wrapping TensorDataset
        # extract the underlying tensors by slicing each dataset (ds[:] returns a tuple of tensors)
        ((tr_x, tr_c), (va_x, va_c)) = [tuple(ds[:]) for ds in dataset]
        trainset, traincond = convert_to_angles(tr_x.to("cpu")), tr_c.to("cpu")
        valset, valcond = convert_to_angles(va_x.to("cpu")), va_c.to("cpu")
        print(f"[main] Loaded cond dataset: train={len(trainset)}, train_cond={len(traincond)}, val={len(valset)}, val_cond={len(valcond)}")

    base_run = ensure_run_folder(args.save_dir, args.dataset, run_name)

    if args.mode in ("uncond", "both"):
        uncond_path = os.path.join(base_run, "uncond")
        os.makedirs(uncond_path, exist_ok=True)
        print("Running unconditional ->", uncond_path)
        # train uncond using CPU tensors converted inside
        train_uncond(trainset, valset, config, uncond_path)

    if args.mode in ("cond", "both"):
        cond_path = os.path.join(base_run, "cond")
        os.makedirs(cond_path, exist_ok=True)
        print("Running conditional ->", cond_path)
        train_cond(trainset, valset, traincond, valcond, config, cond_path)

    print("Done")


if __name__ == "__main__":
    main()

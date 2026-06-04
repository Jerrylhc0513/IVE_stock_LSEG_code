"""
train_paper.py — Training script for the IVE-style replication model

Aligned with Section 3.2 of the IVE paper:
  - AdamW optimizer
  - Learning rate = 3e-4
  - Student-t negative log-likelihood loss
  - Multi-step training and single-step evaluation
  - Cosine learning rate schedule and gradient clipping
"""

import argparse
import numpy as np
import torch
from torch.optim import AdamW

from data_lseg import prepare_dataloaders, FEATURE_COLS
from model import PaperIVE, student_t_nll


def move_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    total_nll, total_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            mu, sigma, nu = model(
                batch["yesterday"], batch["today"],
                batch["today_mask"], batch["stock_id"],
                batch["day_idx"],
            )
            y1 = batch["target"][:, 0]
            mu1, sigma1, nu1 = mu[:, 0], sigma[:, 0], nu[:, 0]
            nll = student_t_nll(
                y1.unsqueeze(-1),
                mu1.unsqueeze(-1),
                sigma1.unsqueeze(-1),
                nu1.unsqueeze(-1),
            ).item()
            mae = (mu1 - y1).abs().mean().item()
            bs = y1.size(0)
            total_nll += nll * bs
            total_mae += mae * bs
            n += bs
    return total_nll / n, total_mae / n


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")

    train_dl, val_dl, train_ds, stock_to_id = prepare_dataloaders(
        parquet_path=args.parquet_path,
        batch_size=args.batch_size,
        val_ratio=0.15,
        multi_step=args.multi_step,
    )

    model = PaperIVE(
        n_features=len(FEATURE_COLS),
        n_stocks=len(stock_to_id),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        multi_step=args.multi_step,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")
    print(f"[train] multi_step: {args.multi_step}")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_dl:
            batch = move_to_device(batch, device)
            mu, sigma, nu = model(
                batch["yesterday"], batch["today"],
                batch["today_mask"], batch["stock_id"],
                batch["day_idx"],
            )
            loss = student_t_nll(batch["target"], mu, sigma, nu)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            losses.append(loss.item())
        sched.step()

        train_nll = np.mean(losses)
        val_nll, val_mae = evaluate(model, val_dl, device)
        marker = ""
        if val_nll < best_val:
            best_val = val_nll
            ckpt = {
                "model": model.state_dict(),
                "mean": train_ds.mean,
                "std": train_ds.std,
                "stock_to_id": stock_to_id,
                "args": vars(args),
            }
            torch.save(ckpt, args.save_path)
            marker = "  saved"
        print(f"[epoch {epoch:02d}] train_nll={train_nll:.4f} | "
              f"val_nll={val_nll:.4f} | val_mae={val_mae:.4f}{marker}")

    print(f"\n[done] best val_nll = {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_path", default="all_stocks_1m.parquet")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--multi_step", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--save_path", default="best_paper.pt")
    args = p.parse_args()
    train(args)
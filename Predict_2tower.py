"""
Predict_2tower.py — Inference and visualisation for the IVE-style replication model

Load a trained PaperIVE checkpoint, run predictions on the validation set,
generate per-stock prediction plots, and compute evaluation metrics.

Following the paper-style setup, only the first-step prediction is used
during inference and evaluation.
"""

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data_lseg import prepare_dataloaders, FEATURE_COLS
from model import PaperIVE


CKPT_PATH = "best_paper.pt"
OUT_FIG = "predictions_all_stocks.png"
OUT_METRICS = "prediction_metrics.txt"
N_PLOT_PER_STOCK = 400


def predict_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[predict] device = {device}")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    args = ckpt["args"]
    stock_to_id = ckpt["stock_to_id"]
    id_to_stock = {v: k for k, v in stock_to_id.items()}

    train_dl, val_dl, _, _ = prepare_dataloaders(
        parquet_path=args["parquet_path"],
        batch_size=args["batch_size"],
        val_ratio=0.15,
        multi_step=args.get("multi_step", 3),
    )

    model = PaperIVE(
        n_features=len(FEATURE_COLS),
        n_stocks=len(stock_to_id),
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        multi_step=args.get("multi_step", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[predict] model loaded with {n_params:,} parameters")

    print("[predict] running predictions...")
    all_results = {ric: {"y": [], "mu": [], "sigma": []} for ric in stock_to_id}

    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            mu, sigma, nu = model(
                batch["yesterday"], batch["today"],
                batch["today_mask"], batch["stock_id"],
                batch["day_idx"],
            )
            y1 = batch["target"][:, 0]
            mu1 = mu[:, 0]
            sigma1 = sigma[:, 0]
            for i in range(y1.size(0)):
                sid = batch["stock_id"][i].item()
                ric = id_to_stock[sid]
                all_results[ric]["y"].append(y1[i].cpu().item())
                all_results[ric]["mu"].append(mu1[i].cpu().item())
                all_results[ric]["sigma"].append(sigma1[i].cpu().item())

    metrics = {}
    for ric in all_results:
        y = np.array(all_results[ric]["y"])
        mu = np.array(all_results[ric]["mu"])
        sigma = np.array(all_results[ric]["sigma"])
        if len(y) == 0:
            continue
        mae = np.abs(mu - y).mean()
        rmse = np.sqrt(((mu - y) ** 2).mean())
        coverage = ((y >= mu - sigma) & (y <= mu + sigma)).mean()
        metrics[ric] = {
            "n": len(y), "mae": mae, "rmse": rmse, "coverage": coverage,
            "y": y, "mu": mu, "sigma": sigma,
        }

    print("\n" + "=" * 70)
    print(f"{'Stock':<10} {'Samples':>8} {'MAE':>10} {'RMSE':>10} {'1-sigma cov.':>12}")
    print("=" * 70)
    lines = ["Stock / Samples / MAE / RMSE / 1-sigma coverage\n" + "=" * 60 + "\n"]
    for ric in sorted(metrics.keys()):
        m = metrics[ric]
        line = (f"{ric:<10} {m['n']:>8} {m['mae']:>10.4f} "
                f"{m['rmse']:>10.4f} {m['coverage']:>11.2%}")
        print(line)
        lines.append(line + "\n")

    all_y = np.concatenate([metrics[r]["y"] for r in metrics])
    all_mu = np.concatenate([metrics[r]["mu"] for r in metrics])
    all_sigma = np.concatenate([metrics[r]["sigma"] for r in metrics])
    global_mae = np.abs(all_mu - all_y).mean()
    global_rmse = np.sqrt(((all_mu - all_y) ** 2).mean())
    global_cov = ((all_y >= all_mu - all_sigma) & (all_y <= all_mu + all_sigma)).mean()
    print("-" * 70)
    glob = (f"{'Global':<10} {len(all_y):>8} {global_mae:>10.4f} "
            f"{global_rmse:>10.4f} {global_cov:>11.2%}")
    print(glob)
    lines.append("-" * 60 + "\n" + glob + "\n")
    print("=" * 70)

    Path(OUT_METRICS).write_text("".join(lines), encoding="utf-8")
    print(f"\n[predict] metrics saved to: {OUT_METRICS}")

    print("[predict] generating plots...")
    n_stocks = len(metrics)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(14, 3 * n_stocks))
    if n_stocks == 1:
        axes = [axes]

    for ax, ric in zip(axes, sorted(metrics.keys())):
        m = metrics[ric]
        n = min(N_PLOT_PER_STOCK, len(m["y"]))
        y, mu, sigma = m["y"][:n], m["mu"][:n], m["sigma"][:n]

        ax.plot(y, label="Target", color="#D85A30", alpha=0.7, lw=0.8)
        ax.plot(mu, label="Pred mean", color="#185FA5", lw=1.2)
        ax.fill_between(range(n), mu - sigma, mu + sigma,
                        alpha=0.2, color="#185FA5", label="+/-1 sigma")
        ax.set_title(
            f"{ric}  |  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
            f"coverage={m['coverage']:.1%}",
            fontsize=11,
        )
        ax.set_ylabel("log(volume_ratio)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("validation sample index")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    print(f"[predict] plot saved to: {OUT_FIG}")
    print(f"\nDone. Open {OUT_FIG} to view the results.")


if __name__ == "__main__":
    predict_all()
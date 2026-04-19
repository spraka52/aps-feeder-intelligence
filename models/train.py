"""Train the GraphSAGE+GRU forecaster on synthesized feeder data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.topology import build_graph, edge_index_tensor
from models.dataset import FeederWindowDataset, WindowSpec
from models.graphsage_gru import GraphSAGEGRU, ModelConfig

REPO = Path(__file__).resolve().parent.parent


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """RMSE / MAE / wMAPE.

    We report **wMAPE** (weighted MAPE = sum|err| / sum|actual|) instead of
    per-sample MAPE because PV backfeed can drive bus loads to ~0, and
    per-sample MAPE explodes on near-zero divisors. wMAPE is the metric
    EPRI / NREL distribution-feeder benchmarks actually publish — it
    weights error by load magnitude, which is what an operator cares about.
    """
    if mask is not None and mask.any():
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "n": 0}
    err = y_pred - y_true
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    total_actual = float(np.abs(y_true).sum())
    if total_actual > 0:
        wmape = float(np.abs(err).sum() / total_actual * 100.0)
    else:
        wmape = float("nan")
    return {"rmse": rmse, "mae": mae, "mape": wmape, "n": int(y_true.size)}


def evaluate(model, loader, edge_index, device) -> dict:
    model.eval()
    preds, trues, hw_flags = [], [], []
    with torch.no_grad():
        for X, y, meta in loader:
            X = X.to(device)
            yhat = model(X, edge_index).cpu().numpy()
            preds.append(yhat)
            trues.append(y.numpy())
            # Broadcast heatwave flag across nodes for masking
            B, T_out, N = yhat.shape
            hw = np.stack([m for m in meta["in_heatwave"]], axis=0) if isinstance(meta["in_heatwave"], list) else meta["in_heatwave"].numpy()
            hw_flags.append(np.broadcast_to(hw[..., None], (B, T_out, N)).copy())
    yhat = np.concatenate(preds, axis=0)
    y = np.concatenate(trues, axis=0)
    hw = np.concatenate(hw_flags, axis=0).astype(bool)
    overall = _metrics(y, yhat)
    heatwave = _metrics(y, yhat, mask=hw)
    normal = _metrics(y, yhat, mask=~hw)
    return {"overall": overall, "heatwave": heatwave, "normal": normal}


def train(
    npz_paths,
    out_dir: Path,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 2e-3,
    horizon_in: int = 24,
    horizon_out: int = 24,
    val_frac: float = 0.2,
    seed: int = 7,
):
    """Train on one or more .npz scenarios. If multiple, the per-scenario
    sliding-window datasets are ConcatDataset'd so the model sees every
    documented stress level (ev_growth_pct ∈ {0, 20, 35, 50}, etc.) and
    doesn't have to extrapolate at inference time."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"  # works everywhere; small model

    fg = build_graph()
    edge_index, _ = edge_index_tensor(fg)
    edge_index = edge_index.to(device)

    if isinstance(npz_paths, (str, Path)):
        npz_paths = [Path(npz_paths)]
    npz_paths = [Path(p) for p in npz_paths]

    spec = WindowSpec(horizon_in=horizon_in, horizon_out=horizon_out, stride=1)
    per_ds = [FeederWindowDataset(p, spec) for p in npz_paths]

    # Per-dataset 80/20 split so each scenario is evenly represented in val
    train_subsets, val_subsets = [], []
    for ds in per_ds:
        n_total = len(ds)
        n_val = int(val_frac * n_total)
        n_train = n_total - n_val
        train_subsets.append(torch.utils.data.Subset(ds, list(range(n_train))))
        val_subsets.append(torch.utils.data.Subset(ds, list(range(n_train, n_total))))
    train_subset = torch.utils.data.ConcatDataset(train_subsets)
    val_subset = torch.utils.data.ConcatDataset(val_subsets)
    full_ds = per_ds[0]  # use first dataset's bus_order/scaler for the checkpoint
    print(f"Training on {len(per_ds)} scenario(s): {[p.name for p in npz_paths]}")

    def _collate(batch):
        X = torch.stack([b[0] for b in batch], dim=0)
        y = torch.stack([b[1] for b in batch], dim=0)
        hw = torch.from_numpy(np.stack([b[2]["in_heatwave"] for b in batch], axis=0))
        return X, y, {"in_heatwave": hw}

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    cfg = ModelConfig(
        n_nodes=len(fg.g.nodes()),
        in_features=7,
        sage_hidden=32,
        sage_layers=2,
        gru_hidden=64,
        horizon_in=horizon_in,
        horizon_out=horizon_out,
        dropout=0.1,
    )
    model = GraphSAGEGRU(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    print(f"Train windows: {len(train_subset)}  Val windows: {len(val_subset)}")
    print(f"Trainable params: {model.num_parameters():,}")

    history = []
    best_val = float("inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "graphsage_gru.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for X, y, _ in train_loader:
            X = X.to(device); y = y.to(device)
            opt.zero_grad()
            yhat = model(X, edge_index)
            loss = loss_fn(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= max(1, len(train_subset))

        val_metrics = evaluate(model, val_loader, edge_index, device)
        val_rmse = val_metrics["overall"]["rmse"]
        dt = time.time() - t0
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_rmse={val_rmse:.3f}  val_mape={val_metrics['overall']['mape']:.2f}%  "
            f"hw_rmse={val_metrics['heatwave']['rmse']:.3f}  ({dt:.1f}s)"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}_{m}": v for k, d in val_metrics.items() for m, v in d.items()}})

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "state_dict": model.state_dict(),
                "model_config": cfg.__dict__,
                "scaler": full_ds.scaler,
                "bus_order": full_ds.bus_order,
            }, ckpt_path)

    final_metrics = evaluate(model, val_loader, edge_index, device)
    report = {
        "best_val_rmse": best_val,
        "final_metrics": final_metrics,
        "trainable_params": model.num_parameters(),
        "epochs": epochs,
        "horizon_in": horizon_in,
        "horizon_out": horizon_out,
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved report     -> {out_dir / 'training_report.json'}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data", type=Path, nargs="+",
        default=[
            REPO / "data" / "synthetic" / "baseline.npz",
            REPO / "data" / "synthetic" / "mild_stress_ev20_pv4.npz",
            REPO / "data" / "synthetic" / "stress_ev35_pv8.npz",
            REPO / "data" / "synthetic" / "severe_stress_ev50_pv12.npz",
        ],
        help="One or more .npz scenarios. Default trains on all four documented scenarios.",
    )
    p.add_argument("--out", type=Path, default=REPO / "models" / "checkpoints")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    train(args.data, args.out, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

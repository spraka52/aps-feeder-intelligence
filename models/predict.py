"""Load a trained checkpoint and produce 24-hour forecasts on demand."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from data.topology import build_graph, edge_index_tensor
from models.dataset import FeederWindowDataset, WindowSpec
from models.graphsage_gru import GraphSAGEGRU, ModelConfig

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Forecaster:
    model: GraphSAGEGRU
    edge_index: torch.Tensor
    scaler: dict
    bus_order: List[str]
    horizon_in: int
    horizon_out: int

    @classmethod
    def load(cls, ckpt_path: Path) -> "Forecaster":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["model_config"])
        model = GraphSAGEGRU(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        fg = build_graph()
        edge_index, _ = edge_index_tensor(fg)
        return cls(
            model=model,
            edge_index=edge_index,
            scaler=ckpt["scaler"],
            bus_order=ckpt["bus_order"],
            horizon_in=cfg.horizon_in,
            horizon_out=cfg.horizon_out,
        )

    def forecast_window(self, ds: FeederWindowDataset, t0: int) -> np.ndarray:
        """Run a forecast where input window starts at index t0.

        Returns array shape [horizon_out, N] of predicted load_kw.
        """
        N = len(self.bus_order)
        Xs = np.stack([ds._features_at(t0 + k, N) for k in range(self.horizon_in)], axis=0)
        X = torch.from_numpy(Xs[None, ...])  # [1, T, N, F]
        with torch.no_grad():
            yhat = self.model(X, self.edge_index).numpy()[0]  # [T_out, N]
        return yhat

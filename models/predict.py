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

        Returns array shape [horizon_out, N] of predicted load_kw. Output is
        clipped to ≥ 0 since negative load is non-physical (the model occasionally
        outputs tiny negatives — within numerical noise — for buses near zero).
        """
        N = len(self.bus_order)
        Xs = np.stack([ds._features_at(t0 + k, N) for k in range(self.horizon_in)], axis=0)
        X = torch.from_numpy(Xs[None, ...])  # [1, T, N, F]
        with torch.no_grad():
            yhat = self.model(X, self.edge_index).numpy()[0]  # [T_out, N]
        return np.clip(yhat, 0.0, None)

    def forecast_window_with_uncertainty(
        self, ds: FeederWindowDataset, t0: int, n_samples: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast + uncertainty band via Monte-Carlo dropout.

        Standard inference disables dropout (model.eval()) and emits a single
        point forecast. For planning-grade decisions a planner needs to know
        *how confident* the model is — a 245 kW peak forecast is useless if
        the 95% interval is [180, 320].

        Method: switch dropout layers ON at inference, run K = `n_samples`
        forward passes, and treat the empirical mean / std across samples
        as the predictive distribution (Gal & Ghahramani 2016, "Dropout as
        a Bayesian Approximation").

        Returns
        -------
        mean : ndarray  [horizon_out, N]
            Per-bus per-hour mean forecast (kW).
        p10  : ndarray  [horizon_out, N]
            10th percentile lower band.
        p90  : ndarray  [horizon_out, N]
            90th percentile upper band.
        """
        N = len(self.bus_order)
        Xs = np.stack([ds._features_at(t0 + k, N) for k in range(self.horizon_in)], axis=0)
        X = torch.from_numpy(Xs[None, ...])

        # Force dropout layers on while keeping batchnorm in eval mode.
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
        try:
            samples = []
            with torch.no_grad():
                for _ in range(int(n_samples)):
                    yhat = self.model(X, self.edge_index).numpy()[0]
                    samples.append(np.clip(yhat, 0.0, None))
        finally:
            self.model.eval()

        arr = np.stack(samples, axis=0)  # [K, T, N]
        mean = arr.mean(axis=0)
        p10 = np.percentile(arr, 10, axis=0)
        p90 = np.percentile(arr, 90, axis=0)
        return mean, p10, p90

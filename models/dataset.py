"""Sliding-window dataset for the GraphSAGE+GRU forecaster."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.synthesize import load_dataset
from data.topology import SPOT_LOADS_KW


@dataclass
class WindowSpec:
    horizon_in: int = 24
    horizon_out: int = 24
    stride: int = 1


def _hour_features(times: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    h = times.hour.values
    return np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0)


class FeederWindowDataset(Dataset):
    """Builds (X_seq, y_seq, meta) windows.

    X_seq: [T_in, N, F]
    y_seq: [T_out, N]   future load_kw
    meta:  dict with timestamps + heatwave mask for the target window
    """

    def __init__(
        self,
        npz_path: Path,
        spec: WindowSpec,
        bus_index: Optional[dict[str, int]] = None,
        scaler: Optional[dict] = None,
    ):
        self.payload = load_dataset(npz_path)
        self.spec = spec

        bus_list = self.payload["bus_list"]
        if bus_index is None:
            bus_index = {b: i for i, b in enumerate(sorted(bus_list))}
        self.bus_index = bus_index
        # Use the same order the graph topology uses so edge_index aligns.
        from data.topology import build_graph
        fg = build_graph()
        self.bus_order = sorted(fg.g.nodes())

        # Reorder loads to canonical bus order
        bus_to_src_idx = {b: i for i, b in enumerate(bus_list)}
        order = [bus_to_src_idx[b] for b in self.bus_order]
        self.loads = self.payload["loads_kw"][order]  # [N, T]
        self.temp = self.payload["temp_c"]
        self.ghi = self.payload["ghi"]
        self.ev_pct = self.payload["ev_growth_pct"]
        self.times = pd.DatetimeIndex(self.payload["time"])
        self.heatwave = self.payload["in_heatwave"]
        self.bus_baseline = np.array([SPOT_LOADS_KW.get(b, 0.0) for b in self.bus_order], dtype=np.float32)

        h_sin, h_cos = _hour_features(self.times)
        self.h_sin = h_sin.astype(np.float32)
        self.h_cos = h_cos.astype(np.float32)

        if scaler is None:
            scaler = self._fit_scaler()
        self.scaler = scaler

    def _fit_scaler(self) -> dict:
        return {
            "load_mean": float(self.loads.mean()),
            "load_std": float(self.loads.std() + 1e-6),
            "temp_mean": float(self.temp.mean()),
            "temp_std": float(self.temp.std() + 1e-6),
            "ghi_max": float(max(self.ghi.max(), 1.0)),
            "bus_baseline_max": float(max(self.bus_baseline.max(), 1.0)),
        }

    def _features_at(self, t: int, N: int) -> np.ndarray:
        """Feature matrix [N, F] at time t."""
        load_norm = (self.loads[:, t] - self.scaler["load_mean"]) / self.scaler["load_std"]
        temp_norm = np.full(N, (self.temp[t] - self.scaler["temp_mean"]) / self.scaler["temp_std"], dtype=np.float32)
        ghi_norm = np.full(N, self.ghi[t] / self.scaler["ghi_max"], dtype=np.float32)
        h_sin = np.full(N, self.h_sin[t], dtype=np.float32)
        h_cos = np.full(N, self.h_cos[t], dtype=np.float32)
        ev = np.full(N, self.ev_pct[t] / 100.0, dtype=np.float32)
        baseline = (self.bus_baseline / self.scaler["bus_baseline_max"]).astype(np.float32)
        return np.stack([load_norm, temp_norm, ghi_norm, h_sin, h_cos, ev, baseline], axis=1).astype(np.float32)

    def __len__(self) -> int:
        T = self.loads.shape[1]
        n = T - self.spec.horizon_in - self.spec.horizon_out + 1
        return max(0, n // self.spec.stride)

    def __getitem__(self, idx: int):
        t0 = idx * self.spec.stride
        N = self.loads.shape[0]
        Xs = np.stack([self._features_at(t0 + k, N) for k in range(self.spec.horizon_in)], axis=0)
        ys = self.loads[:, t0 + self.spec.horizon_in : t0 + self.spec.horizon_in + self.spec.horizon_out].T
        meta_times = self.times[t0 + self.spec.horizon_in : t0 + self.spec.horizon_in + self.spec.horizon_out]
        meta_hw = self.heatwave[t0 + self.spec.horizon_in : t0 + self.spec.horizon_in + self.spec.horizon_out]
        return (
            torch.from_numpy(Xs),
            torch.from_numpy(ys.astype(np.float32)),
            {
                "t_target_start": int(t0 + self.spec.horizon_in),
                "in_heatwave": meta_hw,
                "times": meta_times,
            },
        )

    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return y  # targets are kept in raw kW

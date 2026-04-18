"""GraphSAGE + GRU spatio-temporal forecaster (PyTorch Geometric).

Architecture:
    For each input timestep t (t = 1..H_in):
        node features x_t ∈ R^{N x F_in}  (load_kw, temp_c, ghi, hour_sin, hour_cos, ev_pct, bus_kw_baseline)
        h_t = GraphSAGE(x_t, edge_index)       # spatial mixing across the feeder
        h_t = ReLU + Dropout
    Then per-bus temporal GRU over the H_in encoded states ->
        next-step embedding -> linear head -> H_out future load_kw values per bus.

This is a real trainable model — learned weights live in:
    - SAGEConv layers (W_self, W_neigh per layer)
    - GRU input/hidden weights
    - Linear output head
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


@dataclass
class ModelConfig:
    n_nodes: int
    in_features: int = 7
    sage_hidden: int = 32
    sage_layers: int = 2
    gru_hidden: int = 64
    horizon_in: int = 24
    horizon_out: int = 24
    dropout: float = 0.1


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for i in range(layers):
            self.convs.append(SAGEConv(in_dim if i == 0 else hidden, hidden, aggr="mean"))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GraphSAGEGRU(nn.Module):
    """Spatio-temporal forecaster.

    Inputs:
        x_seq: [B, T_in, N, F_in]
        edge_index: [2, E] (shared across the batch)
    Output:
        y_hat: [B, T_out, N]   (predicted load_kw per bus per future hour)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GraphSAGEEncoder(cfg.in_features, cfg.sage_hidden, cfg.sage_layers, cfg.dropout)
        self.gru = nn.GRU(
            input_size=cfg.sage_hidden,
            hidden_size=cfg.gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.gru_hidden),
            nn.ReLU(),
            nn.Linear(cfg.gru_hidden, cfg.horizon_out),
        )

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, T, N, F_in = x_seq.shape
        # Encode each timestep with the shared graph
        encoded = []
        for t in range(T):
            x_t = x_seq[:, t]                        # [B, N, F]
            x_flat = x_t.reshape(B * N, F_in)        # [B*N, F]
            # Replicate edge_index across the batch via offset trick
            ei = self._batched_edge_index(edge_index, B, N, x_seq.device)
            h = self.encoder(x_flat, ei)             # [B*N, H]
            encoded.append(h.view(B, N, -1))
        h_seq = torch.stack(encoded, dim=1)          # [B, T, N, H]

        # Per-node GRU
        h_seq = h_seq.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # [B*N, T, H]
        _, h_last = self.gru(h_seq)                  # h_last: [1, B*N, Hg]
        h_last = h_last.squeeze(0)                   # [B*N, Hg]
        out = self.head(h_last)                      # [B*N, T_out]
        out = out.view(B, N, -1).permute(0, 2, 1)    # [B, T_out, N]
        return out

    @staticmethod
    def _batched_edge_index(edge_index: torch.Tensor, B: int, N: int, device) -> torch.Tensor:
        """Tile edge_index across B independent graphs by adding node offsets."""
        if B == 1:
            return edge_index.to(device)
        ei = edge_index.to(device)
        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)  # [B,1,1]
        ei_b = ei.unsqueeze(0).expand(B, -1, -1) + offsets            # [B,2,E]
        return ei_b.permute(1, 0, 2).reshape(2, -1)                   # [2, B*E]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

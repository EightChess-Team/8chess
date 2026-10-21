"""Torch model for the 8Chess NNUE with quantization-aware forward.

Mirrors the engine's structure exactly (`src/nnue.rs`, `forward/scalar.rs`):

    acc[pov]  = ft_bias + sum(ft_piece_rows) + sum(ft_threat_rows)   (pov)
    ft_out    = concat(pair(acc[stm]), pair(acc[opp]))               (L1_SIZE)
    l1        = clamp(ft_out @ l1_weights[bucket] + l1_biases[bucket], 0, 1)
    l2, l3    = affine + clamp(0, 1) / affine
    score_cp  = l3_out * 380

`pair(v)` squares `clamp(v[:L1_SIZE//2], 0, 1)` with `clamp(v[L1_SIZE//2:], 0, 1)`.
Quantization-aware scales: 255 for the FT (i16 piece / i8 threat rows),
64 for L1 (i8 rows), using straight-through estimators so the exported
integer weights match training closely.

Feature indices are pre-mirrored to each accumulator's POV by the encoder
(`encoder.py`); both accumulators share the same embedding tables, exactly
like the engine. The output is from the side-to-move perspective (negamax
convention used by `evalnum`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    FT_QUANT,
    L1_QUANT,
    L1_SIZE,
    L2_SIZE,
    L3_SIZE,
    NETWORK_SCALE,
    OUTPUT_BUCKETS,
    PIECE_FEATURES,
    THREAT_FEATURES,
)

FT_PIECE_MAX = 32767
FT_THREAT_MAX = 127
L1_MAX = 127

_HALF = L1_SIZE // 2


def _ste_quantize(w: torch.Tensor, scale: int, max_int: int) -> torch.Tensor:
    q = torch.clamp(torch.round(w * scale), -max_int, max_int) / scale
    return w + (q - w).detach()


def _pair_activation(acc: torch.Tensor) -> torch.Tensor:
    act = acc.clamp(0.0, 1.0)
    left = act[:, :_HALF]
    right = act[:, _HALF:]
    return torch.floor(left * right * 512.0) / 512.0


class EightChessNNUE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ft_piece = nn.Embedding(PIECE_FEATURES, L1_SIZE)
        self.ft_threat = nn.Embedding(THREAT_FEATURES, L1_SIZE)
        self.ft_bias = nn.Parameter(torch.zeros(L1_SIZE))
        self.l1_weight = nn.Parameter(torch.empty(OUTPUT_BUCKETS, L2_SIZE, L1_SIZE))
        self.l1_bias = nn.Parameter(torch.zeros(OUTPUT_BUCKETS, L2_SIZE))
        self.l2_weight = nn.Parameter(torch.empty(OUTPUT_BUCKETS, L2_SIZE, L3_SIZE))
        self.l2_bias = nn.Parameter(torch.zeros(OUTPUT_BUCKETS, L3_SIZE))
        self.l3_weight = nn.Parameter(torch.empty(OUTPUT_BUCKETS, L3_SIZE))
        self.l3_bias = nn.Parameter(torch.zeros(OUTPUT_BUCKETS))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.ft_piece.weight, -0.2, 0.2)
        nn.init.uniform_(self.ft_threat.weight, -0.05, 0.05)
        nn.init.uniform_(self.l1_weight, -0.1, 0.1)
        nn.init.uniform_(self.l2_weight, -0.1, 0.1)
        nn.init.uniform_(self.l3_weight, -0.1, 0.1)

    @classmethod
    def from_network(cls, net: dict[str, np.ndarray]) -> EightChessNNUE:
        """Initializes a PyTorch model from a deserialized engine network dict."""
        model = cls()
        with torch.no_grad():
            model.ft_piece.weight.copy_(torch.from_numpy(net["ft_piece_weights"].astype(np.float32) / FT_QUANT))
            model.ft_threat.weight.copy_(torch.from_numpy(net["ft_threat_weights"].astype(np.float32) / FT_QUANT))
            model.ft_bias.copy_(torch.from_numpy(net["ft_biases"].astype(np.float32) / FT_QUANT))

            l1_raw = net["l1_weights"].astype(np.float32) / L1_QUANT
            l1_model = (
                l1_raw.reshape(OUTPUT_BUCKETS, L1_SIZE // 4, L2_SIZE, 4)
                .transpose(0, 2, 1, 3)
                .reshape(OUTPUT_BUCKETS, L2_SIZE, L1_SIZE)
            )
            model.l1_weight.copy_(torch.from_numpy(l1_model))
            model.l1_bias.copy_(torch.from_numpy(net["l1_biases"]))

            model.l2_weight.copy_(torch.from_numpy(net["l2_weights"]))
            model.l2_bias.copy_(torch.from_numpy(net["l2_biases"]))
            model.l3_weight.copy_(torch.from_numpy(net["l3_weights"]))
            model.l3_bias.copy_(torch.from_numpy(net["l3_biases"]))
        return model

    @classmethod
    def load_network_file(cls, path: str) -> EightChessNNUE:
        from . import format as nnue_format
        return cls.from_network(nnue_format.read_network(path))

    def _accumulator(self, piece_idx, piece_off, threat_idx, threat_off, fw, tw, bias):
        acc = F.embedding_bag(piece_idx, fw, piece_off, mode="sum")
        acc = acc + F.embedding_bag(threat_idx, tw, threat_off, mode="sum")
        return acc + bias

    def forward(
        self,
        piece_stm: torch.Tensor,
        piece_off_stm: torch.Tensor,
        threat_stm: torch.Tensor,
        threat_off_stm: torch.Tensor,
        piece_opp: torch.Tensor,
        piece_off_opp: torch.Tensor,
        threat_opp: torch.Tensor,
        threat_off_opp: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        fw = _ste_quantize(self.ft_piece.weight, FT_QUANT, FT_PIECE_MAX)
        tw = _ste_quantize(self.ft_threat.weight, FT_QUANT, FT_THREAT_MAX)
        bias = _ste_quantize(self.ft_bias, FT_QUANT, FT_PIECE_MAX)

        acc_stm = self._accumulator(piece_stm, piece_off_stm, threat_stm, threat_off_stm, fw, tw, bias)
        acc_opp = self._accumulator(piece_opp, piece_off_opp, threat_opp, threat_off_opp, fw, tw, bias)

        ft = torch.cat([_pair_activation(acc_stm), _pair_activation(acc_opp)], dim=1)

        l1w = _ste_quantize(self.l1_weight, L1_QUANT, L1_MAX)[buckets]
        l1 = torch.bmm(l1w, ft.unsqueeze(-1)).squeeze(-1)
        l1 = (l1 + self.l1_bias[buckets]).clamp(0.0, 1.0)

        l2w = self.l2_weight[buckets].transpose(1, 2)
        l2 = torch.bmm(l2w, l1.unsqueeze(-1)).squeeze(-1)
        l2 = (l2 + self.l2_bias[buckets]).clamp(0.0, 1.0)

        out = (self.l3_weight[buckets] * l2).sum(dim=1) + self.l3_bias[buckets]
        return out

    @torch.no_grad()
    def forward_cp(self, *args) -> torch.Tensor:
        return self.forward(*args) * NETWORK_SCALE


def wdl_from_cp(score_cp: torch.Tensor, temperature: float = 173.72) -> torch.Tensor:
    """Expected score from centipawns via a logistic link (Elo == cp)."""
    return torch.sigmoid(score_cp / temperature)

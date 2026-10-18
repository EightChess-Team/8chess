"""Bit-accurate replica of the engine's SIMD inference for AVX2 (and the
scalar path, which is semantically equal for the activation step).

Semantics captured from src/nnue/forward/vectorized.rs + src/nnue/simd/avx2.rs:
  - accumulator sums are wrapping i16 arithmetic (mod 2^16)
  - activation  : s = wrap16(pst + threat); lhs = clamp(s,0,255);
                  rhs = min(s,255); byte = sat_u8(floor(lhs*rhs / 512))
  - L1 matmul   : maddubs pairwise sums saturate to i16, chunk-pair sums wrap,
                  then widen to i32 (madd) and accumulate
  - float head  : f32 fma in numpy (non-fused mul+add; <=1cp deviation possible)
"""

from __future__ import annotations

import chess
import numpy as np

from .constants import (
    DEQUANT_MULTIPLIER,
    FT_QUANT,
    FT_SHIFT,
    L1_SIZE,
    L2_SIZE,
    L3_SIZE,
    NETWORK_SCALE,
    output_bucket,
)
from .features import piece_feature_indices, threat_feature_indices

_CHUNKS = L1_SIZE // 4  # 192 four-byte chunks


def _wrap16(x: np.ndarray) -> np.ndarray:
    return (x & 0xFFFF) - ((x & 0x8000) << 1)


def _sat16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -32768, 32767)


def _accumulate(weights: np.ndarray, indices: list[int], base: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """wrapping i16 accumulator: base + sum(weights[idx]) mod 2^16."""
    if not indices:
        return base
    idx = np.asarray(indices, dtype=np.int64)
    summed = weights[idx].astype(np.int64).sum(axis=0)
    return _wrap16(base.astype(np.int64) + summed).astype(dtype)


def evaluate(net: dict[str, np.ndarray], board: chess.Board) -> int:
    stm = 0 if board.turn == chess.WHITE else 1

    pst_feats = piece_feature_indices(board)
    thr_feats = threat_feature_indices(board)

    ft_piece = net["ft_piece_weights"]
    ft_threat = net["ft_threat_weights"]
    ft_biases = net["ft_biases"]

    acc_pst = [None, None]
    acc_thr = [None, None]
    for pov in (0, 1):
        acc_pst[pov] = _accumulate(ft_piece, pst_feats[pov], ft_biases, np.int32)
        acc_thr[pov] = _accumulate(ft_threat, thr_feats[pov], np.zeros(L1_SIZE, dtype=np.int64), np.int32)

    # Activation (per flip ordering: flip 0 uses stm pov, flip 1 the opponent).
    ft_out = np.zeros(L1_SIZE, dtype=np.int64)
    for flip in (0, 1):
        pov = stm ^ flip
        s = _wrap16(acc_pst[pov].astype(np.int64) + acc_thr[pov].astype(np.int64))
        lhs = np.clip(s[0 : L1_SIZE // 2], 0, FT_QUANT)
        rhs = np.minimum(s[L1_SIZE // 2 :], FT_QUANT)
        prod = (lhs * rhs) >> FT_SHIFT  # floor division for arithmetic shift
        ft_out[flip * (L1_SIZE // 2) : (flip + 1) * (L1_SIZE // 2)] = np.clip(prod, 0, 255)

    ft_bytes = ft_out.astype(np.uint8)
    bucket = output_bucket(int(board.occupied.bit_count()))

    # L1: maddubs/dpbusd semantics over 4-byte chunks. The engine pairs
    # *nonzero* chunks in nnz order (ascending), with the odd tail handled
    # by a single dpbusd.
    l1_w = net["l1_weights"][bucket].reshape(_CHUNKS, L2_SIZE, 4).astype(np.int64)
    bytes_c = ft_bytes.reshape(_CHUNKS, 4)

    nnz = np.nonzero(bytes_c.any(axis=1))[0]
    pre = np.zeros(L2_SIZE, dtype=np.int64)
    if nnz.size:
        prod = bytes_c[nnz].astype(np.int64)[:, None, :] * l1_w[nnz]  # (k, L2, 4)
        pairwise = _sat16(prod[:, :, 0::2] + prod[:, :, 1::2])  # (k, L2, 2)

        k = nnz.size
        pairs = k - (k % 2)
        if pairs:
            q = _wrap16(pairwise[0:pairs:2] + pairwise[1:pairs:2])  # (pairs/2, L2, 2)
            pre += q.sum(axis=(0, 2))
        if k % 2:
            pre += pairwise[k - 1].sum(axis=1)

    l1_b = net["l1_biases"][bucket].astype(np.float32)
    l1_out = np.clip(pre.astype(np.float32) * np.float32(DEQUANT_MULTIPLIER) + l1_b, 0.0, 1.0)

    l2_w = net["l2_weights"][bucket].astype(np.float32)
    l2_b = net["l2_biases"][bucket].astype(np.float32)
    l2_out = l2_b.copy()
    for i in range(L2_SIZE):
        l2_out = l2_w[i] * l1_out[i] + l2_out
    l2_out = np.clip(l2_out, 0.0, 1.0)

    l3_w = net["l3_weights"][bucket].astype(np.float32)
    l3_b = np.float32(net["l3_biases"][bucket])

    # AVX2 horizontal_sum: LANES=2 vectors of 8; sum vectors, then tree reduce.
    v = l3_w[0:8] * l2_out[0:8] + l3_w[16:24] * l2_out[16:24]
    w = l3_w[8:16] * l2_out[8:16] + l3_w[24:32] * l2_out[24:32]
    vec = v + w
    lo4 = vec[0:4] + vec[4:8]
    two = lo4[0:2] + lo4[2:4]
    total = two[0] + two[1]
    l3_out = np.float32(total + l3_b)
    return int(np.float32(l3_out * np.float32(NETWORK_SCALE)))


def evaluate_fen(net: dict[str, np.ndarray], fen: str) -> int:
    return evaluate(net, chess.Board(fen))

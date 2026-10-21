"""Position encoding for training: features -> ragged index streams.

Produces the four index streams consumed by `model.EightChessNNUE`:

    piece_stm / threat_stm : features of the side-to-move accumulator
    piece_opp / threat_opp : features of the opponent accumulator

plus the output bucket and the result in side-to-move perspective.
Encoding is CPU-bound python-chess work; `encode_fens` parallelizes it.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from functools import partial

import chess
import numpy as np

from .constants import output_bucket
from .features import piece_feature_indices, threat_feature_indices


def encode_board(board: chess.Board) -> tuple[list[int], list[int], list[int], list[int], int, int]:
    """Returns (pieces_stm, threats_stm, pieces_opp, threats_opp, bucket, stm)."""
    stm = 0 if board.turn == chess.WHITE else 1
    pfeats = piece_feature_indices(board)
    tfeats = threat_feature_indices(board)
    bucket = output_bucket(int(board.occupied.bit_count()))
    return (
        pfeats[stm],
        tfeats[stm],
        pfeats[stm ^ 1],
        tfeats[stm ^ 1],
        bucket,
        stm,
    )


def _encode_fen_worker(args: tuple[int, str, str]) -> tuple[int, list[list[int]], int, float]:
    index, fen, result = args
    board = chess.Board(fen)
    ps, ts, po, to, bucket, stm = encode_board(board)
    target = 0.5 if result == "1/2-1/2" else (1.0 if (result == "1-0") == (stm == 0) else 0.0)
    return index, [ps, ts, po, to], bucket, target


def encode_fens(
    fens: list[str],
    results: list[str],
    workers: int = 8,
    chunk: int = 256,
) -> dict[str, np.ndarray]:
    """Encodes positions into packed ragged arrays.

    Returns a dict with flat uint16 indices per stream, int32 offsets
    (cumulative starts, length n+1), uint8 buckets and float32 targets.
    """
    assert len(fens) == len(results)
    n = len(fens)
    streams: list[list[list[int]]] = [[], [], [], []]
    buckets = np.empty(n, dtype=np.uint8)
    targets = np.empty(n, dtype=np.float32)

    args = [(i, f, r) for i, (f, r) in enumerate(zip(fens, results))]
    with mp.Pool(workers) as pool:
        for index, feats, bucket, target in pool.imap_unordered(
            _encode_fen_worker, args, chunksize=chunk
        ):
            for s in range(4):
                streams[s].append(feats[s])
            buckets[index] = bucket
            targets[index] = target

    out: dict[str, np.ndarray] = {}
    names = ["piece_stm", "threat_stm", "piece_opp", "threat_opp"]
    for name, feats in zip(names, streams):
        offsets = np.zeros(n + 1, dtype=np.int64)
        for i, f in enumerate(feats):
            offsets[i + 1] = offsets[i] + len(f)
        flat = np.empty(int(offsets[-1]), dtype=np.int32)
        pos = 0
        for f in feats:
            flat[pos : pos + len(f)] = f
            pos += len(f)
        out[f"{name}_indices"] = flat
        out[f"{name}_offsets"] = offsets
    out["bucket"] = buckets
    out["target"] = targets
    return out


def _encode_fen_scored_worker(
    args: tuple[int, str, int, int],
) -> tuple[int, list[list[int]], int, float, int]:
    index, fen, score, result = args
    board = chess.Board(fen)
    ps, ts, po, to, bucket, stm = encode_board(board)
    result_wdl = 0.5 if result == 0 else (1.0 if result > 0 else 0.0)
    return index, [ps, ts, po, to], bucket, result_wdl, score


def encode_fens_scored(
    fens: list[str],
    scores: list[int],
    results: list[int],
    lam: float = 1.0,
    scaling: float = 410.0,
    workers: int = 8,
    chunk: int = 256,
) -> dict[str, np.ndarray]:
    """Encodes positions with Stockfish-style eval/result blending.

    `lambda=1.0` trains on the search evaluation, `lambda=0.0` trains on game
    results. Scores and results are both from the side-to-move perspective.
    """
    assert len(fens) == len(scores) == len(results)
    n = len(fens)
    streams: list[list[list[int]]] = [[], [], [], []]
    buckets = np.empty(n, dtype=np.uint8)
    targets = np.empty(n, dtype=np.float32)

    args = [(i, f, s, r) for i, (f, s, r) in enumerate(zip(fens, scores, results))]
    with mp.Pool(workers) as pool:
        for index, feats, bucket, result_wdl, score in pool.imap_unordered(
            _encode_fen_scored_worker, args, chunksize=chunk
        ):
            for s in range(4):
                streams[s].append(feats[s])
            buckets[index] = bucket
            clipped = max(-2000, min(2000, score))
            eval_wdl = 1.0 / (1.0 + math.exp(-clipped / scaling))
            targets[index] = lam * eval_wdl + (1.0 - lam) * result_wdl

    out: dict[str, np.ndarray] = {}
    names = ["piece_stm", "threat_stm", "piece_opp", "threat_opp"]
    for name, feats in zip(names, streams):
        offsets = np.zeros(n + 1, dtype=np.int64)
        for i, f in enumerate(feats):
            offsets[i + 1] = offsets[i] + len(f)
        flat = np.empty(int(offsets[-1]), dtype=np.int32)
        pos = 0
        for f in feats:
            flat[pos : pos + len(f)] = f
            pos += len(f)
        out[f"{name}_indices"] = flat
        out[f"{name}_offsets"] = offsets
    out["bucket"] = buckets
    out["target"] = targets
    return out


def save_encoded(data: dict[str, np.ndarray], path: str) -> None:
    np.savez_compressed(path, **data)


def load_encoded(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}

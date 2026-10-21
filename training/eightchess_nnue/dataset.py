"""Training sample extraction from game databases (PGN / FEN+result files).

Samples are (fen, result) pairs where result is `1-0`, `0-1` or
`1/2-1/2`. Self-play and engine-game PGNs (e.g. Lc0, engine-vs-engine) are
both supported; positions with both kings present are kept.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
from typing import Iterable, Iterator

import chess
import chess.pgn


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", errors="replace")
    if path.endswith(".xz") or path.endswith(".lzma"):
        return lzma.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def iter_pgn_positions(
    paths: Iterable[str],
    stride: int = 2,
    min_ply: int = 8,
    skip_plies_after_end: int = 0,
) -> Iterator[tuple[str, str]]:
    """Yields (fen, result) sampled every `stride` plies from each game."""
    for path in paths:
        with _open(path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                if result not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    if ply < min_ply:
                        continue
                    if (ply - min_ply) % stride:
                        continue
                    yield board.fen(), result


def iter_plain(path: str) -> Iterator[tuple[str, int, int]]:
    """Yields (fen, score, result) from Stockfish `.plain` training data.

    Both score and result are from the side-to-move perspective; result is
    -1/0/+1 for loss/draw/win.
    """
    fen: str | None = None
    score = 0
    result = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("fen "):
                fen = line[4:]
            elif line.startswith("score "):
                score = int(line[6:])
            elif line.startswith("result "):
                result = int(line[7:])
            elif line == "e" and fen is not None:
                yield fen, score, result
                fen = None
                score = 0
                result = 0


def iter_fen_results(path: str) -> Iterator[tuple[str, str]]:
    """Yields (fen, result) from a CSV file: `fen,result` per line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            yield parts[0], parts[1]


def sample_positions(
    pgns: Iterable[str] = (),
    fen_files: Iterable[str] = (),
    stride: int = 2,
    min_ply: int = 8,
    max_samples: int | None = None,
) -> tuple[list[str], list[str]]:
    fens: list[str] = []
    results: list[str] = []
    for fen, result in iter_pgn_positions(pgns, stride=stride, min_ply=min_ply):
        fens.append(fen)
        results.append(result)
        if max_samples and len(fens) >= max_samples:
            return fens, results
    for fen_file in fen_files:
        for fen, result in iter_fen_results(fen_file):
            fens.append(fen)
            results.append(result)
            if max_samples and len(fens) >= max_samples:
                return fens, results
    return fens, results


def sample_plain_positions(
    plain_files: Iterable[str],
    stride: int = 1,
    max_samples: int | None = None,
) -> tuple[list[str], list[int], list[int]]:
    fens: list[str] = []
    scores: list[int] = []
    results: list[int] = []
    skip = max(stride, 1)

    for plain_file in plain_files:
        for fen, score, result in iter_plain(plain_file):
            if skip > 1:
                skip -= 1
                continue
            skip = max(stride, 1)
            fens.append(fen)
            scores.append(score)
            results.append(result)
            if max_samples and len(fens) >= max_samples:
                return fens, scores, results

    return fens, scores, results

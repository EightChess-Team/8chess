"""Verify the Python quantized replica against the real engine (`evalnum`).

Usage:
    python -m eightchess_nnue.verify_engine --engine ../../target/release/eightchess \
        --network ../../networks/v60-7f587dfb.nnue --fens fens.txt

Reports exact-match rate and deviation histogram. The feature extraction and
quantized forward are only considered validated when the exact-match rate is
essentially 100% (float FMA differences can cause rare +/-1 cp ties).
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys

import chess
import numpy as np

from . import format as nnue_format
from .quantized import evaluate


def random_fens(count: int, seed: int = 42, max_plies: int = 120) -> list[str]:
    rng = random.Random(seed)
    fens = []
    while len(fens) < count:
        board = chess.Board()
        plies = rng.randint(4, max_plies)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=False):
                break
            board.push(moves[rng.randrange(len(moves))])
        if not board.is_game_over(claim_draw=False):
            fens.append(board.fen())
    return fens


def engine_evals(engine: str, fens: list[str], eval_file: str | None = None) -> list[int]:
    cmd = [engine]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def send(line: str):
        assert proc.stdin is not None
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    if eval_file:
        send(f"setoption name EvalFile value {eval_file}")

    results = []
    assert proc.stdout is not None
    for fen in fens:
        send(f"position fen {fen}")
        send("evalnum")
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("engine closed stdout")
            line = line.strip()
            if line.startswith("info string") or not line:
                continue
            try:
                results.append(int(line))
                break
            except ValueError:
                continue
    send("quit")
    proc.wait(timeout=30)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--fens", default=None, help="file with one FEN per line (else random)")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.fens:
        with open(args.fens) as f:
            fens = [l.strip() for l in f if l.strip()]
    else:
        fens = random_fens(args.count, args.seed)

    print(f"verifying {len(fens)} positions ...")
    net = nnue_format.read_network(args.network)
    expected = engine_evals(args.engine, fens, eval_file=None)

    exact = 0
    diffs = np.zeros(len(fens), dtype=np.int64)
    for i, (fen, eng) in enumerate(zip(fens, expected)):
        mine = evaluate(net, chess.Board(fen))
        diffs[i] = mine - eng
        if mine == eng:
            exact += 1
        elif i < 20:
            print(f"  mismatch: {fen} engine={eng} python={mine}")

    total = len(fens)
    print(f"exact match: {exact}/{total} ({100.0 * exact / total:.2f}%)")
    print(f"max |diff|: {int(np.abs(diffs).max())}  mean |diff|: {np.abs(diffs).mean():.4f}")
    hist = {int(v): int((diffs == v).sum()) for v in np.unique(diffs)}
    print(f"diff histogram: {hist}")

    return 0 if exact >= total * 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())

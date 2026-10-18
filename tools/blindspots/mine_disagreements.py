"""Mine tactical and positional blindspots: Stockfish deep search vs NNUE evaluation.

Finds positions where deep tactical search (e.g. Stockfish depth 30..50) sharply
disagrees with fast static neural evaluation (NNUE / Lc0). Positions with large
evaluation deltas represent blindspots (e.g. tactical traps, quiet horizon blunders,
or fortress mis-evaluations) and serve as a high-value curriculum for fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure training package is importable
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_root, "training"))

import chess
import chess.engine
import numpy as np

from eightchess_nnue import format as nnf
from eightchess_nnue.quantized import evaluate


def evaluate_stockfish_deep(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    time_limit: float | None = None,
) -> tuple[int, chess.Move | None]:
    """Returns (eval_cp from STM perspective, best_move)."""
    limit = chess.engine.Limit(depth=depth, time=time_limit)
    info = engine.analyse(board, limit)
    score = info.get("score")
    best_move = info.get("pv", [None])[0]

    if score is None:
        return 0, best_move

    # Convert to centipawns from STM POV
    cp = score.white().score(mate_score=10000)
    if cp is None:
        cp = 0
    if board.turn == chess.BLACK:
        cp = -cp

    return cp, best_move


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens", required=True, help="file containing FENs to evaluate (1 per line)")
    parser.add_argument("--stockfish", default="/usr/games/stockfish", help="path to Stockfish binary")
    parser.add_argument("--network", default="networks/v60-7f587dfb.nnue", help="path to 8Chess .nnue")
    parser.add_argument("--depth", type=int, default=30, help="Stockfish search depth (e.g. 30..50)")
    parser.add_argument("--time-per-pos", type=float, default=2.0, help="max search time per position")
    parser.add_argument("--threshold", type=int, default=150, help="min |SF - NNUE| delta in cp to flag as blindspot")
    parser.add_argument("--out-csv", default="blindspots.csv", help="output CSV file for mined blindspots")
    parser.add_argument("--threads", type=int, default=2, help="Stockfish threads")
    args = parser.parse_args()

    print(f"Loading NNUE network: {args.network}")
    net = nnf.read_network(args.network)

    with open(args.fens) as f:
        fens = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    print(f"Loaded {len(fens)} positions to audit")

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"Threads": args.threads, "Hash": 128})

    blindspots = []
    print(f"Auditing with Stockfish depth {args.depth} (threshold: {args.threshold} cp)...\n")

    try:
        with open(args.out_csv, "w") as out:
            out.write("fen,sf_cp,nnue_cp,delta,sf_move\n")

            for i, fen in enumerate(fens):
                board = chess.Board(fen)
                nnue_cp = evaluate(net, board)

                sf_cp, sf_move = evaluate_stockfish_deep(
                    sf, board, depth=args.depth, time_limit=args.time_per_pos
                )

                delta = abs(sf_cp - nnue_cp)
                move_str = sf_move.uci() if sf_move else "none"

                is_blindspot = delta >= args.threshold
                status = "[BLINDSPOT]" if is_blindspot else "           "

                print(
                    f"{status} [{i+1:3d}/{len(fens)}] NNUE: {nnue_cp:+5d} cp | SF(d={args.depth}): {sf_cp:+5d} cp | Delta: {delta:4d} cp"
                )

                if is_blindspot:
                    blindspots.append((fen, sf_cp, nnue_cp, delta, move_str))
                    out.write(f'"{fen}",{sf_cp},{nnue_cp},{delta},{move_str}\n')
                    out.flush()

    finally:
        sf.quit()

    print(f"\nAudit complete: found {len(blindspots)} / {len(fens)} blindspots (saved to {args.out_csv})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

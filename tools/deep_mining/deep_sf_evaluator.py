"""High-depth Stockfish 19 evaluator for tactical truth mining.

Runs Stockfish 19 locally with all available CPU threads at deep search
(depth 30..50) to evaluate candidate positions and extract ground-truth
tactical evaluations and best moves.

Usage:
    python3 tools/deep_mining/deep_sf_evaluator.py \
        --fens data/raw_positions.csv --depth 35 --threads 10 \
        --out-csv data/sf_depth35_gold.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import chess.engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens", required=True, help="input CSV or text file containing FENs")
    parser.add_argument(
        "--stockfish",
        default="/home/hemesh/Videos/8chess/bin/stockfish/stockfish-linux-x86-64-universal",
    )
    parser.add_argument("--depth", type=int, default=35, help="search depth (e.g. 30..50)")
    parser.add_argument("--nodes", type=int, default=None, help="optional node limit per position")
    parser.add_argument("--time-limit", type=float, default=30.0, help="max seconds per position")
    parser.add_argument("--threads", type=int, default=10, help="Stockfish threads")
    parser.add_argument("--hash", type=int, default=1024, help="Stockfish hash size in MB")
    parser.add_argument("--out-csv", default="data/sf_deep_gold.csv")
    parser.add_argument("--limit-positions", type=int, default=None)
    args = parser.parse_args()

    fens = []
    with open(args.fens) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fen = line.split(",")[0].strip('"')
            fens.append(fen)

    if args.limit_positions:
        fens = fens[: args.limit_positions]

    print(f"Loaded {len(fens):,} positions to evaluate at depth {args.depth}")
    print(f"Stockfish 19 binary: {args.stockfish} ({args.threads} threads, {args.hash} MB Hash)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    # Resume support
    already_done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for line in f:
                p = line.strip().split(",")
                if p:
                    already_done.add(p[0].strip('"'))
        print(f"Resuming with {len(already_done):,} positions already evaluated in {args.out_csv}")

    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    sf.configure({"Threads": args.threads, "Hash": args.hash})

    evaluated = len(already_done)
    t0 = time.time()

    try:
        with open(args.out_csv, "a") as out:
            if evaluated == 0:
                out.write("fen,depth,score_cp,best_move,nodes,time_s\n")

            for i, fen in enumerate(fens):
                if fen in already_done:
                    continue

                board = chess.Board(fen)
                t_pos = time.time()
                limit = chess.engine.Limit(
                    depth=args.depth,
                    nodes=args.nodes,
                    time=args.time_limit,
                )
                info = sf.analyse(board, limit)
                dt = time.time() - t_pos

                score_obj = info.get("score")
                if score_obj is None:
                    continue

                # Centipawns from side-to-move perspective
                cp = score_obj.relative.score(mate_score=10000)
                if cp is None:
                    cp = 10000 if score_obj.relative.mate() > 0 else -10000

                actual_depth = info.get("depth", args.depth)
                nodes = info.get("nodes", 0)
                pv_list = info.get("pv", [])
                best_move = pv_list[0].uci() if pv_list else "none"

                out.write(f'"{fen}",{actual_depth},{cp},{best_move},{nodes},{dt:.2f}\n')
                out.flush()
                already_done.add(fen)
                evaluated += 1

                elapsed = time.time() - t0
                avg_time = elapsed / max(evaluated - len(already_done) + 1, 1)
                print(
                    f"[{evaluated:,}/{len(fens):,}] d={actual_depth:2d} | Score: {cp:+5d} cp | "
                    f"Move: {best_move:5s} | {nodes:8,d} nodes ({dt:.1f}s)",
                    flush=True,
                )

    finally:
        sf.quit()

    print(f"\nEvaluation complete! Evaluated {evaluated:,} positions in {time.time()-t0:.1f}s")
    print(f"Saved to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Parallel High-Depth Stockfish 19 Evaluator for Tactical Truth Mining.

Evaluates thousands of positions using concurrent Stockfish 19 instances,
recording exact depth, centipawn score, nodes, time, and best move.
Supports resume automatically.

Usage:
    python3 tools/deep_mining/deep_sf_evaluator.py \
        --fens data/critical_loss_fens.txt --depth 22 --concurrency 5 --threads-per-eng 2 \
        --out-csv data/sf_deep_gold.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess
import chess.engine


def eval_one_fen(
    engine_path: str,
    fen: str,
    depth: int,
    time_limit: float,
    threads: int,
    hash_mb: int,
) -> tuple[str, int, int, str, int, float] | None:
    board = chess.Board(fen)
    sf = chess.engine.SimpleEngine.popen_uci(engine_path)
    sf.configure({"Threads": threads, "Hash": hash_mb})

    try:
        t0 = time.time()
        limit = chess.engine.Limit(depth=depth, time=time_limit)
        info = sf.analyse(board, limit)
        dt = time.time() - t0

        score_obj = info.get("score")
        if score_obj is None:
            return None

        cp = score_obj.relative.score(mate_score=10000)
        if cp is None:
            cp = 10000 if score_obj.relative.mate() > 0 else -10000

        actual_depth = info.get("depth", depth)
        nodes = info.get("nodes", 0)
        pv = info.get("pv", [])
        best_move = pv[0].uci() if pv else "none"

        return fen, actual_depth, cp, best_move, nodes, dt

    finally:
        try:
            sf.quit()
        except Exception:
            pass


def worker_loop(
    engine_path: str,
    fens_queue: list[str],
    depth: int,
    time_limit: float,
    threads: int,
    hash_mb: int,
    out_csv: str,
    lock: threading.Lock,
    progress: list[int],
    total: int,
    t_start: float,
):
    sf = chess.engine.SimpleEngine.popen_uci(engine_path)
    sf.configure({"Threads": threads, "Hash": hash_mb})

    try:
        for fen in fens_queue:
            board = chess.Board(fen)
            t0 = time.time()
            limit = chess.engine.Limit(depth=depth, time=time_limit)
            info = sf.analyse(board, limit)
            dt = time.time() - t0

            score_obj = info.get("score")
            if score_obj is None:
                continue

            cp = score_obj.relative.score(mate_score=10000)
            if cp is None:
                cp = 10000 if score_obj.relative.mate() > 0 else -10000

            actual_depth = info.get("depth", depth)
            nodes = info.get("nodes", 0)
            pv = info.get("pv", [])
            best_move = pv[0].uci() if pv else "none"

            line = f'"{fen}",{actual_depth},{cp},{best_move},{nodes},{dt:.2f}\n'
            with lock:
                with open(out_csv, "a") as f:
                    f.write(line)
                progress[0] += 1
                done = progress[0]
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-5)
                eta = (total - done) / max(rate, 1e-5)
                print(
                    f"[{done:5d}/{total:5d}] d={actual_depth:2d} | Score: {cp:+5d} cp | "
                    f"Move: {best_move:5s} | {rate:.1f} pos/s (ETA: {eta/60:.1f}m)",
                    end="\r",
                    flush=True,
                )
    finally:
        try:
            sf.quit()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens", required=True, help="input file containing FENs")
    parser.add_argument(
        "--stockfish",
        default="/home/hemesh/Videos/8chess/bin/stockfish/stockfish-linux-x86-64-universal",
    )
    parser.add_argument("--depth", type=int, default=25, help="search depth (default: 25)")
    parser.add_argument("--time-limit", type=float, default=30.0, help="max seconds per position (default: 30s)")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel engine instances (default: 5)")
    parser.add_argument("--threads-per-eng", type=int, default=2, help="threads per instance (default: 2)")
    parser.add_argument("--hash", type=int, default=256, help="hash size per instance in MB (default: 256 MB)")
    parser.add_argument("--out-csv", default="data/sf_depth25_gold.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    fens = []
    with open(args.fens) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fen = line.split(",")[0].strip('"')
            fens.append(fen)

    if args.limit:
        fens = fens[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    # Resume support
    already_done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for line in f:
                p = line.strip().split(",")
                if p and not line.startswith("fen"):
                    already_done.add(p[0].strip('"'))

    todo = [f for f in fens if f not in already_done]
    print(f"Total positions: {len(fens):,} | Already evaluated: {len(already_done):,} | To evaluate: {len(todo):,}")
    print(
        f"Stockfish: {args.stockfish} ({args.concurrency} instances x {args.threads_per_eng} threads = {args.concurrency*args.threads_per_eng} threads total)"
    )
    print(f"Target Depth: {args.depth} | Output: {args.out_csv}\n")

    if not todo:
        print("All positions already evaluated!")
        return 0

    if not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0:
        with open(args.out_csv, "w") as f:
            f.write("fen,depth,score_cp,best_move,nodes,time_s\n")

    # Split todo into chunks for workers
    chunks = [[] for _ in range(args.concurrency)]
    for i, fen in enumerate(todo):
        chunks[i % args.concurrency].append(fen)

    lock = threading.Lock()
    progress = [len(already_done)]
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                worker_loop,
                args.stockfish,
                chunks[i],
                args.depth,
                args.time_limit,
                args.threads_per_eng,
                args.hash,
                args.out_csv,
                lock,
                progress,
                len(fens),
                t_start,
            )
            for i in range(args.concurrency)
            if chunks[i]
        ]
        for fut in as_completed(futures):
            fut.result()

    print(f"\n\nEvaluation complete in {time.time()-t_start:.1f}s!")
    print(f"Saved to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

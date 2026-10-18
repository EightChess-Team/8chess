"""High-throughput 10,000-game match runner & loss harvester: 8Chess vs Stockfish 19.

Plays color-balanced pairs across the 5,000 official Stockfish 8-move openings
(10,000 games total), tracking live Elo and streaming both the full PGN and
critical loss positions in real time.

Usage:
    python3 tools/deep_mining/game_generator.py \
        --games 10000 --concurrency 6 --tc "1.0+0.01" \
        --openings books/8moves_v3.fen \
        --pgn data/matches_10k.pgn \
        --out-csv data/positions_10k.csv \
        --losses-csv data/losses_10k.csv
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess
import chess.engine
import chess.pgn


def elo_stats(wins: int, losses: int, draws: int) -> tuple[float, float, float]:
    n = wins + losses + draws
    if n == 0:
        return 0.5, 0.0, 0.0

    score = (wins + 0.5 * draws) / n
    var = ((wins * (1.0 - score) ** 2) + (draws * (0.5 - score) ** 2) + (losses * (0.0 - score) ** 2)) / max(n - 1, 1)
    se = math.sqrt(var / n)

    clamped_score = max(min(score, 0.999), 0.001)
    elo = -400.0 * math.log10(1.0 / clamped_score - 1.0)

    score_lo = max(min(score - 1.96 * se, 0.999), 0.001)
    score_hi = max(min(score + 1.96 * se, 0.999), 0.001)
    elo_lo = -400.0 * math.log10(1.0 / score_lo - 1.0)
    elo_hi = -400.0 * math.log10(1.0 / score_hi - 1.0)
    margin = abs(elo_hi - elo_lo) / 2.0

    return score, elo, margin


def play_one_game(
    engine1_path: str,
    engine2_path: str,
    fen: str,
    e1_white: bool,
    base_time: float,
    inc_time: float,
    opt1: dict[str, str],
    opt2: dict[str, str],
    max_plies: int = 250,
) -> tuple[str, bool, list[tuple[str, int]], chess.pgn.Game]:
    """Plays one game. Returns (res_str, e1_lost, history, pgn_game)."""
    w_path = engine1_path if e1_white else engine2_path
    b_path = engine2_path if e1_white else engine1_path
    w_opt = opt1 if e1_white else opt2
    b_opt = opt2 if e1_white else opt1

    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "8Chess vs Stockfish 19 (10k Gauntlet)"
    game.headers["White"] = "8Chess" if e1_white else "Stockfish19"
    game.headers["Black"] = "Stockfish19" if e1_white else "8Chess"
    game.headers["FEN"] = fen
    game.setup(board)

    w_eng = chess.engine.SimpleEngine.popen_uci(w_path)
    b_eng = chess.engine.SimpleEngine.popen_uci(b_path)

    for k, v in w_opt.items():
        w_eng.configure({k: v})
    for k, v in b_opt.items():
        b_eng.configure({k: v})

    wtime, btime = base_time, base_time
    history = []
    node = game

    try:
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            eng = w_eng if board.turn == chess.WHITE else b_eng
            t0 = time.time()
            limit = chess.engine.Limit(
                white_clock=max(wtime, 0.01),
                black_clock=max(btime, 0.01),
                white_inc=inc_time,
                black_inc=inc_time,
            )
            res = eng.play(board, limit)
            dt = time.time() - t0

            if board.turn == chess.WHITE:
                wtime = max(wtime - dt + inc_time, 0.005)
            else:
                btime = max(btime - dt + inc_time, 0.005)

            if res.move is None or res.move not in board.legal_moves:
                break

            board.push(res.move)
            node = node.add_variation(res.move)
            ply += 1
            if ply >= 10:
                history.append((board.fen(), ply))

        outcome = board.outcome(claim_draw=True)
        res_str = outcome.result() if outcome else "1/2-1/2"
        game.headers["Result"] = res_str

        # Did Engine 1 (8Chess) lose?
        e1_lost = (e1_white and res_str == "0-1") or (not e1_white and res_str == "1-0")
        return res_str, e1_lost, history, game

    finally:
        try:
            w_eng.quit()
        except Exception:
            pass
        try:
            b_eng.quit()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 10,000 games of 8Chess vs Stockfish 19")
    parser.add_argument("--engine1", default="target/release/eightchess")
    parser.add_argument("--opt1", action="append", default=["EvalFile=networks/eightchess_lc0_best.nnue"])
    parser.add_argument(
        "--engine2",
        default="/home/hemesh/Videos/8chess/bin/stockfish/stockfish-linux-x86-64-universal",
    )
    parser.add_argument("--opt2", action="append", default=[])
    parser.add_argument("--openings", default="books/8moves_v3.fen")
    parser.add_argument("--games", type=int, default=10000, help="total games (default: 10,000)")
    parser.add_argument("--concurrency", type=int, default=6, help="parallel game workers")
    parser.add_argument("--tc", default="1.0+0.01", help="time control base+inc in seconds")
    parser.add_argument("--pgn", default="data/matches_10k.pgn")
    parser.add_argument("--out-csv", default="data/positions_10k.csv")
    parser.add_argument("--losses-csv", default="data/losses_10k.csv")
    args = parser.parse_args()

    parts = args.tc.split("+")
    base_time = float(parts[0])
    inc_time = float(parts[1]) if len(parts) > 1 else 0.0

    def parse_opts(opts):
        d = {}
        for o in opts:
            if "=" in o:
                k, v = o.split("=", 1)
                d[k.strip()] = v.strip()
        return d

    opt1 = parse_opts(args.opt1)
    opt2 = parse_opts(args.opt2)

    with open(args.openings) as f:
        raw_openings = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(raw_openings)} opening books from {args.openings}")

    # Build balanced color-pairs: each opening played twice (once White, once Black)
    pairs_needed = (args.games + 1) // 2
    tasks = []
    for i in range(pairs_needed):
        fen = raw_openings[i % len(raw_openings)]
        tasks.append((fen, True))   # 8Chess White
        tasks.append((fen, False))  # 8Chess Black
    tasks = tasks[: args.games]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.pgn)), exist_ok=True)
    file_lock = threading.Lock()

    total_positions = 0
    total_losses = 0
    seen_fens = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for line in f:
                p = line.strip().split(",")
                if p:
                    seen_fens.add(p[0].strip('"'))
        total_positions = len(seen_fens)
        print(f"Resuming with {total_positions:,} existing positions in {args.out_csv}")

    print(f"=== Launching 10,000 Games: 8Chess vs Stockfish 19 ===")
    print(f"Concurrency: {args.concurrency} workers | TC: {args.tc}")
    print(f"Engine 1: {args.engine1} {opt1}")
    print(f"Engine 2: {args.engine2} {opt2}")
    print(f"PGN Archive: {args.pgn}")
    print(f"Harvested Positions: {args.out_csv}")
    print(f"Mined Losses: {args.losses_csv}\n")

    t0 = time.time()
    games_completed = 0
    e1_wins = 0
    e1_losses = 0
    draws = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for fen, e1_white in tasks[: args.concurrency * 2]:
            fut = pool.submit(play_one_game, args.engine1, args.engine2, fen, e1_white, base_time, inc_time, opt1, opt2)
            futures[fut] = (fen, e1_white)

        task_idx = len(futures)

        while futures:
            for fut in as_completed(list(futures.keys())):
                fen, e1_white = futures.pop(fut)
                try:
                    res_str, e1_lost, history, pgn_game = fut.result()
                except Exception:
                    continue

                games_completed += 1

                # Tally score
                if e1_white:
                    if res_str == "1-0":
                        e1_wins += 1
                    elif res_str == "0-1":
                        e1_losses += 1
                    else:
                        draws += 1
                else:
                    if res_str == "0-1":
                        e1_wins += 1
                    elif res_str == "1-0":
                        e1_losses += 1
                    else:
                        draws += 1

                # Save PGN and positions
                new_records = []
                new_losses = []
                for fen_str, ply in history:
                    if fen_str not in seen_fens:
                        seen_fens.add(fen_str)
                        new_records.append(f'"{fen_str}",{res_str},{ply}\n')
                        if e1_lost:
                            new_losses.append(f'"{fen_str}",{res_str},{ply}\n')

                with file_lock:
                    with open(args.pgn, "a") as f_pgn:
                        print(pgn_game, file=f_pgn, end="\n\n")
                    if new_records:
                        with open(args.out_csv, "a") as f_csv:
                            f_csv.writelines(new_records)
                        total_positions += len(new_records)
                    if new_losses:
                        with open(args.losses_csv, "a") as f_loss:
                            f_loss.writelines(new_losses)
                        total_losses += len(new_losses)

                score, elo, margin = elo_stats(e1_wins, e1_losses, draws)
                sign = "+" if elo >= 0 else ""
                elapsed = time.time() - t0
                gps = games_completed / max(elapsed, 1)

                print(
                    f"[{games_completed:5d}/{args.games:5d}] 8Chess: +{e1_wins} ={draws} -{e1_losses} "
                    f"({score*100:.1f}%) | Elo: {sign}{elo:.1f} +/- {margin:.1f} | "
                    f"Pos: {total_positions:,} (Losses: {total_losses:,}) | {gps:.1f} g/s",
                    end="\r",
                    flush=True,
                )

                # Refill queue
                if task_idx < len(tasks):
                    next_fen, next_white = tasks[task_idx]
                    task_idx += 1
                    next_fut = pool.submit(play_one_game, args.engine1, args.engine2, next_fen, next_white, base_time, inc_time, opt1, opt2)
                    futures[next_fut] = (next_fen, next_white)

                break

    print(f"\n\n=== 10,000 Games Completed in {time.time()-t0:.1f}s ===")
    score, elo, margin = elo_stats(e1_wins, e1_losses, draws)
    sign = "+" if elo >= 0 else ""
    print(f"Final 8Chess Score: {e1_wins + 0.5 * draws}/{games_completed} ({score*100:.2f}%)")
    print(f"Record: +{e1_wins} ={draws} -{e1_losses}")
    print(f"Elo Difference: {sign}{elo:.1f} +/- {margin:.1f}")
    print(f"Harvested Unique Positions: {total_positions:,}")
    print(f"Mined Loss Positions: {total_losses:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

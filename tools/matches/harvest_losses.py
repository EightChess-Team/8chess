"""High-throughput continuous match & loss harvester: 8Chess vs Stockfish 19.

Runs continuous parallel games between 8Chess and Stockfish 19 on local CPU cores.
Filters every lost game by 8Chess, extracts the critical positions, and continuously
appends them to a structured loss dataset ready to upload to Kaggle.
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

_OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkb1r/pppppp1p/5np1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
]


def play_harvest_game(
    eightchess_path: str,
    stockfish_path: str,
    fen: str,
    eightchess_color: chess.Color,
    base_time: float,
    inc_time: float,
    eightchess_opt: dict[str, str],
    stockfish_opt: dict[str, str],
) -> tuple[str, list[str]]:
    """Plays a single game. If 8Chess loses, returns (result, list_of_fens). Else (result, [])."""
    white_path = eightchess_path if eightchess_color == chess.WHITE else stockfish_path
    black_path = stockfish_path if eightchess_color == chess.WHITE else eightchess_path
    white_opt = eightchess_opt if eightchess_color == chess.WHITE else stockfish_opt
    black_opt = stockfish_opt if eightchess_color == chess.WHITE else eightchess_opt

    board = chess.Board(fen)
    w_eng = chess.engine.SimpleEngine.popen_uci(white_path)
    b_eng = chess.engine.SimpleEngine.popen_uci(black_path)

    for k, v in white_opt.items():
        w_eng.configure({k: v})
    for k, v in black_opt.items():
        b_eng.configure({k: v})

    wtime, btime = base_time, base_time
    history_fens = []

    try:
        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= 200:
            eng = w_eng if board.turn == chess.WHITE else b_eng
            t0 = time.time()
            limit = chess.engine.Limit(
                white_clock=max(wtime, 0.02),
                black_clock=max(btime, 0.02),
                white_inc=inc_time,
                black_inc=inc_time,
            )
            res = eng.play(board, limit)
            elapsed = time.time() - t0

            if board.turn == chess.WHITE:
                wtime = max(wtime - elapsed + inc_time, 0.005)
            else:
                btime = max(btime - elapsed + inc_time, 0.005)

            if res.move is None or res.move not in board.legal_moves:
                break

            board.push(res.move)
            if board.fullmove_number >= 6:
                history_fens.append(board.fen())

        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "1/2-1/2"

        eightchess_lost = (eightchess_color == chess.WHITE and result == "0-1") or (
            eightchess_color == chess.BLACK and result == "1-0"
        )

        if eightchess_lost:
            return result, history_fens
        return result, []

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
    parser = argparse.ArgumentParser(description="Continuous 8Chess vs Stockfish 19 loss harvester")
    parser.add_argument("--eightchess", default="target/release/eightchess")
    parser.add_argument(
        "--stockfish",
        default="/home/hemesh/Videos/8chess/bin/stockfish/stockfish-linux-x86-64-universal",
    )
    parser.add_argument("--tc", default="1.5+0.02", help="base+inc time control in seconds")
    parser.add_argument("--concurrency", type=int, default=6, help="concurrent games")
    parser.add_argument("--target-losses", type=int, default=5000, help="stop after mining N loss positions")
    parser.add_argument("--out-csv", default="data/harvested_losses.csv")
    args = parser.parse_args()

    parts = args.tc.split("+")
    base_time = float(parts[0])
    inc_time = float(parts[1]) if len(parts) > 1 else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    csv_lock = threading.Lock()

    print(f"=== Starting Continuous Loss Harvester ===")
    print(f"8Chess: {args.eightchess}")
    print(f"Stockfish 19: {args.stockfish}")
    print(f"TC: {args.tc} | Concurrency: {args.concurrency}")
    print(f"Writing mined losses to: {args.out_csv}\n")

    total_games = 0
    eightchess_wins = 0
    eightchess_losses = 0
    draws = 0
    mined_positions = 0
    t0 = time.time()

    # Track unique positions to avoid duplicate FENs
    seen_fens = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for line in f:
                parts = line.strip().split(",")
                if parts:
                    seen_fens.add(parts[0])
        mined_positions = len(seen_fens)
        print(f"Resuming with {mined_positions} existing positions in {args.out_csv}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while mined_positions < args.target_losses:
            batch_tasks = []
            for _ in range(args.concurrency * 2):
                fen = random.choice(_OPENINGS)
                color = random.choice([chess.WHITE, chess.BLACK])
                batch_tasks.append(
                    pool.submit(
                        play_harvest_game,
                        args.eightchess,
                        args.stockfish,
                        fen,
                        color,
                        base_time,
                        inc_time,
                        {},
                        {},
                    )
                )

            for fut in as_completed(batch_tasks):
                try:
                    result, loss_fens = fut.result()
                except Exception as e:
                    continue

                total_games += 1
                if loss_fens:
                    eightchess_losses += 1
                    # New loss positions
                    new_fens = [f for f in loss_fens if f not in seen_fens]
                    with csv_lock:
                        with open(args.out_csv, "a") as f:
                            for fen_str in new_fens:
                                f.write(f"{fen_str},0.0\n")
                                seen_fens.add(fen_str)
                    mined_positions += len(new_fens)
                elif result == "1/2-1/2":
                    draws += 1
                else:
                    eightchess_wins += 1

                elapsed = time.time() - t0
                gps = total_games / max(elapsed, 1)
                print(
                    f"Games: {total_games:4d} | 8Chess: +{eightchess_wins} ={draws} -{eightchess_losses} | "
                    f"Mined Loss Positions: {mined_positions:,} / {args.target_losses:,} | {gps:.1f} games/s",
                    end="\r",
                    flush=True,
                )

                if mined_positions >= args.target_losses:
                    break

    print(f"\n\nHarvest complete! Mined {mined_positions:,} loss positions in {time.time()-t0:.1f}s")
    print(f"Dataset saved to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

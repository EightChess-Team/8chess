"""High-performance UCI match runner with color-pair openings and Elo statistics.

Usage:
    python -m tools.matches.match_runner \
        --engine1 target/release/eightchess --name1 "8Chess" \
        --engine2 /usr/games/stockfish --name2 "Stockfish16" \
        --games 100 --concurrency 4 --tc "5+0.05" \
        --pgn results/matches.pgn
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
from typing import Optional

import chess
import chess.engine
import chess.pgn


def elo_stats(wins: int, losses: int, draws: int) -> tuple[float, float, float]:
    """Calculates score, Elo difference, and 95% confidence interval error margin."""
    n = wins + losses + draws
    if n == 0:
        return 0.5, 0.0, 0.0

    score = (wins + 0.5 * draws) / n
    # Approximate variance of score
    var = ((wins * (1.0 - score) ** 2) + (draws * (0.5 - score) ** 2) + (losses * (0.0 - score) ** 2)) / max(n - 1, 1)
    se = math.sqrt(var / n)

    # Elo difference
    clamped_score = max(min(score, 0.999), 0.001)
    elo = -400.0 * math.log10(1.0 / clamped_score - 1.0)

    # 95% confidence interval
    score_lo = max(min(score - 1.96 * se, 0.999), 0.001)
    score_hi = max(min(score + 1.96 * se, 0.999), 0.001)
    elo_lo = -400.0 * math.log10(1.0 / score_lo - 1.0)
    elo_hi = -400.0 * math.log10(1.0 / score_hi - 1.0)
    margin = abs(elo_hi - elo_lo) / 2.0

    return score, elo, margin


def parse_tc(tc_str: str) -> tuple[float, float]:
    """Parses time control string 'base+inc' (in seconds), e.g. '5+0.05'."""
    parts = tc_str.split("+")
    base = float(parts[0])
    inc = float(parts[1]) if len(parts) > 1 else 0.0
    return base, inc


# Standard openings for book-based testing
_DEFAULT_OPENINGS = [
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
    "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkbnr/pp1ppppp/8/8/2pP4/4P3/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkb1r/pppppp1p/5np1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
]


def play_game(
    engine1_path: str,
    engine2_path: str,
    name1: str,
    name2: str,
    fen: str,
    swapped: bool,
    base_time: float,
    inc_time: float,
    opt1: dict[str, str],
    opt2: dict[str, str],
    max_moves: int = 250,
) -> chess.pgn.Game:
    """Plays a single game between two engines from a given opening FEN."""
    white_path, black_path = (engine2_path, engine1_path) if swapped else (engine1_path, engine2_path)
    white_name, black_name = (name2, name1) if swapped else (name1, name2)
    white_opt, black_opt = (opt2, opt1) if swapped else (opt1, opt2)

    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "8Chess Benchmark Match"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["FEN"] = fen
    game.setup(board)

    # Initialize engines
    white_eng = chess.engine.SimpleEngine.popen_uci(white_path)
    black_eng = chess.engine.SimpleEngine.popen_uci(black_path)

    for k, v in white_opt.items():
        white_eng.configure({k: v})
    for k, v in black_opt.items():
        black_eng.configure({k: v})

    wtime = base_time
    btime = base_time

    node = game
    try:
        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
            eng = white_eng if board.turn == chess.WHITE else black_eng
            t0 = time.time()
            limit = chess.engine.Limit(
                white_clock=max(wtime, 0.05),
                black_clock=max(btime, 0.05),
                white_inc=inc_time,
                black_inc=inc_time,
            )
            result = eng.play(board, limit)
            elapsed = time.time() - t0

            if board.turn == chess.WHITE:
                wtime = max(wtime - elapsed + inc_time, 0.01)
            else:
                btime = max(btime - elapsed + inc_time, 0.01)

            if result.move is None or result.move not in board.legal_moves:
                break

            board.push(result.move)
            node = node.add_variation(result.move)

        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            game.headers["Result"] = outcome.result()
            game.headers["Termination"] = outcome.termination.name
        else:
            game.headers["Result"] = "1/2-1/2"
            game.headers["Termination"] = "MAX_MOVES_EXCEEDED"

    finally:
        try:
            white_eng.quit()
        except Exception:
            pass
        try:
            black_eng.quit()
        except Exception:
            pass

    return game


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UCI engine matches with live Elo estimation")
    parser.add_argument("--engine1", required=True, help="path to engine 1 executable")
    parser.add_argument("--name1", default="Engine1")
    parser.add_argument("--opt1", action="append", default=[], help="UCI option key=val for engine 1")
    parser.add_argument("--engine2", required=True, help="path to engine 2 executable")
    parser.add_argument("--name2", default="Engine2")
    parser.add_argument("--opt2", action="append", default=[], help="UCI option key=val for engine 2")
    parser.add_argument("--games", type=int, default=50, help="total number of games (even count)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel worker games")
    parser.add_argument("--tc", default="5+0.05", help="time control base+inc in seconds")
    parser.add_argument("--openings", default=None, help="path to file with opening FENs (1 per line)")
    parser.add_argument("--pgn", default="matches.pgn", help="output PGN file")
    args = parser.parse_args()

    base_time, inc_time = parse_tc(args.tc)

    def parse_opts(opts_list: list[str]) -> dict[str, str]:
        d = {}
        for o in opts_list:
            if "=" in o:
                k, v = o.split("=", 1)
                d[k.strip()] = v.strip()
        return d

    opt1 = parse_opts(args.opt1)
    opt2 = parse_opts(args.opt2)

    # Load opening FENs
    if args.openings and os.path.exists(args.openings):
        with open(args.openings) as f:
            openings = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        openings = _DEFAULT_OPENINGS

    # Organize games in color-swapped pairs
    pairs_needed = (args.games + 1) // 2
    tasks = []
    for i in range(pairs_needed):
        fen = openings[i % len(openings)]
        tasks.append((fen, False))
        tasks.append((fen, True))
    tasks = tasks[: args.games]

    print(f"=== Match: {args.name1} vs {args.name2} ===")
    print(f"TC: {args.tc} | Games: {len(tasks)} | Concurrency: {args.concurrency}")
    print(f"Engine 1: {args.engine1} {opt1}")
    print(f"Engine 2: {args.engine2} {opt2}")
    print(f"Recording to: {args.pgn}\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.pgn)), exist_ok=True)
    pgn_lock = threading.Lock()

    wins1 = 0
    losses1 = 0
    draws = 0
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                play_game,
                args.engine1,
                args.engine2,
                args.name1,
                args.name2,
                fen,
                swapped,
                base_time,
                inc_time,
                opt1,
                opt2,
            ): (fen, swapped)
            for fen, swapped in tasks
        }

        for fut in as_completed(futures):
            fen, swapped = futures[fut]
            try:
                game = fut.result()
            except Exception as e:
                print(f"[ERROR] Game error: {e}", file=sys.stderr)
                continue

            res = game.headers.get("Result", "*")
            with pgn_lock:
                with open(args.pgn, "a") as f:
                    print(game, file=f, end="\n\n")

            completed += 1

            # Engine 1 was White if not swapped, Black if swapped
            if not swapped:
                if res == "1-0":
                    wins1 += 1
                elif res == "0-1":
                    losses1 += 1
                else:
                    draws += 1
            else:
                if res == "0-1":
                    wins1 += 1
                elif res == "1-0":
                    losses1 += 1
                else:
                    draws += 1

            score, elo, margin = elo_stats(wins1, losses1, draws)
            sign = "+" if elo >= 0 else ""
            elapsed = time.time() - t0
            print(
                f"[{completed:3d}/{len(tasks):3d}] {args.name1}: +{wins1} ={draws} -{losses1} "
                f"({score*100:.1f}%) | Elo: {sign}{elo:.1f} +/- {margin:.1f} | {elapsed:.0f}s"
            )

    print("\n=== Match Final Result ===")
    score, elo, margin = elo_stats(wins1, losses1, draws)
    sign = "+" if elo >= 0 else ""
    print(f"Total: {completed} games")
    print(f"{args.name1} score: {wins1 + 0.5 * draws}/{completed} ({score*100:.2f}%)")
    print(f"{args.name1}: +{wins1} ={draws} -{losses1}")
    print(f"Estimated Elo diff: {sign}{elo:.1f} +/- {margin:.1f} (95% CI)")
    print(f"PGN saved to {args.pgn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

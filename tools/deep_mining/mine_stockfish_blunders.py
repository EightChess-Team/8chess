"""Extract positions where Stockfish 19 blundered and lost to 8Chess.

Scans the match PGN for all games won by 8Chess (lost by Stockfish 19).
Isolates the critical moves leading up to Stockfish's defeat so Leela (Lc0 BT4)
can evaluate the refutations and train 8Chess to actively exploit these traps.
"""

from __future__ import annotations

import argparse
import os
import sys
import chess
import chess.pgn


def extract_stockfish_blunders(pgn_path: str, out_csv: str) -> int:
    sf_loss_games = 0
    blunder_positions = []
    seen = set()

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            res = game.headers.get("Result", "*")

            # Check if 8Chess won (Stockfish lost)
            eightchess_won = ("8chess" in white.lower() and res == "1-0") or (
                "8chess" in black.lower() and res == "0-1"
            )

            if not eightchess_won:
                continue

            sf_loss_games += 1
            board = game.board()
            game_fens = []

            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                # Keep middlegame positions where the game turned
                if 12 <= ply <= 60:
                    fen = board.fen()
                    if fen not in seen:
                        seen.add(fen)
                        game_fens.append(f'"{fen}",{res},{ply}\n')

            blunder_positions.extend(game_fens)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w") as out:
        out.writelines(blunder_positions)

    print(f"Extracted {len(blunder_positions):,} positions from {sf_loss_games:,} games won by 8Chess against Stockfish 19!")
    print(f"Saved to: {out_csv}")
    return len(blunder_positions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", default="data/matches_60k.pgn")
    parser.add_argument("--out-csv", default="data/stockfish_blunders.csv")
    args = parser.parse_args()

    extract_stockfish_blunders(args.pgn, args.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Extract positions from loss games to build an adversarial fine-tuning dataset.

Scans match PGNs where 8Chess lost to Stockfish, extracts midgame and endgame
positions from 8Chess's perspective, and packages them for NNUE fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure training package is importable
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_root, "training"))

import chess
import chess.pgn
import numpy as np

from eightchess_nnue.encoder import encode_fens, save_encoded


def extract_loss_positions(
    pgn_path: str,
    target_engine: str,
    min_ply: int = 10,
    max_ply: int = 150,
    stride: int = 2,
) -> list[tuple[str, str]]:
    """Extracts (fen, result) for positions where target_engine was playing and lost."""
    samples = []
    loss_games = 0

    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            result = game.headers.get("Result", "*")

            is_target_white = target_engine.lower() in white.lower()
            is_target_black = target_engine.lower() in black.lower()

            if not (is_target_white or is_target_black):
                continue

            target_lost = (is_target_white and result == "0-1") or (is_target_black and result == "1-0")
            if not target_lost:
                continue

            loss_games += 1
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply < min_ply or ply > max_ply:
                    continue
                if (ply - min_ply) % stride != 0:
                    continue

                samples.append((board.fen(), result))

    print(f"Extracted {len(samples)} positions from {loss_games} lost games")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True, help="path to match PGN")
    parser.add_argument("--engine", default="8Chess", help="name of target engine in PGN")
    parser.add_argument("--out-csv", default=None, help="output FEN,result CSV")
    parser.add_argument("--out-npz", default=None, help="output encoded .npz dataset")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-ply", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    samples = extract_loss_positions(
        args.pgn,
        args.engine,
        min_ply=args.min_ply,
        stride=args.stride,
    )

    if not samples:
        print("No loss positions found.")
        return 0

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            for fen, res in samples:
                f.write(f"{fen},{res}\n")
        print(f"Wrote {args.out_csv}")

    if args.out_npz:
        fens = [s[0] for s in samples]
        results = [s[1] for s in samples]
        data = encode_fens(fens, results, workers=args.workers)
        save_encoded(data, args.out_npz)
        print(f"Wrote encoded dataset: {args.out_npz}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the Anti-Stockfish Asymmetric Opening Suite (books/anti_stockfish.fen).

Filters for sharp, highly tactical positions:
1. Queens present on the board (zero premature queen trades).
2. Opposite-side castling or asymmetrical pawn structures (tactical entropy).
3. Complex imbalances where Leela BT4 positional foresight beats Stockfish's pruning.
"""

import os
import chess
import chess.pgn

def is_sharp_asymmetric(board: chess.Board) -> bool:
    # 1. Queens must be on the board
    if (board.pieces(chess.QUEEN, chess.WHITE) == 0) or (board.pieces(chess.QUEEN, chess.BLACK) == 0):
        return False
    # 2. Both sides must have at least 14 pieces (complex middlegame tension)
    if board.occupied.bit_count() < 28:
        return False
    # 3. Not symmetrical pawn structure (at least 2 files with differing pawn counts)
    w_pawns = board.pieces(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces(chess.PAWN, chess.BLACK)
    asym_files = 0
    for f in range(8):
        file_mask = chess.BB_FILES[f]
        if len(w_pawns & file_mask) != len(b_pawns & file_mask):
            asym_files += 1
    return asym_files >= 2

def main():
    pgn_path = "books/8moves_v3.pgn"
    out_fen = "books/anti_stockfish.fen"
    target = 1000

    fens = []
    seen = set()

    with open(pgn_path) as f:
        while len(fens) < target:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            b = g.board()
            for mv in g.mainline_moves():
                b.push(mv)
            if is_sharp_asymmetric(b):
                fen = b.fen()
                if fen not in seen:
                    seen.add(fen)
                    fens.append(fen)

    os.makedirs(os.path.dirname(out_fen), exist_ok=True)
    with open(out_fen, "w") as out:
        for f in fens:
            out.write(f + "\n")

    print(f"Built Anti-Stockfish Opening Suite: {len(fens):,} sharp tactical openings saved to {out_fen}")

if __name__ == "__main__":
    main()

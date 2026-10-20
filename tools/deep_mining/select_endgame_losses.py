"""Extract 30,000 late-endgame loss positions from the 60,000-game match.

Filters specifically for plies >= 50 and piece counts <= 10 (Output Buckets 0 and 1)
to patch 8Chess's late-game grinding vulnerabilities against Stockfish 19.
"""

import os
import chess

def main():
    in_csv = "data/losses_60k.csv"
    out_csv = "data/endgame_losses.csv"
    target_count = 30000

    seen = set()
    records = []

    with open(in_csv) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 3:
                fen = p[0].strip('"')
                ply = int(p[2])
                res = p[1].strip()
                if ply >= 50 and fen not in seen:
                    b = chess.Board(fen)
                    if b.occupied.bit_count() <= 10:
                        seen.add(fen)
                        records.append(f'"{fen}",{res},{ply}\n')
                    if len(records) >= target_count:
                        break

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as out:
        out.writelines(records)

    print(f"Saved {len(records):,} late-endgame loss positions to {out_csv}")

if __name__ == "__main__":
    main()

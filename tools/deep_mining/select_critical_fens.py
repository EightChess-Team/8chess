"""Extract high-signal middlegame loss positions across all 2,932 lost games.

Focuses on plies 14-45 where critical tactical and positional blunders occurred,
filtering out opening theory and trivial endgame conversions.
"""

import sys

def main():
    in_csv = "data/losses_10k.csv"
    out_csv = "data/critical_loss_fens.txt"
    target_count = 5000

    seen = set()
    selected = []

    with open(in_csv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                fen = parts[0].strip('"')
                ply = int(parts[2])
                # Filter for critical middlegame plies (14 to 45)
                if 14 <= ply <= 45:
                    if fen not in seen:
                        seen.add(fen)
                        selected.append(fen)

    # Subsample evenly if more than target_count
    if len(selected) > target_count:
        step = len(selected) / target_count
        sampled = [selected[int(i * step)] for i in range(target_count)]
    else:
        sampled = selected

    with open(out_csv, "w") as out:
        for fen in sampled:
            out.write(fen + "\n")

    print(f"Extracted {len(sampled):,} critical loss positions (plies 14-45) to {out_csv}")

if __name__ == "__main__":
    main()

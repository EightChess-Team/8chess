"""Curate the highest-signal tactical battleground positions for Kaggle GPU mining.

Combines:
1. All 9,689 Stockfish 19 Depth-25 tactical evaluations.
2. 10,000 decisive positions where Stockfish 19 blundered and lost to 8Chess.
3. 10,000 critical positions where 8Chess was defeated by Stockfish 19.
Total: ~30,000 elite positions representing both offensive and defensive chess truth.
"""

import os
import random

def sample_csv(path, target_count, min_ply=14, max_ply=50):
    lines = []
    seen = set()
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 3 and not line.startswith("fen"):
                fen = p[0].strip('"')
                ply = int(p[2])
                if min_ply <= ply <= max_ply and fen not in seen:
                    seen.add(fen)
                    lines.append(line.strip())
    
    if len(lines) > target_count:
        step = len(lines) / target_count
        return [lines[int(i * step)] for i in range(target_count)]
    return lines

def main():
    sf_gold_path = "data/sf_depth25_gold.csv"
    sf_blunders_path = "data/stockfish_blunders_60k.csv"
    losses_path = "data/losses_60k.csv"
    out_path = "data/curated_battlegrounds.csv"

    # 1. All Depth-25 evaluations
    sf_gold = []
    with open(sf_gold_path) as f:
        for line in f:
            if not line.startswith("fen") and line.strip():
                sf_gold.append(line.strip())
    print(f"Loaded {len(sf_gold):,} Stockfish 19 Depth-25 evaluations")

    # 2. Stockfish blunders (offensive gold)
    sf_blunders = sample_csv(sf_blunders_path, 10000)
    print(f"Sampled {len(sf_blunders):,} Stockfish 19 blunder refutations")

    # 3. 8Chess losses (defensive gold)
    losses = sample_csv(losses_path, 10000)
    print(f"Sampled {len(losses):,} 8Chess loss turning points")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out:
        out.write("fen,info1,info2,info3,info4,info5\n")
        for l in sf_gold:
            out.write(l + "\n")
        for l in sf_blunders:
            out.write(l + "\n")
        for l in losses:
            out.write(l + "\n")

    total = len(sf_gold) + len(sf_blunders) + len(losses)
    print(f"Total curated battlegrounds: {total:,} positions saved to {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()

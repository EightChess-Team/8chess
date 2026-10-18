"""Build a packed, encoded training dataset from PGNs or FEN+result files.

Usage:
    python -m eightchess_nnue.build_dataset \
        --pgn lc0_games.pgn.gz --out data/lc0.npz \
        --stride 2 --min-ply 8 --max 2000000 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .dataset import sample_positions
from .encoder import encode_fens, save_encoded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", action="append", default=[])
    parser.add_argument("--fens", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max", type=int, default=None, dest="max_samples")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=8192, help="positions per encoding batch")
    args = parser.parse_args()

    if not args.pgn and not args.fens:
        parser.error("need at least one --pgn or --fens input")

    t0 = time.time()
    fens, results = sample_positions(
        pgns=args.pgn,
        fen_files=args.fens,
        stride=args.stride,
        min_ply=args.min_ply,
        max_samples=args.max_samples,
    )
    print(f"collected {len(fens)} positions in {time.time() - t0:.1f}s")
    if not fens:
        return 1

    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, len(fens), args.chunk):
        stop = min(start + args.chunk, len(fens))
        chunk_data = encode_fens(
            fens[start:stop], results[start:stop], workers=args.workers
        )
        parts.append(chunk_data)
        done = stop
        rate = done / max(time.time() - t0, 1e-9)
        print(f"encoded {done}/{len(fens)} ({rate:.0f} pos/s)", end="\r", flush=True)

    merged: dict[str, np.ndarray] = {}
    names = [k for k in parts[0] if not k.endswith("_offsets")]
    for name in names:
        merged[name] = np.concatenate([p[name] for p in parts])
    for stream in ("piece_stm", "threat_stm", "piece_opp", "threat_opp"):
        offsets = [np.zeros(1, dtype=np.int64)]
        total = 0
        for p in parts:
            o = p[f"{stream}_offsets"][1:]
            offsets.append(o + total)
            total += int(p[f"{stream}_indices"].size)
        merged[f"{stream}_offsets"] = np.concatenate(offsets)
        merged[f"{stream}_indices"] = np.concatenate([p[f"{stream}_indices"] for p in parts])

    save_encoded(merged, args.out)
    print(f"\nwrote {args.out} ({len(fens)} positions) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

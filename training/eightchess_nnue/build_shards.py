"""Stream .plain / .tsv / .pgn data into compressed .npz shards.

Encoding in bounded batches avoids the O(n) Python-list memory blowup of
`build_dataset.py`, and makes the job resumable. Each shard is a normal
.npz consumable by `train.py`.

Usage:
    python -m eightchess_nnue.build_shards \
        --input /tmp/data.plain --out-dir data/shards --shard-size 2000000
    python -m eightchess_nnue.build_shards \
        --input data/lc0_pgns --format pgn --file-stride 10 --stride 6 \
        --out-dir data/lc0_shards --shard-size 2000000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

from .dataset import iter_pgn_positions, iter_plain
from .encoder import encode_fens, encode_fens_scored, save_encoded


def iter_tsv(path: str):
    """Yields (fen, score, result) from binpack_to_tsv output.

    The converter writes `1-0` when the side to move won, matching the
    side-to-move convention of `iter_plain`.
    """
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            fen, score, result = parts[0], parts[1], parts[2]
            r = 1 if result == "1-0" else (-1 if result == "0-1" else 0)
            try:
                yield fen, int(score), r
            except ValueError:
                continue


def _write_progress(path: str, raw_consumed: int, shard_idx: int, positions: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {"raw_consumed": raw_consumed, "shard_idx": shard_idx, "positions": positions},
            f,
        )
    os.replace(tmp, path)


def _resolve_pgn_inputs(spec: str) -> list[str]:
    if os.path.isdir(spec):
        paths = sorted(glob.glob(os.path.join(spec, "*.pgn")))
    elif any(c in spec for c in "*?["):
        paths = sorted(glob.glob(spec))
    else:
        paths = [spec]
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--format", choices=["plain", "tsv", "pgn"], default="plain")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=2_000_000)
    ap.add_argument("--stride", type=int, default=1, help="ply stride for pgn, position stride for plain/tsv")
    ap.add_argument("--file-stride", type=int, default=1, help="take every Nth pgn file")
    ap.add_argument("--file-start", type=int, default=0, help="skip the first N pgn files")
    ap.add_argument("--file-limit", type=int, default=0, help="use at most N pgn files (0 = all)")
    ap.add_argument("--prefix", default=None, help="override output shard prefix")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--max", type=int, default=None, dest="max_samples")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--scaling", type=float, default=410.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.format == "pgn":
        label = os.path.basename(os.path.normpath(args.input))
    else:
        label = os.path.basename(args.input)
    prefix = args.prefix or re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    progress_path = os.path.join(args.out_dir, prefix + ".progress.json")

    raw_consumed = 0
    shard_idx = 0
    total_positions = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            st = json.load(f)
        raw_consumed = int(st["raw_consumed"])
        shard_idx = int(st["shard_idx"])
        total_positions = int(st["positions"])
        print(f"resuming {prefix}: raw_consumed={raw_consumed:,} shards={shard_idx}", flush=True)

    if args.format == "plain":
        it = iter_plain(args.input)
    elif args.format == "tsv":
        it = iter_tsv(args.input)
    else:
        paths = _resolve_pgn_inputs(args.input)
        if args.file_stride > 1:
            paths = paths[:: args.file_stride]
        start = args.file_start
        if args.file_limit:
            paths = paths[start : start + args.file_limit]
        elif start:
            paths = paths[start:]
        print(
            f"pgn: {len(paths)} files (file-start {args.file_start}, "
            f"file-stride {args.file_stride})",
            flush=True,
        )
        it = iter_pgn_positions(paths, stride=args.stride, min_ply=args.min_ply)

    fens: list[str] = []
    scores: list[int] = []
    results: list = []
    raw_idx = 0
    t0 = time.time()
    positions_this_run = 0
    stop = False

    def flush() -> None:
        nonlocal shard_idx, total_positions, positions_this_run
        path = os.path.join(args.out_dir, f"{prefix}.{shard_idx:05d}.npz")
        if args.format == "pgn":
            data = encode_fens(fens, results, workers=args.workers)
        else:
            data = encode_fens_scored(
                fens, scores, results, lam=args.lam, scaling=args.scaling, workers=args.workers
            )
        save_encoded(data, path)
        shard_idx += 1
        total_positions += len(fens)
        positions_this_run += len(fens)
        _write_progress(progress_path, raw_idx, shard_idx, total_positions)
        elapsed = time.time() - t0
        rate = positions_this_run / max(elapsed, 1e-9)
        print(
            f"shard {path} ({len(fens)} pos) | total {total_positions:,} | "
            f"{rate:.0f} pos/s this run",
            flush=True,
        )
        fens.clear()
        scores.clear()
        results.clear()

    try:
        for item in it:
            if raw_idx < raw_consumed:
                raw_idx += 1
                continue
            # PGN ply stride is handled inside iter_pgn_positions.
            take = True if args.format == "pgn" else (raw_idx % args.stride == 0)
            raw_idx += 1
            if not take:
                continue
            if args.format == "pgn":
                fen, result = item
                fens.append(fen)
                results.append(result)
            else:
                fen, score, result = item
                fens.append(fen)
                scores.append(score)
                results.append(result)
            if args.max_samples is not None and total_positions + len(fens) >= args.max_samples:
                stop = True
            if len(fens) >= args.shard_size:
                flush()
                if stop:
                    break
        if fens:
            flush()
    except KeyboardInterrupt:
        print("\ninterrupted; progress saved", flush=True)
        return 130

    print(
        f"done {prefix}: {total_positions:,} positions in {shard_idx} shards, "
        f"raw_consumed={raw_idx:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

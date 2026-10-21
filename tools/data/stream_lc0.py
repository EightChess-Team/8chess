import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
from eightchess_nnue.encoder import encode_fens_scored, save_encoded


def blocks(stream):
    fen = None
    score = 0
    result = 0
    for line in stream:
        line = line.rstrip("\n")
        if line.startswith("fen "):
            fen = line[4:]
        elif line.startswith("score "):
            score = int(line[6:])
        elif line.startswith("result "):
            result = int(line[7:])
        elif line == "e" and fen is not None:
            yield fen, score, result
            fen = None
            score = 0
            result = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default="-", help="plain file or '-' for stdin")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=100_000)
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lambda", type=float, default=1.0, dest="lam")
    ap.add_argument("--scaling", type=float, default=410.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stream = sys.stdin if args.plain == "-" else open(args.plain)
    fens, scores, results = [], [], []
    total = 0
    chunk = 0
    t0 = time.time()

    for fen, score, result in blocks(stream):
        fens.append(fen)
        scores.append(score)
        results.append(result)
        if len(fens) < args.chunk_size:
            continue
        data = encode_fens_scored(
            fens,
            scores,
            results,
            lam=args.lam,
            scaling=args.scaling,
            workers=args.workers,
        )
        path = os.path.join(args.out_dir, f"chunk_{chunk:04d}.npz")
        save_encoded(data, path)
        total += len(fens)
        chunk += 1
        rate = total / max(time.time() - t0, 1e-9)
        print(f"{total:,} positions, {rate:,.0f}/s, wrote {path}", flush=True)
        fens, scores, results = [], [], []
        if args.max_positions and total >= args.max_positions:
            break

    if fens:
        data = encode_fens_scored(
            fens, scores, results, lam=args.lam, scaling=args.scaling, workers=args.workers
        )
        path = os.path.join(args.out_dir, f"chunk_{chunk:04d}.npz")
        save_encoded(data, path)
        total += len(fens)
        print(f"{total:,} positions, wrote {path}", flush=True)

    print(f"done: {total:,} positions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

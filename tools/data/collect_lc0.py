import argparse
import math
import os


def blocks(path):
    fen = None
    score = 0
    result = 0
    with open(path) as f:
        for line in f:
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
    ap.add_argument("--plain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sf-fens", default=None)
    ap.add_argument("--sf-every", type=int, default=20)
    ap.add_argument("--white-relative", action="store_true")
    ap.add_argument("--lc0-scaling", type=float, default=250.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    sf = open(args.sf_fens, "w") if args.sf_fens else None
    with open(args.out, "w") as out:
        for i, (fen, score, result) in enumerate(blocks(args.plain)):
            if args.white_relative and fen.split()[1] == "b":
                score = -score
                result = -result
            if args.lc0_scaling > 0:
                z = max(-60.0, min(60.0, score / args.lc0_scaling))
                p = 1.0 / (1.0 + math.exp(-z))
                p = min(max(p, 1e-6), 1.0 - 1e-6)
                score = int(round(410.0 * math.log(p / (1.0 - p))))
                score = max(-4000, min(4000, score))
            out.write(f"{fen}\t{score}\t{result}\n")
            if sf is not None and i % args.sf_every == 0:
                sf.write(fen + "\n")
            n += 1
            if n % 500_000 == 0:
                print(f"{n:,} positions", flush=True)
    if sf is not None:
        sf.close()
    print(f"done: {n:,} positions", flush=True)


if __name__ == "__main__":
    main()

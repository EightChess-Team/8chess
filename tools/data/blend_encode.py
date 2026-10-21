import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
from eightchess_nnue.encoder import encode_fens_scored, save_encoded


def _p(cp, scaling=410.0):
    z = max(-60.0, min(60.0, cp / scaling))
    return 1.0 / (1.0 + math.exp(-z))


def blend_cp(cps, weights, scaling=410.0):
    p = sum(w * _p(cp, scaling) for w, cp in zip(weights, cps))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return int(round(scaling * math.log(p / (1.0 - p))))


def concat(parts):
    out = {}
    for s in ("piece_stm", "threat_stm", "piece_opp", "threat_opp"):
        idx = [p[f"{s}_indices"] for p in parts]
        offs = [p[f"{s}_offsets"] for p in parts]
        merged_off = [np.zeros(1, dtype=np.int64)]
        total = 0
        for o in offs:
            merged_off.append(o[1:] + total)
            total += int(o[-1])
        out[f"{s}_indices"] = np.concatenate(idx)
        out[f"{s}_offsets"] = np.concatenate(merged_off)
    out["bucket"] = np.concatenate([p["bucket"] for p in parts])
    out["target"] = np.concatenate([p["target"] for p in parts])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--sf", default=None)
    ap.add_argument("--disagreements", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--disag-repeat", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.8, dest="lam")
    ap.add_argument("--scaling", type=float, default=410.0)
    args = ap.parse_args()

    fens, lc0s, results = [], [], []
    with open(args.sample) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3:
                continue
            fens.append(p[0])
            lc0s.append(int(p[1]))
            results.append(int(p[2]))
    print(f"sample: {len(fens):,} positions", flush=True)

    sf = {}
    if args.sf:
        with open(args.sf) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 2:
                    sf[p[0]] = int(p[1])
        print(f"stockfish labels: {len(sf):,}", flush=True)

    bulk = encode_fens_scored(
        fens, lc0s, results, lam=args.lam, scaling=args.scaling, workers=args.workers
    )
    print("encoded Lc0 bulk", flush=True)

    parts = [bulk]

    if sf:
        bf, bs, br = [], [], []
        for fen, lc0, res in zip(fens, lc0s, results):
            s = sf.get(fen)
            if s is None:
                continue
            cp = blend_cp([s, lc0], [0.5, 0.5])
            bf.append(fen)
            bs.append(cp)
            br.append(res)
        print(f"fresh SF+Lc0 subset: {len(bf):,}", flush=True)
        blend = encode_fens_scored(
            bf, bs, br, lam=args.lam, scaling=args.scaling, workers=args.workers
        )
        parts += [blend] * args.repeat
        print("encoded fresh SF+Lc0 blend", flush=True)

    if args.disagreements:
        import csv

        df, ds, dr = [], [], []
        with open(args.disagreements) as f:
            for row in csv.DictReader(f):
                s = float(row["sf_cp"])
                l = float(row["lc0_cp"])
                blind = row["is_blindspot"].strip() == "True"
                w = [0.25, 0.75] if blind else [0.5, 0.5]
                cp = blend_cp([s, l], w)
                df.append(row["fen"])
                ds.append(cp)
                dr.append(0)
        print(f"depth-25 disagreement subset: {len(df):,}", flush=True)
        dis = encode_fens_scored(
            df, ds, dr, lam=1.0, scaling=args.scaling, workers=args.workers
        )
        parts += [dis] * args.disag_repeat
        print("encoded depth-25 disagreement blend", flush=True)

    mixed = concat(parts)
    save_encoded(mixed, args.out)
    print(
        f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, {len(mixed['bucket']):,} positions)",
        flush=True,
    )


if __name__ == "__main__":
    main()

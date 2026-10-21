import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
from eightchess_nnue.encoder import encode_fens_scored, save_encoded


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


def _p(cp, scaling=410.0):
    z = max(-60.0, min(60.0, cp / scaling))
    return 1.0 / (1.0 + math.exp(-z))


def blend_cp(cps, weights, scaling=410.0):
    p = sum(w * _p(cp, scaling) for w, cp in zip(weights, cps))
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return int(round(scaling * math.log(p / (1.0 - p))))


def encode_range(fens, scores, results, chunk, workers, lam, label):
    parts = []
    for start in range(0, len(fens), chunk):
        sl = slice(start, min(start + chunk, len(fens)))
        part = encode_fens_scored(
            fens[sl], scores[sl], results[sl], lam=lam, scaling=410.0, workers=workers
        )
        parts.append(part)
        print(f"{label}: {sl.stop:,}/{len(fens):,}", flush=True)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--bulk-sf", default=None)
    ap.add_argument("--sf", default=None)
    ap.add_argument("--disagreements", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--disag-repeat", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.8, dest="lam")
    ap.add_argument("--skip-bulk", action="store_true")
    ap.add_argument("--fresh-sf-weight", type=float, default=0.5)
    ap.add_argument("--disag-sf-weight", type=float, default=None)
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
    if args.limit:
        fens, lc0s, results = fens[: args.limit], lc0s[: args.limit], results[: args.limit]
    print(f"sample: {len(fens):,}", flush=True)

    parts = []
    if not args.skip_bulk:
        bulk_sf = {}
        if args.bulk_sf:
            with open(args.bulk_sf) as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) == 2:
                        bulk_sf[p[0]] = int(p[1])
            print(f"bulk stockfish labels: {len(bulk_sf):,}", flush=True)

        if bulk_sf:
            bulk_scores = [
                blend_cp([bulk_sf[fen], lc0], [0.8, 0.2]) if fen in bulk_sf else lc0
                for fen, lc0 in zip(fens, lc0s)
            ]
        else:
            bulk_scores = lc0s

        parts = encode_range(fens, bulk_scores, results, args.chunk, args.workers, args.lam, "bulk")

    sf = {}
    if args.sf:
        with open(args.sf) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 2:
                    sf[p[0]] = int(p[1])
        print(f"stockfish labels: {len(sf):,}", flush=True)

    if sf:
        bf, bs, br = [], [], []
        for fen, lc0, res in zip(fens, lc0s, results):
            s = sf.get(fen)
            if s is None:
                continue
            bf.append(fen)
            w_sf = args.fresh_sf_weight
            bs.append(s if w_sf >= 1.0 else blend_cp([s, lc0], [w_sf, 1.0 - w_sf]))
            br.append(res)
        print(f"fresh SF+Lc0 subset: {len(bf):,}", flush=True)
        fresh = encode_range(bf, bs, br, args.chunk, args.workers, args.lam, "fresh")
        parts += fresh * args.repeat

    if args.disagreements:
        df, ds, dr = [], [], []
        with open(args.disagreements) as f:
            for row in csv.DictReader(f):
                s = float(row["sf_cp"])
                l = float(row["lc0_cp"])
                blind = row["is_blindspot"].strip() == "True"
                if args.disag_sf_weight is not None:
                    w = [args.disag_sf_weight, 1.0 - args.disag_sf_weight]
                else:
                    w = [0.25, 0.75] if blind else [0.5, 0.5]
                df.append(row["fen"])
                ds.append(s if w[0] >= 1.0 else blend_cp([s, l], w))
                dr.append(0)
        print(f"depth-25 disagreement subset: {len(df):,}", flush=True)
        dis = encode_range(df, ds, dr, args.chunk, args.workers, 1.0, "disagreements")
        parts += dis * args.disag_repeat

    mixed = concat(parts)
    save_encoded(mixed, args.out)
    print(
        f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, {len(mixed['bucket']):,} positions)",
        flush=True,
    )


if __name__ == "__main__":
    main()

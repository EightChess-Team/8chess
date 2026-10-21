import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
from eightchess_nnue.encoder import encode_fens_scored, load_encoded, save_encoded


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


def read_tsv(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    out[p[0]] = int(float(p[1]))
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", default="data/hq_sf16_fens.txt")
    ap.add_argument("--sf", default="data/hq_sf25.tsv")
    ap.add_argument("--sf-existing", default="data/hq_sf25_existing.tsv")
    ap.add_argument("--lc0", default="data/hq_lc0.tsv")
    ap.add_argument("--disagreements", default="data/disagreements_gold.csv")
    ap.add_argument("--bulk", default="data/lc0_t80/sf_bulk_mixed.npz")
    ap.add_argument("--out", default="data/hq25/hq_train.npz")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--sf-weight", type=float, default=0.8)
    ap.add_argument("--disag-sf-weight", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    fens = []
    with open(args.fens) as f:
        for line in f:
            line = line.strip()
            if line:
                fens.append(line)
    sf = read_tsv(args.sf)
    if args.sf_existing and os.path.exists(args.sf_existing):
        sf.update(read_tsv(args.sf_existing))
    lc0 = read_tsv(args.lc0) if args.lc0 and os.path.exists(args.lc0) else {}
    print(f"fens={len(fens):,} sf={len(sf):,} lc0={len(lc0):,}", flush=True)

    hq_fens, hq_cps = [], []
    miss_sf = miss_lc0 = 0
    for fen in fens:
        s = sf.get(fen)
        l = lc0.get(fen)
        if s is None:
            miss_sf += 1
        if l is None:
            miss_lc0 += 1
        if s is None and l is None:
            continue
        if s is None:
            cp = l
        elif l is None:
            cp = s
        else:
            cp = blend_cp([s, l], [args.sf_weight, 1.0 - args.sf_weight])
        hq_fens.append(fen)
        hq_cps.append(cp)
    print(
        f"hq labelled={len(hq_fens):,} missing_sf={miss_sf:,} missing_lc0={miss_lc0:,}",
        flush=True,
    )

    hq = encode_fens_scored(
        hq_fens,
        hq_cps,
        [0] * len(hq_fens),
        lam=args.lam,
        scaling=410.0,
        workers=args.workers,
    )

    dfens, dcps = [], []
    with open(args.disagreements) as f:
        for row in csv.DictReader(f):
            s = float(row["sf_cp"])
            l = float(row["lc0_cp"])
            blind = row["is_blindspot"].strip() == "True"
            w = 0.25 if blind else args.disag_sf_weight
            cp = blend_cp([s, l], [w, 1.0 - w])
            dfens.append(row["fen"])
            dcps.append(int(max(-2000, min(2000, cp))))
    dis = encode_fens_scored(
        dfens, dcps, [0] * len(dfens), lam=args.lam, scaling=410.0, workers=args.workers
    )
    print(f"disagreements={len(dfens):,}", flush=True)

    hq_all = concat([dis, hq])
    parts = [hq_all] * args.repeat
    if args.bulk:
        bulk = load_encoded(args.bulk)
        print(f"bulk={len(bulk['bucket']):,}", flush=True)
        parts.append(bulk)
    data = concat(parts)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_encoded(data, args.out)
    t = data["target"]
    print(
        f"wrote {args.out}: {len(t):,} positions, mean={t.mean():.4f}, "
        f"q={np.percentile(t, [0, 25, 50, 75, 100]).round(3).tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

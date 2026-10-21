import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
from eightchess_nnue.encoder import encode_fens, encode_fens_scored, load_encoded, save_encoded


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
    ap.add_argument("--d9", default="data/sf_d9/train_368k.npz")
    ap.add_argument("--disagreements", default="data/disagreements_gold.csv")
    ap.add_argument("--endgames", default="data/endgame_losses.csv")
    ap.add_argument("--blunders", default="data/stockfish_blunders_60k.csv")
    ap.add_argument("--blunder-limit", type=int, default=30000)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", default="data/sf_d9/mixed_train.npz")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    d9 = load_encoded(args.d9)
    print(f"d9: {len(d9['bucket']):,}")

    dfens, dscores = [], []
    with open(args.disagreements) as f:
        for row in csv.DictReader(f):
            sf, lc0 = float(row["sf_cp"]), float(row["lc0_cp"])
            blind = row["is_blindspot"].strip() == "True"
            cp = (0.25 * sf + 0.75 * lc0) if blind else (0.5 * sf + 0.5 * lc0)
            dfens.append(row["fen"])
            dscores.append(int(max(-2000, min(2000, cp))))
    dis = encode_fens_scored(
        dfens, dscores, [0] * len(dfens), lam=1.0, scaling=410.0, workers=args.workers
    )
    print(f"disagreements: {len(dfens):,}")

    efens, eres = [], []
    with open(args.endgames) as f:
        for line in f:
            p = next(csv.reader([line]))
            if len(p) >= 2 and not line.startswith("fen"):
                efens.append(p[0].strip('"'))
                eres.append(p[1])
    end = encode_fens(efens, eres, workers=args.workers)
    print(f"endgames: {len(efens):,}")

    bfens, bres = [], []
    with open(args.blunders) as f:
        for line in f:
            if len(bfens) >= args.blunder_limit:
                break
            p = next(csv.reader([line]))
            if len(p) >= 2 and not line.startswith("fen"):
                bfens.append(p[0].strip('"'))
                bres.append(p[1])
    blu = encode_fens(bfens, bres, workers=args.workers)
    print(f"blunders: {len(bfens):,}")

    hq = concat([dis, end, blu])
    mixed = concat([hq] * args.repeat + [d9])
    print(f"mixed: {len(mixed['bucket']):,} (hq x{args.repeat} + d9)")
    save_encoded(mixed, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

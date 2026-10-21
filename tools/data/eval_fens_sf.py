import argparse
import multiprocessing as mp
import re
import subprocess
import time

_SF = None
_DEPTH = 18


def _send(cmd):
    _SF.stdin.write(cmd + "\n")
    _SF.stdin.flush()


def _wait(token):
    while True:
        line = _SF.stdout.readline()
        if not line or token in line:
            return


def _init(sf, depth, hash_mb):
    global _SF, _DEPTH
    _DEPTH = depth
    _SF = subprocess.Popen(
        [sf],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    _send("uci")
    _wait("uciok")
    _send("setoption name Threads value 1")
    _send(f"setoption name Hash value {hash_mb}")
    _send("isready")
    _wait("readyok")


def _eval(fen):
    _send(f"position fen {fen}")
    _send(f"go depth {_DEPTH}")
    cp = 0
    mate = None
    while True:
        line = _SF.stdout.readline()
        if not line:
            break
        if line.startswith("bestmove"):
            break
        m = re.search(r"\bscore cp (-?\d+)", line)
        if m:
            cp = int(m.group(1))
            mate = None
        m = re.search(r"\bscore mate (-?\d+)", line)
        if m:
            mate = int(m.group(1))
    if mate is not None:
        cp = 2000 if mate >= 0 else -2000
    return fen, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sf", default="/tmp/opencode/sf-tools/src/stockfish")
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    fens = []
    with open(args.fens) as f:
        for line in f:
            line = line.strip()
            if line:
                fens.append(line)
    if args.limit:
        fens = fens[: args.limit]
    print(f"evaluating {len(fens):,} positions at depth {args.depth}", flush=True)

    t0 = time.time()
    done = 0
    with open(args.out, "w") as out, mp.Pool(
        args.workers, initializer=_init, initargs=(args.sf, args.depth, args.hash)
    ) as pool:
        for fen, cp in pool.imap_unordered(_eval, fens, chunksize=64):
            out.write(f"{fen}\t{cp}\n")
            done += 1
            if done % 5000 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"{done:,}/{len(fens):,} ({rate:,.0f}/s)", flush=True)
    print(f"done: {done:,} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

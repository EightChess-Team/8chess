"""Diff Python accumulators against the engine's `dumpacc` output."""
import subprocess
import sys

import chess
import numpy as np

sys.path.insert(0, ".")
from eightchess_nnue import format as nnf
from eightchess_nnue.features import piece_feature_indices, threat_feature_indices
from eightchess_nnue.quantized import _accumulate

ENGINE = "../target/release/eightchess"
NET = "../networks/v60-7f587dfb.nnue"

FENS = [
    "8/8/8/8/8/8/P7/K6k w - - 0 1",
    "2b2br1/r2nk3/p2p2pn/Ppp1P2p/P1P1p2P/NQ4PN/1B1P1K2/R4B1R w - - 1 26",
]

proc = subprocess.Popen(
    [ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
)


def send(line):
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


net = nnf.read_network(NET)

for fen in FENS:
    send(f"position fen {fen}")
    send("dumpacc white")
    pst_engine = None
    thr_engine = None
    while pst_engine is None or thr_engine is None:
        line = proc.stdout.readline().strip()
        if line.startswith("pst "):
            pst_engine = np.array([int(v) for v in line[4:].split(",")], dtype=np.int64)
        elif line.startswith("threat "):
            thr_engine = np.array([int(v) for v in line[7:].split(",")], dtype=np.int64)

    board = chess.Board(fen)
    pf = piece_feature_indices(board)
    tf = threat_feature_indices(board)

    pst_py = _accumulate(net["ft_piece_weights"], pf[0], net["ft_biases"], np.int32).astype(np.int64)
    thr_py = _accumulate(net["ft_threat_weights"], tf[0], np.zeros(768, dtype=np.int64), np.int32).astype(np.int64)

    pst_diff = np.nonzero(pst_py != pst_engine)[0]
    thr_diff = np.nonzero(thr_py != thr_engine)[0]
    print(f"FEN: {fen}")
    print(f"  piece features: {len(pf[0])}, threat features: {len(tf[0])}")
    print(f"  pst diffs: {len(pst_diff)} / 768   threat diffs: {len(thr_diff)} / 768")
    if len(pst_diff):
        i = pst_diff[0]
        print(f"  first pst diff at {i}: engine={pst_engine[i]} python={pst_py[i]} d={pst_engine[i]-pst_py[i]}")
    if len(thr_diff):
        i = thr_diff[0]
        print(f"  first thr diff at {i}: engine={thr_engine[i]} python={thr_py[i]} d={thr_engine[i]-thr_py[i]}")

send("quit")
proc.wait(timeout=30)

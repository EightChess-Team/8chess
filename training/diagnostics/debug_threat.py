"""Zoom in on the threat feature mismatch for the minimal position."""
import subprocess
import sys

import chess
import numpy as np

sys.path.insert(0, ".")
from eightchess_nnue import format as nnf
from eightchess_nnue.features import threat_feature_indices, threat_index
from eightchess_nnue.quantized import _accumulate

ENGINE = "../target/release/eightchess"
NET = "../networks/v60-7f587dfb.nnue"

FEN = "8/8/8/8/8/8/P7/K6k w - - 0 1"
board = chess.Board(FEN)

proc = subprocess.Popen([ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def send(line):
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


send(f"position fen {FEN}")
send("dumpacc white")
pst_engine = thr_engine = None
while pst_engine is None or thr_engine is None:
    line = proc.stdout.readline().strip()
    if line.startswith("pst "):
        pst_engine = np.array([int(v) for v in line[4:].split(",")], dtype=np.int64)
    elif line.startswith("threat "):
        thr_engine = np.array([int(v) for v in line[7:].split(",")], dtype=np.int64)

net = nnf.read_network(NET)
feats = threat_feature_indices(board)[0]
print("python threat features:", feats)

for idx in feats:
    row = net["ft_threat_weights"][idx].astype(np.int64)
    print(f"  feature {idx}: python-accumulated row[:8]={row[:8]}")
    print(f"  engine accumulator[:8]     ={thr_engine[:8]}")

# Find which single feature row reproduces the engine's accumulator (brute test
# over king-square attacks on own pawn for all possible feature indices).
target = thr_engine
matches = []
for idx in range(66864):
    row = net["ft_threat_weights"][idx].astype(np.int64)
    if np.array_equal(row, target):
        matches.append(idx)
print("feature indices whose weight row equals engine accumulator:", matches)

send("quit")
proc.wait(timeout=30)

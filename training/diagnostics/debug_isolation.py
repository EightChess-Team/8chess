"""Isolate replica mismatches by zeroing parts of the network."""
import sys
import chess
import numpy as np

sys.path.insert(0, ".")
from eightchess_nnue import format as nnf
from eightchess_nnue.quantized import evaluate
from eightchess_nnue.verify_engine import engine_evals, random_fens

ENGINE = "../target/release/eightchess"
NET = "../networks/v60-7f587dfb.nnue"

orig = nnf.read_network(NET)
fens = random_fens(30, seed=7)

variants = {}

z = nnf.zero_network()
z["ft_biases"] = orig["ft_biases"].copy()
variants["biases_only"] = z

z = nnf.zero_network()
z["ft_biases"] = orig["ft_biases"].copy()
z["ft_piece_weights"] = orig["ft_piece_weights"].copy()
variants["piece_only"] = z

z = nnf.zero_network()
z["ft_biases"] = orig["ft_biases"].copy()
z["ft_threat_weights"] = orig["ft_threat_weights"].copy()
variants["threat_only"] = z

variants["full"] = {k: v.copy() for k, v in orig.items()}

for name, net in variants.items():
    path = f"/tmp/opencode/isolation_{name}.nnue"
    nnf.write_network(net, path)
    eng = engine_evals(ENGINE, fens, eval_file=path)
    bad = 0
    maxd = 0
    for fen, e in zip(fens, eng):
        m = evaluate(net, chess.Board(fen))
        maxd = max(maxd, abs(m - e))
        if m != e:
            bad += 1
    print(f"{name:12s}: mismatches {bad}/{len(fens)}  max|diff| {maxd}")
    if bad and bad < 5:
        for fen, e in zip(fens, eng):
            m = evaluate(net, chess.Board(fen))
            if m != e:
                print(f"   {fen}  engine={e} python={m}")

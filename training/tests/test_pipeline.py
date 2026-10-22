"""Smoke test for the entire NNUE pipeline:
dataset build -> model training -> export -> engine eval verification.
"""

import os
import random
import subprocess
import sys
import tempfile

# Ensure training/ is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.pgn

def make_sample_pgn(path: str, count: int = 15):
    rng = random.Random(42)
    with open(path, "w") as f:
        for i in range(count):
            game = chess.pgn.Game()
            game.headers["Event"] = f"Test Game {i}"
            board = game.board()
            node = game
            result = rng.choice(["1-0", "0-1", "1/2-1/2"])
            game.headers["Result"] = result

            for ply in range(rng.randint(20, 60)):
                moves = list(board.legal_moves)
                if not moves or board.is_game_over():
                    break
                mv = rng.choice(moves)
                board.push(mv)
                node = node.add_variation(mv)
            print(game, file=f, end="\n\n")

def main():
    print("=== 8Chess NNUE End-to-End Pipeline Smoke Test ===")
    workdir = tempfile.mkdtemp(prefix="8chess_smoke_")
    print(f"Working in: {workdir}")

    pgn_path = os.path.join(workdir, "games.pgn")
    npz_path = os.path.join(workdir, "dataset.npz")
    ckpt_dir = os.path.join(workdir, "checkpoints")

    print("\n1. Generating synthetic PGN games...")
    make_sample_pgn(pgn_path, count=25)

    print("\n2. Building packed dataset via build_dataset...")
    cmd_build = [
        sys.executable, "-m", "eightchess_nnue.build_dataset",
        "--pgn", pgn_path,
        "--out", npz_path,
        "--stride", "2",
        "--min-ply", "4",
        "--workers", "4",
        "--chunk", "512",
    ]
    subprocess.check_call(cmd_build, cwd="training")

    print("\n3. Training model for 2 epochs via train.py...")
    cmd_train = [
        sys.executable, "-m", "eightchess_nnue.train",
        "--data", npz_path,
        "--out-dir", ckpt_dir,
        "--epochs", "2",
        "--batch-size", "64",
        "--lr", "0.001",
        "--export-every", "1",
        "--device", "cpu",
    ]
    subprocess.check_call(cmd_train, cwd="training")

    exported_nnue = os.path.join(ckpt_dir, "epoch_2.nnue")
    assert os.path.exists(exported_nnue), f"Missing exported nnue: {exported_nnue}"
    assert os.path.getsize(exported_nnue) == 84349504, f"Incorrect size: {os.path.getsize(exported_nnue)}"
    print(f"\n4. Exported network verified: {exported_nnue} (84,349,504 bytes)")

    print("\n5. Verifying engine compatibility and bit-exact evaluation on the exported network...")
    engine_bin = os.path.abspath("target/release/eightchess")
    cmd_verify = [
        sys.executable, "-m", "eightchess_nnue.verify_engine",
        "--engine", engine_bin,
        "--network", exported_nnue,
        "--count", "100",
    ]
    # We pass the eval file to engine_evals
    # Note: verify_engine currently expects the engine default net, unless eval_file is passed.
    # Let's run a customized verification with EvalFile option set!
    from eightchess_nnue.verify_engine import engine_evals, random_fens
    from eightchess_nnue import format as nnf
    from eightchess_nnue.quantized import evaluate

    fens = random_fens(100, seed=999)
    print("Running engine_evals with EvalFile set to the new network...")
    eng_scores = engine_evals(engine_bin, fens, eval_file=exported_nnue)

    net = nnf.read_network(exported_nnue)
    exact = 0
    max_diff = 0
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        py_eval = evaluate(net, b)
        eng_eval = eng_scores[i]
        diff = abs(py_eval - eng_eval)
        max_diff = max(max_diff, diff)
        if diff == 0:
            exact += 1
        else:
            print(f"Mismatch at {fen}: py={py_eval}, eng={eng_eval}")

    print(f"\nExact match rate: {exact}/{len(fens)} ({100.0 * exact / len(fens):.2f}%)")
    print(f"Max |diff|: {max_diff}")
    assert exact == len(fens), f"Exact match failed! {exact}/{len(fens)}"

    print("\n=== SMOKE TEST PASSED PERFECTLY ===")

if __name__ == "__main__":
    main()

"""Export a trained torch model to the engine's binary NNUE format.

Quantization (see docs/nnue_format.md):
    ft_piece_weights : round(w * 255) clamped to i16
    ft_threat_weights: round(w * 255) clamped to i8
    ft_biases        : round(b * 255) clamped to i16
    l1_weights       : round(w * 64) clamped to i8
    all other layers : f32 verbatim

Usage:
    python -m eightchess_nnue.export --checkpoint ckpt.pt --out net.nnue \
        [--engine ../../target/release/eightchess --verify-count 200]
"""

from __future__ import annotations

import argparse
import sys

import chess
import numpy as np
import torch

from . import format as nnue_format
from .constants import FT_QUANT, L1_QUANT
from .model import EightChessNNUE
from .quantized import evaluate


def quantize_weights(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    def arr(name: str) -> np.ndarray:
        return state[name].detach().cpu().numpy()

    l1_raw = np.clip(np.round(arr("l1_weight") * L1_QUANT), -127, 127).astype(np.int8)
    l1_serialized = l1_raw.reshape(8, 16, 192, 4).transpose(0, 2, 1, 3).reshape(8, 12288)

    return {
        "ft_piece_weights": np.clip(np.round(arr("ft_piece.weight") * FT_QUANT), -32767, 32767).astype(np.int16),
        "ft_threat_weights": np.clip(np.round(arr("ft_threat.weight") * FT_QUANT), -127, 127).astype(np.int8),
        "ft_biases": np.clip(np.round(arr("ft_bias") * FT_QUANT), -32767, 32767).astype(np.int16),
        "l1_weights": l1_serialized,
        "l1_biases": arr("l1_bias").astype(np.float32),
        "l2_weights": arr("l2_weight").astype(np.float32),
        "l2_biases": arr("l2_bias").astype(np.float32),
        "l3_weights": arr("l3_weight").astype(np.float32),
        "l3_biases": arr("l3_bias").astype(np.float32),
    }


def load_checkpoint(path: str) -> EightChessNNUE:
    model = EightChessNNUE()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine", default=None, help="also verify against this engine binary")
    parser.add_argument("--verify-count", type=int, default=200)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    net = quantize_weights(model.state_dict())
    nnue_format.write_network(net, args.out)
    print(f"wrote {args.out} ({nnue_format.NETWORK_SIZE} bytes)")

    from .verify_engine import engine_evals, random_fens
    fens = random_fens(args.verify_count, seed=1234)
    engine_scores = None
    if args.engine:
        engine_scores = engine_evals(args.engine, fens, eval_file=args.out)

    exact = 0
    max_diff = 0
    for i, fen in enumerate(fens):
        mine = evaluate(net, chess.Board(fen))
        if engine_scores is not None:
            if mine == engine_scores[i]:
                exact += 1
            else:
                max_diff = max(max_diff, abs(mine - engine_scores[i]))
    if engine_scores is not None:
        print(f"engine exact match: {exact}/{len(fens)} (max |diff| {max_diff})")
        if exact < len(fens):
            print("ERROR: exported network does not match engine semantics", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Training loop for 8Chess NNUE.

Reads packed .npz datasets produced by `build_dataset.py`, trains the
quantization-aware `EightChessNNUE` with BCE loss on WDL outcomes,
and exports engine-compatible .nnue binaries periodically.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import format as nnue_format
from .encoder import load_encoded
from .export import quantize_weights
from .model import EightChessNNUE, wdl_from_cp


class EncodedDataset:
    def __init__(self, data: dict[str, np.ndarray], device: torch.device):
        self.device = device
        self.size = int(data["bucket"].shape[0])
        self.bucket = data["bucket"]
        self.target = data["target"]

        self.streams = {}
        for s in ("piece_stm", "threat_stm", "piece_opp", "threat_opp"):
            self.streams[f"{s}_idx"] = data[f"{s}_indices"]
            self.streams[f"{s}_off"] = data[f"{s}_offsets"]

    def get_batch(self, batch_indices: np.ndarray):
        """Constructs batched embedding bag inputs for the given sample indices."""
        k = len(batch_indices)
        batch_tensors = {}

        for s in ("piece_stm", "threat_stm", "piece_opp", "threat_opp"):
            flat_idx = self.streams[f"{s}_idx"]
            offsets = self.streams[f"{s}_off"]

            starts = offsets[batch_indices]
            ends = offsets[batch_indices + 1]
            lengths = ends - starts

            total_feats = int(np.sum(lengths))
            bag_indices = np.empty(total_feats, dtype=np.int64)
            bag_offsets = np.zeros(k, dtype=np.int64)

            pos = 0
            for i, (st, en) in enumerate(zip(starts, ends)):
                bag_offsets[i] = pos
                length = en - st
                bag_indices[pos : pos + length] = flat_idx[st:en]
                pos += length

            batch_tensors[f"{s}_indices"] = torch.from_numpy(bag_indices).to(self.device, non_blocking=True)
            batch_tensors[f"{s}_offsets"] = torch.from_numpy(bag_offsets).to(self.device, non_blocking=True)

        batch_tensors["buckets"] = torch.from_numpy(self.bucket[batch_indices].astype(np.int64)).to(
            self.device, non_blocking=True
        )
        batch_tensors["targets"] = torch.from_numpy(self.target[batch_indices]).to(
            self.device, non_blocking=True
        )
        return batch_tensors


def evaluate_dataset(model: EightChessNNUE, dataset: EncodedDataset, batch_size: int, temperature: float):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    decisive_samples = 0

    indices = np.arange(dataset.size)
    with torch.no_grad():
        for start in range(0, dataset.size, batch_size):
            end = min(start + batch_size, dataset.size)
            batch = dataset.get_batch(indices[start:end])
            out = model(
                batch["piece_stm_indices"],
                batch["piece_stm_offsets"],
                batch["threat_stm_indices"],
                batch["threat_stm_offsets"],
                batch["piece_opp_indices"],
                batch["piece_opp_offsets"],
                batch["threat_opp_indices"],
                batch["threat_opp_offsets"],
                batch["buckets"],
            )
            score_cp = out * 380.0
            pred = wdl_from_cp(score_cp, temperature)
            loss = F.binary_cross_entropy(pred, batch["targets"])

            n = end - start
            total_loss += float(loss.item()) * n
            total_samples += n

            # Accuracy on decisive games (> 0.5 vs < 0.5)
            t = batch["targets"]
            decisive_mask = t != 0.5
            if decisive_mask.any():
                pred_win = pred[decisive_mask] > 0.5
                actual_win = t[decisive_mask] > 0.5
                correct += int((pred_win == actual_win).sum().item())
                decisive_samples += int(decisive_mask.sum().item())

    mean_loss = total_loss / max(total_samples, 1)
    acc = correct / max(decisive_samples, 1)
    return mean_loss, acc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to train .npz")
    parser.add_argument("--base", default=None, help="base .nnue to fine-tune from")
    parser.add_argument("--val", default=None, help="path to val .npz")
    parser.add_argument("--out-dir", default="checkpoints", help="output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ft-lr-scale", type=float, default=None, help="LR multiplier for FT parameters")
    parser.add_argument("--freeze-base-ft", action="store_true", help="freeze original 768 FT channels (0..384, 512..896)")
    parser.add_argument("--temperature", type=float, default=173.72)
    parser.add_argument("--export-every", type=int, default=1, help="export .nnue every N epochs")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading training data: {args.data}")
    train_raw = load_encoded(args.data)
    train_dataset = EncodedDataset(train_raw, device)
    print(f"Loaded {train_dataset.size} training positions")

    val_dataset = None
    if args.val:
        print(f"Loading validation data: {args.val}")
        val_raw = load_encoded(args.val)
        val_dataset = EncodedDataset(val_raw, device)
        print(f"Loaded {val_dataset.size} validation positions")

    if args.base:
        print(f"Loading base network: {args.base}")
        model = EightChessNNUE.load_network_file(args.base).to(device)
    else:
        model = EightChessNNUE().to(device)

    ft_lr_scale = args.ft_lr_scale if args.ft_lr_scale is not None else (0.1 if args.base else 1.0)
    ft_params = [model.ft_piece.weight, model.ft_threat.weight, model.ft_bias]
    head_params = [model.l1_weight, model.l1_bias, model.l2_weight, model.l2_bias, model.l3_weight, model.l3_bias]
    optimizer = AdamW(
        [
            {"params": ft_params, "lr": args.lr * ft_lr_scale, "weight_decay": 0.0},
            {"params": head_params, "lr": args.lr, "weight_decay": 1e-5},
        ]
    )
    total_steps = (train_dataset.size // args.batch_size + 1) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    print(f"Starting training: {args.epochs} epochs, batch size {args.batch_size}, {total_steps} steps total")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(train_dataset.size)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, train_dataset.size, args.batch_size):
            end = min(start + args.batch_size, train_dataset.size)
            batch_idxs = perm[start:end]
            batch = train_dataset.get_batch(batch_idxs)

            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["piece_stm_indices"],
                batch["piece_stm_offsets"],
                batch["threat_stm_indices"],
                batch["threat_stm_offsets"],
                batch["piece_opp_indices"],
                batch["piece_opp_offsets"],
                batch["threat_opp_indices"],
                batch["threat_opp_offsets"],
                batch["buckets"],
            )

            score_cp = out * 380.0
            pred = wdl_from_cp(score_cp, args.temperature)
            loss = F.binary_cross_entropy(pred, batch["targets"])

            loss.backward()
            if args.freeze_base_ft:
                with torch.no_grad():
                    for p in (model.ft_piece.weight, model.ft_threat.weight):
                        if p.grad is not None:
                            p.grad[:, :384] = 0.0
                            p.grad[:, 512:896] = 0.0
                    if model.ft_bias.grad is not None:
                        model.ft_bias.grad[:384] = 0.0
                        model.ft_bias.grad[512:896] = 0.0
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            batches += 1

        elapsed = time.time() - t0
        avg_train_loss = epoch_loss / max(batches, 1)

        val_info = ""
        if val_dataset:
            val_loss, val_acc = evaluate_dataset(model, val_dataset, args.batch_size, args.temperature)
            val_info = f" | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.1f}%"

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | train_loss: {avg_train_loss:.4f}{val_info} | {elapsed:.1f}s ({train_dataset.size/elapsed:.0f} pos/s)"
        )

        if epoch % args.export_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )

            nnue_path = os.path.join(args.out_dir, f"epoch_{epoch}.nnue")
            quantized = quantize_weights(model.state_dict())
            nnue_format.write_network(quantized, nnue_path)
            print(f"  -> Exported {nnue_path}")

    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

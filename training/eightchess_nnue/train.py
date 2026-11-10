"""Training loop for 8Chess NNUE.

Reads packed .npz datasets (or directories / globs of shards) produced by
`build_shards.py`, trains the quantization-aware `EightChessNNUE` with BCE
loss on WDL targets, and exports engine-compatible .nnue binaries.

Shards are streamed one at a time so datasets larger than RAM are supported.
"""

from __future__ import annotations

import argparse
import gc
import glob
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
            lengths = (ends - starts).astype(np.int64)

            # Vectorized concatenation of ragged slices: for every output slot,
            # map back to the source position in flat_idx.
            bag_offsets = np.zeros(k, dtype=np.int64)
            np.cumsum(lengths[:-1], out=bag_offsets[1:])
            total_feats = int(lengths.sum())
            gather = (
                np.arange(total_feats, dtype=np.int64)
                - np.repeat(bag_offsets, lengths)
                + np.repeat(starts, lengths)
            )
            bag_indices = flat_idx[gather]

            batch_tensors[f"{s}_indices"] = torch.from_numpy(bag_indices).to(self.device, non_blocking=True)
            batch_tensors[f"{s}_offsets"] = torch.from_numpy(bag_offsets).to(self.device, non_blocking=True)

        batch_tensors["buckets"] = torch.from_numpy(self.bucket[batch_indices].astype(np.int64)).to(
            self.device, non_blocking=True
        )
        batch_tensors["targets"] = torch.from_numpy(self.target[batch_indices]).to(
            self.device, non_blocking=True
        )
        return batch_tensors


def resolve_shards(specs: list[str]) -> list[str]:
    """Expands directories and globs into a deduplicated list of .npz paths."""
    paths: list[str] = []
    for spec in specs:
        if os.path.isdir(spec):
            paths.extend(sorted(glob.glob(os.path.join(spec, "*.npz"))))
        elif any(c in spec for c in "*?["):
            paths.extend(sorted(glob.glob(spec)))
        else:
            paths.append(spec)
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def shard_positions(path: str) -> int:
    with np.load(path) as f:
        return int(f["bucket"].shape[0])


def evaluate_shards(
    model: EightChessNNUE,
    paths: list[str],
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    decisive_samples = 0

    with torch.no_grad():
        for path in paths:
            raw = load_encoded(path)
            dataset = EncodedDataset(raw, device)
            for start in range(0, dataset.size, batch_size):
                end = min(start + batch_size, dataset.size)
                batch = dataset.get_batch(np.arange(start, end))
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

                t = batch["targets"]
                decisive_mask = t != 0.5
                if decisive_mask.any():
                    pred_win = pred[decisive_mask] > 0.5
                    actual_win = t[decisive_mask] > 0.5
                    correct += int((pred_win == actual_win).sum().item())
                    decisive_samples += int(decisive_mask.sum().item())
            del raw, dataset
            gc.collect()

    mean_loss = total_loss / max(total_samples, 1)
    acc = correct / max(decisive_samples, 1)
    return mean_loss, acc


def export_nnue(model: EightChessNNUE, path: str) -> None:
    quantized = quantize_weights(model.state_dict())
    nnue_format.write_network(quantized, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, nargs="+", help="npz shard(s), glob(s) or dir(s)")
    parser.add_argument("--base", default=None, help="base .nnue to fine-tune from")
    parser.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    parser.add_argument("--val", default=None, nargs="+", help="validation shard(s), glob(s) or dir(s)")
    parser.add_argument("--val-frac", type=float, default=0.0,
                        help="hold out every 1/val_frac-th shard as validation (e.g. 0.02 = 2%%)")
    parser.add_argument("--out-dir", default="checkpoints", help="output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ft-lr-scale", type=float, default=None, help="LR multiplier for FT parameters")
    parser.add_argument("--freeze-base-ft", action="store_true", help="freeze original 768 FT channels (0..384, 512..896)")
    parser.add_argument("--freeze-backbone", action="store_true", help="freeze entire 768 backbone (FT_old, L1_old, L2, L3) and train only the 256 new residual channels")
    parser.add_argument("--temperature", type=float, default=173.72)
    parser.add_argument("--export-every", type=int, default=1, help="export .nnue every N epochs")
    parser.add_argument("--save-every-shards", type=int, default=0, help="checkpoint/export every N shards (0 = only per epoch)")
    parser.add_argument("--max-shards", type=int, default=0, help="stop after this many shards per epoch (0 = all)")
    parser.add_argument("--max-seconds", "--time-limit", type=float, default=0, dest="max_seconds",
                        help="stop after this many seconds (0 = no limit)")
    parser.add_argument("--revive-new-channels", action="store_true",
                        help="random-init the dormant 256 residual channels before fine-tuning")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    train_shards = resolve_shards(args.data)
    if not train_shards:
        print("error: no training shards found", file=sys.stderr)
        return 1
    print(f"Counting {len(train_shards)} training shards...")
    sizes = [shard_positions(p) for p in train_shards]
    total_train = int(sum(sizes))
    print(f"{len(train_shards)} shards, {total_train:,} positions")

    val_shards: list[str] = []
    if args.val:
        val_shards = resolve_shards(args.val)
        print(f"{len(val_shards)} validation shards")
    elif args.val_frac and args.val_frac > 0:
        step = max(2, int(round(1.0 / args.val_frac)))
        val_shards = train_shards[::step]
        train_shards = [p for i, p in enumerate(train_shards) if i % step != 0]
        total_train = int(sum(shard_positions(p) for p in train_shards))
        print(f"held out {len(val_shards)} validation shards; {total_train:,} training positions")

    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model = EightChessNNUE().to(device)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    elif args.base:
        print(f"Loading base network: {args.base}")
        model = EightChessNNUE.load_network_file(args.base).to(device)
    else:
        model = EightChessNNUE().to(device)

    raw_model = model
    if args.base and args.revive_new_channels:
        with torch.no_grad():
            g = torch.Generator(device=device).manual_seed(1234)
            for sl in (slice(384, 512), slice(896, 1024)):
                # weight tensors are [num_features, L1_SIZE]: revive columns, not rows.
                raw_model.ft_piece.weight[:, sl].uniform_(-0.05, 0.05, generator=g)
                raw_model.ft_threat.weight[:, sl].uniform_(-0.02, 0.02, generator=g)
                raw_model.ft_bias[sl].uniform_(-0.02, 0.02, generator=g)
                raw_model.l1_weight[:, :, sl].uniform_(-0.02, 0.02, generator=g)
        print("revived 256 dormant FT channels (two 128-wide residual slices)", flush=True)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel", flush=True)
    else:
        print(f"Using single device: {device}", flush=True)

    ft_lr_scale = args.ft_lr_scale if args.ft_lr_scale is not None else (0.1 if (args.base or args.resume) else 1.0)
    ft_params = [raw_model.ft_piece.weight, raw_model.ft_threat.weight, raw_model.ft_bias]
    if args.freeze_backbone:
        for p in (raw_model.l1_bias, raw_model.l2_weight, raw_model.l2_bias, raw_model.l3_weight, raw_model.l3_bias):
            p.requires_grad_(False)
        head_params = [raw_model.l1_weight]
        head_wd = 0.0
    else:
        head_params = [raw_model.l1_weight, raw_model.l1_bias, raw_model.l2_weight, raw_model.l2_bias, raw_model.l3_weight, raw_model.l3_bias]
        head_wd = 1e-5
    optimizer = AdamW(
        [
            {"params": ft_params, "lr": args.lr * ft_lr_scale, "weight_decay": 0.0},
            {"params": head_params, "lr": args.lr, "weight_decay": head_wd},
        ]
    )
    steps_per_epoch = (total_train // args.batch_size + 1) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch, eta_min=1e-6)
    if args.resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if args.resume and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    run_start = time.time()
    stop_requested = False

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        batches = 0
        seen = 0
        shards_done = 0

        order = np.random.permutation(len(train_shards))
        if args.max_shards:
            order = order[: args.max_shards]

        for si in order:
            path = train_shards[int(si)]
            raw = load_encoded(path)
            dataset = EncodedDataset(raw, device)
            perm = np.random.permutation(dataset.size)

            for start in range(0, dataset.size, args.batch_size):
                end = min(start + args.batch_size, dataset.size)
                batch_idxs = perm[start:end]
                batch = dataset.get_batch(batch_idxs)

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
                if args.freeze_base_ft or args.freeze_backbone:
                    with torch.no_grad():
                        for p in (raw_model.ft_piece.weight, raw_model.ft_threat.weight):
                            if p.grad is not None:
                                p.grad[:, :384] = 0.0
                                p.grad[:, 512:896] = 0.0
                        if raw_model.ft_bias.grad is not None:
                            raw_model.ft_bias.grad[:384] = 0.0
                            raw_model.ft_bias.grad[512:896] = 0.0
                if args.freeze_backbone:
                    with torch.no_grad():
                        if raw_model.l1_weight.grad is not None:
                            raw_model.l1_weight.grad[:, :, :384] = 0.0
                            raw_model.l1_weight.grad[:, :, 512:896] = 0.0
                        for p in (raw_model.l1_bias, raw_model.l2_weight, raw_model.l2_bias, raw_model.l3_weight, raw_model.l3_bias):
                            if p.grad is not None:
                                p.grad.zero_()
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                batches += 1

            seen += dataset.size
            shards_done += 1
            del raw, dataset
            gc.collect()

            if args.save_every_shards and shards_done % args.save_every_shards == 0:
                ckpt_path = os.path.join(args.out_dir, "latest.pt")
                torch.save({"epoch": epoch, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, ckpt_path)
                nnue_path = os.path.join(args.out_dir, "latest.nnue")
                export_nnue(raw_model, nnue_path)
                elapsed = time.time() - t0
                print(
                    f"  checkpoint @ {shards_done} shards ({seen:,} pos, {seen/max(elapsed,1e-9):.0f} pos/s) -> {nnue_path}",
                    flush=True,
                )

            if args.max_seconds and time.time() - run_start >= args.max_seconds:
                print("time limit reached; saving and stopping", flush=True)
                stop_requested = True
                break

        elapsed = time.time() - t0
        avg_train_loss = epoch_loss / max(batches, 1)

        val_info = ""
        if val_shards:
            val_loss, val_acc = evaluate_shards(model, val_shards, device, args.batch_size, args.temperature)
            val_info = f" | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.1f}%"

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | train_loss: {avg_train_loss:.4f}{val_info} | "
            f"{elapsed:.1f}s ({seen/max(elapsed,1e-9):.0f} pos/s)",
            flush=True,
        )

        if epoch % args.export_every == 0 or epoch == args.epochs or stop_requested:
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
            torch.save({"epoch": epoch, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, ckpt_path)
            nnue_path = os.path.join(args.out_dir, f"epoch_{epoch}.nnue")
            export_nnue(raw_model, nnue_path)
            print(f"  -> Exported {nnue_path}", flush=True)

        if stop_requested:
            break

    print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

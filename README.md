# 8Chess

8Chess is an experimental 1024-wide NNUE chess engine derived from
[Reckless](https://github.com/codedeliveryservice/Reckless). It remains a
single self-contained Rust binary: the network is embedded at build time and
there are no runtime model files.

## Status

- 1024-wide threat + piece NNUE, 10 input buckets / 8 output buckets.
- Current embedded network: `eightchess_1024_tuned768.nnue`.
- Direct match record vs Stockfish 19 (no opening book, `1.0+0.01`):
  - 1,000 games: 67.75% (+487 =381 -132, +129 +/- 17 Elo)
- Goal: continue improving the engine against Stockfish 19 in long no-book matches.

## Build

```sh
cargo build --release
```

`build/build.rs` downloads the embedded network from the network release if
`networks/eightchess_1024_tuned768.nnue` is missing. Override with `EVALFILE` or
place a compatible `.nnue` in `networks/`.

The resulting executable is `target/release/eightchess`. It supports the
`EvalFile` UCI option, so an external network can be tested without rebuilding:

```sh
setoption name EvalFile value /path/to/network.nnue
```

## CpuPlay: play online, powered by donors

CpuPlay is the open-source web platform for 8Chess: play the engine in
your browser while volunteer donor CPUs (signed-in, account-gated)
compute the moves. Includes live donation stats, worker/device
authorization, and volunteer-powered game review.

- Repo: https://github.com/EightChess-Team/cpuplay

## Training and data

The training code lives in `training/eightchess_nnue/` (PyTorch). It encodes
positions into the sparse piece/threat feature format, trains soft-target WDL
heads, and exports the exact binary layout consumed by `src/nnue.rs`.

Data and checkpoints are intentionally not committed here. The official
artifacts are published in the **EightChess-Team** organisation on Hugging
Face (same organisation name as GitHub):

- [EightChess-Team/eightchess-networks](https://huggingface.co/EightChess-Team/eightchess-networks)
  - all trained NNUE networks (15 unique files, ~1.06 GiB), including the
  champion `eightchess_1024_tuned768.nnue` embedded in v1.0.10
- [EightChess-Team/eightchess-data](https://huggingface.co/datasets/EightChess-Team/eightchess-data)
  - training data, currently `gensfen.binpack` (sfbinpack format)

Each repo ships a `MANIFEST.txt` with the SHA-256 and size of every file.

Local-only paths:

- `data/` - FEN sets, SF/Lc0 labels, encoded `.npz` datasets
- `checkpoints/` - training checkpoints (`.nnue`, `.pt`)
- `networks/` - local network copies, including Lc0 nets
- `results/` - PGN match records

Useful entry points:

```sh
python3 tools/data/eval_fens_sf.py --fens data/fens.txt --out data/sf.tsv \
  --sf /path/to/stockfish --depth 25 --workers 11
python3 tools/data/build_hq25.py --help
PYTHONPATH=training python3 -m eightchess_nnue.train --help
PYTHONPATH=training python3 -m eightchess_nnue.verify_engine \
  --engine target/release/eightchess --network networks/eightchess_1024_tuned768.nnue --count 100
python3 tools/matches/match_runner.py --help
```

## License

AGPL-3.0, inherited from Reckless. See [LICENSE](LICENSE) and the upstream
project for the original engine and network tooling.

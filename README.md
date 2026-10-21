# 8Chess

8Chess is an experimental 1024-wide NNUE chess engine derived from
[Reckless](https://github.com/codedeliveryservice/Reckless). It remains a
single self-contained Rust binary: the network is embedded at build time and
there are no runtime model files.

## Status

- 1024-wide threat + piece NNUE, 10 input buckets / 8 output buckets.
- Current embedded network: `eightchess_1024_master.nnue`, a fine-tune on
  968,660 SF/Lc0-labelled positions.
- Match record vs Stockfish 19 (no opening book, `1.0+0.01`):
  - 100 games: 51.0%
  - 500 games: 47.3% (+57 =359 -84, -18.8 +/- 16.2 Elo)
- Goal: beat Stockfish 19 in a 60,000-game no-book match with zero Stockfish
  wins.

## Build

```sh
cargo build --release
```

`build/build.rs` downloads the current network from the network release if
`networks/eightchess_1024_master.nnue` is missing. Override with `EVALFILE` or
place a compatible `.nnue` in `networks/`.

The resulting executable is `target/release/eightchess`. It supports the
`EvalFile` UCI option, so an external network can be tested without rebuilding:

```sh
setoption name EvalFile value /path/to/network.nnue
```

## Training and data

The training code lives in `training/eightchess_nnue/` (PyTorch). It encodes
positions into the sparse piece/threat feature format, trains soft-target WDL
heads, and exports the exact binary layout consumed by `src/nnue.rs`.

Data and checkpoints are intentionally not committed:

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
  --engine target/release/eightchess --network networks/eightchess_1024_master.nnue --count 100
python3 tools/matches/match_runner.py --help
```

## License

AGPL-3.0, inherited from Reckless. See [LICENSE](LICENSE) and the upstream
project for the original engine and network tooling.

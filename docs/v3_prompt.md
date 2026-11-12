# 8Chess v3 Development Prompt

## Project Overview

8Chess is an open-source chess engine written in Rust with an embedded NNUE network. The engine is strong (near Stockfish 19 level) and has external coverage (chessengines.blogspot.com, chess engine league qualifiers). Version 2 has been shipped with engine improvements (SIMD, PGO, tablebases, correction pass) and the champion NNUE net (768-wide effective, 47.3% vs Stockfish 19 at 1.0+0.01).

**Goal for v3:** Train a genuinely stronger NNUE network that beats Stockfish 19 by a clear margin (≥54% score, +47 Elo) and ideally holds its own against Stockfish 20. Ship with all platform binaries including true runtime SIMD detection.

## Critical Lessons Learned (v2)

These are hard-won findings. Do not repeat these mistakes:

### 1. Fine-tuning the champion net degrades it
Every fine-tune attempt on the champion (768-wide, trained on ~1M positions) made it **worse** vs Stockfish 19:
- All-channel fine-tune (lr 1e-4): 28.4% (−160 Elo)
- Freeze-backbone (head frozen): 18.2% (−261 Elo)
- Soft-freeze (head at 1e-5): 21.5% (−225 Elo)
- Pure-SF fine-tune (lr 1e-5, no revival): 27.6% (−168 Elo)

**Root cause:** The champion is a sharp local optimum. Any gradient update moves away from it and degrades play. More data at low LR does not help — the champion's weights are too sensitive.

**Implication:** Do NOT fine-tune the champion. Train from scratch.

### 2. Reviving dead channels doesn't work with a frozen head
The champion has 256 dead FT channels (768-wide effective). Attempting to revive them and train only the new channels (freeze-backbone) failed because:
- The frozen head (L2/L3) is calibrated for 768 active channels.
- New channels produce activations the head misinterprets as noise.
- The head can't adapt without forgetting the champion's eval.

**Implication:** If using 1024-wide, train ALL channels together from scratch. No freezing, no revival. The channels must co-adapt from random init.

### 3. Val loss is not a reliable proxy for playing strength
In every failed run, val_loss improved slightly while playing strength cratered. Do not trust val_loss alone. Always run match games (100+ vs 8Chess 1.0, 500+ vs SF19) to evaluate.

### 4. Kaggle output publishing is unreliable for long runs
Kaggle kernels that run >10 h sometimes get stuck in RUNNING state and never publish output files. The training completes but the .nnue files are inaccessible.

**Mitigation:** Have the training notebook upload the final net to a Kaggle dataset or HuggingFace Hub at the end of training (before finalization). This saves the net even if output publishing fails.

### 5. Runtime SIMD dispatch requires major refactoring
The engine uses compile-time `#[cfg(target_feature = "...")]` gates. `simd::I16_LANES` is a compile-time const that determines array sizes and loop strides. Making it runtime requires:
- Function pointers for dispatch
- Dynamic arrays (Vec) instead of fixed-size arrays
- Separate code paths for each SIMD level

This is weeks of work. The practical approach (used by Stockfish) is separate binaries per platform/SIMD level + a launcher script.

## Current State

### Engine
- Rust chess engine, ~880k nps with PGO
- SIMD: AVX2, AVX512, NEON, WASM (compile-time selection)
- Syzygy tablebase support
- Correction pass: qsearch, SEE, draw score, time management, FT STE gradient
- Version 2.0.0, champion net embedded (768-wide effective)

### Training Data
- SF mixed: ~376M positions (SF evals at 5000 nodes, sigmoid(score/410) targets)
- Lc0: ~300M positions (game results as targets)
- t91: ~390M positions (SF evals at lower nodes)
- fishpack32: ~42M positions
- Total: ~1.1B positions

### Training Code
- `training/eightchess_nnue/train.py` — sharded streaming trainer, DataParallel, RaggedDataParallel (position-aligned sharding for ragged batches)
- `training/eightchess_nnue/build_shards.py` — streaming .plain/.tsv/.pgn encoder
- `training/eightchess_nnue/model.py` — 1024-wide NNUE (L1_SIZE=1024, 256 dead channels in champion)
- Champion recipe: target sigmoid(score/410), loss sigmoid(cp/173.72), NETWORK_SCALE=380

### What Works
- Engine code is solid (36/36 tests, perft verified)
- SIMD/PGO builds work
- Kaggle training pipeline works (sharded streaming, RaggedDataParallel verified on CPU)
- Website is clean (no SF claims, news page added)
- Match runner (tools/matches/uci_match.py) works for direct-UCI matches

### What Doesn't Work
- Fine-tuning the champion (any approach) — degrades playing strength
- Reviving 256 dead channels — introduces noise the head can't handle
- Kaggle output publishing for runs >10 h (intermittent)
- True runtime SIMD detection (compile-time only)

## v3 Goals

### Primary: Train a stronger NNUE from scratch
- **Architecture:** 1024-wide (all channels active, no dead channels)
- **Data:** Full 1.08B corpus (SF evals + Lc0 results + t91)
- **Training:** 3–5 epochs, from random init, no champion prior
- **LR:** Standard NNUE recipe (lr 1e-3 for head, 1e-4 for FT, cosine decay)
- **Expected:** Should reach champion strength (47.3%) by epoch 2-3, then improve
- **Compute needed:** ~30 h GPU (fits Nov 14 quota refill)

### Secondary: Runtime SIMD detection
- Refactor SIMD layer to use function pointers for dispatch
- Make `simd::I16_LANES` a runtime value (or use separate code paths per level)
- Single binary that detects CPU features and uses best available SIMD
- This is a significant refactor — estimate 1-2 weeks

### Tertiary: Platform binaries
- Windows (x86-64, ARM64)
- macOS (Intel, Apple Silicon) — universal binary via lipo
- Linux (x86-64, ARM64, RISC-V)
- WASM (already working)

## Specific Improvements to Try

### Training
1. **From-scratch on 1024-wide** — the main experiment. No champion, no revival. Clean.
2. **Curriculum learning** — train on high-quality SF data (5000 nodes) first, then add Lc0 results.
3. **Higher SF node counts** — re-encode SF data at 10k+ nodes for better targets.
4. **Position filtering** — drop opening book positions, weight decisive games.
5. **Knowledge distillation** — train on SF20's evals (when available) instead of SF19's.
6. **Self-play data** — use the new net to generate on-policy positions, re-train.

### Architecture (only if from-scratch plateaus)
- Larger L2/L3 (32/64 instead of 16/32) — more capacity, but changes net size
- More threat features — better positional understanding, but changes format
- Different bucket schemes — better eval for different game phases

### Engine
- Runtime SIMD detection (see above)
- Multi-threaded search improvements
- Better time management
- Opening book (currently broken — only built-in openings)

## Constraints
- **Compute:** 30 h/week Kaggle (free), or GCP credits (apply when ready)
- **Quota refreshes:** Nov 14 (weekly rolling)
- **Credibility:** Must publish honest match results with exact conditions
- **Version compatibility:** v3 net must work with v2 engine (or document breaking changes)
- **No quantization** until training + verification complete
- **Keep 1024-wide** — no smaller/weaker net

## What NOT to do (learned from v2)
- Don't fine-tune the champion — it degrades
- Don't revive dead channels on a frozen head — introduces noise
- Don't trust val_loss alone — always run match games
- Don't train on mixed targets without proper blending (lam parameter)
- Don't spend quota on experiments that can't produce a clean result
- Don't claim 54%+ without 500+ game match evidence

## Success Criteria
- v3 net scores ≥ 47.3% vs SF19 (champion level) — minimum bar
- v3 net scores ≥ 54% vs SF19 (+47 Elo) — target
- 500+ game match with 95% CI
- Honest publication of results regardless of outcome
- v2 engine improvements (search, SIMD, tablebases) remain intact

## Files to Know
- `src/nnue.rs` — NNUE model, SIMD dispatch (compile-time cfg)
- `src/nnue/simd/{avx2,avx512,neon,wasm,scalar}.rs` — SIMD implementations
- `src/nnue/accumulator.rs` — accumulator with PSQ and threats
- `training/eightchess_nnue/train.py` — sharded streaming trainer (DataParallel, RaggedDataParallel)
- `training/eightchess_nnue/model.py` — PyTorch model (1024-wide, STE quantization)
- `training/eightchess_nnue/build_shards.py` — data encoder
- `kaggle/gputrain/` — Kaggle training kernel
- `tools/matches/uci_match.py` — direct-UCI match runner
- `docs/news.html` — website news page
- `.github/workflows/release.yml` — multi-platform release CI
- `Cargo.toml` — version, features, dependencies
- `.cargo/config.toml` — RUSTFLAGS (target-cpu)
- `build/build.rs` — build script (embeds NNUE, generates lookup tables)

## Compute Budget
- Kaggle: 30 h/week (free), rolling 7-day window
- Each full training run: ~10.5 h (1 epoch on 1.08B positions)
- From-scratch needs 3–5 epochs = 29–49 h
- Plan: 1 run per week, evaluate, iterate
- Apply for GCP open-source credits when project has more visibility
- Consider Modal ($30 free credits) for one-shot experiments
- Consider OpenBench for distributed match testing (free, community compute)

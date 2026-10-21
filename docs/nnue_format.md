# 8Chess NNUE format specification

This document specifies the network architecture, feature set, quantization,
and binary layout used by the 8Chess engine (inherited from the Reckless
lineage and kept binary-compatible with its inference code).

The format is **fixed-layout**: the engine deserializes a network file by
transmuting the file contents into a `#[repr(C)]` struct (`src/nnue.rs`,
`Parameters::embedded`). A network file is therefore a raw little-endian
memory image with no header or metadata. Load it with the `EvalFile` UCI
option, or at build time with the `EVALFILE` environment variable.

Everything below is asserted in code and continuously verified by
`training/eightchess_nnue/verify_engine.py` against the compiled engine
(`evalnum`), currently at a 100% exact-match rate.

## 1. Architecture

```
piece features (10 x 768)   ──┐
                              ├──> feature transformer (1024) ──> clipped-pair
threat features (66864)     ──┘                                   activation (1024)
                                                                       │
                                               input-clipped affine (1024 -> 16)  x 8 output buckets
                                                                      │
                                                         affine (16 -> 32) x 8 buckets
                                                                      │
                                                          affine (32 -> 1) x 8 buckets
                                                                      │
                                                       score = output * 380
```

* Feature transformer output: 1024 (`L1_SIZE`), two halves of 512.
* Input buckets: 10, selected by the POV king square (layout below).
* Threat features: **not** input-bucketed (shared weights across king zones).
* Output buckets: 8, selected by total piece count.
* Activations: clipped-pair squaring for the FT, then `clamp(0,1)` on the
  L1/L2 outputs (i.e. clipped ReLU).

### Input bucket layout (indexed by square, a1 = 0)

```
row 1: 0 1 2 3 3 2 1 0
row 2: 4 5 6 7 7 6 5 4
rows 3-8: 8 8 8 8 8 8 8 8   (row 3)
          9 9 9 9 9 9 9 9   (rows 4-8)
```

### Output bucket layout (indexed by occupied square count, 2..32)

```
pieces:  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
bucket:  0  0  0  0  0  0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  4  4  4  5  5  5  6  6  6  7  7  7  7
```

## 2. Features

### 2.1 Piece features (7680 total)

For POV `pov` with POV king on `king`:

```
kingside = (king & 7) >= 4
flip     = (7 if kingside else 0) ^ (56 if pov == black else 0)
bucket   = INPUT_BUCKETS_LAYOUT[king ^ flip]
index    = bucket * 768
         + 384 * (piece_color != pov)
         + 64  * piece_type            # P=0 N=1 B=2 R=3 Q=4 K=5
         + (piece_square ^ flip)
```

All 32 pieces (both colors) contribute. Indices are in `[0, 7680)`.

### 2.2 Threat features (66864 total)

A threat is a pair (attacker, target) where the attacker attacks an occupied
square. Attack sets respect blockers (sliding pieces use occupancy-aware
attacks); attacks on friendly pieces (defenses) are included.

The feature index is built from precomputed tables (see
`src/nnue/accumulator/threats/threat_index.rs` and the Python port in
`training/eightchess_nnue/constants.py`):

```
from' = relative_to(from, pov) ^ (7 if pov_king_is_kingside else 0)
to'   = relative_to(to, pov)   ^ (7 if pov_king_is_kingside else 0)
attacking = attacker_piece ^ pov       # piece code = type*2 + color
attacked  = target_piece   ^ pov

index = piece_pair_base[attacking][attacked]
      + piece_square_offset[attacking][from']
      + attack_index[attacking][from'][to']
```

where `attack_index` counts the attacked squares strictly below `to'`. Pairs
marked *excluded* in `PIECE_INTERACTION_MAP` (e.g. pawn attacks on bishops or
queens) are not features, and *semi-excluded* pairs (same piece type, e.g.
Knight–Knight) are stored only once: the pair contributes only when
`from' >= to'`. Excluded features yield indices outside `[0, 66864)` in the
engine's formulation and are skipped.

Piece codes: `WhitePawn=0, BlackPawn=1, WhiteKnight=2, BlackKnight=3, ...`
(`piece = type*2 + color`, White=0, Black=1). Square index: `rank*8 + file`,
a1 = 0.

## 3. Inference semantics (exact)

Accumulators are `i16` and **wrap** on overflow (SIMD `add_epi16`):

```
pst[pov]    = wrap16(ft_bias + sum(ft_piece_weights[piece_features(pov)]))
threat[pov] = wrap16(sum(ft_threat_weights[threat_features(pov)]))
```

Activation (per side-to-move flip):

```
s     = wrap16(pst[pov] + threat[pov])              # same wrap
lhs   = clamp(s[0:512],   0, 255)
rhs   = min(s[512:1024], 255)                       # may be negative
byte  = saturate_u8(floor(lhs * rhs / 512))          # packus semantics
ft_out[flip*512 ..] = byte
```

L1 layer (bucket `b` from piece count):

```
pre[j] = sum over nonzero 4-byte chunks i, over k=0..3:
             ft_out[4i+k] * l1_weights[b][ (i*16 + j)*4 + k ]
l1[j]  = clamp(pre[j] * 512 / (255*255*64) + l1_biases[b][j], 0, 1)
```

(The `512/(255*255*64)` dequant means an `l1_weight` of `w` corresponds to a
float weight of `w/64`.)

L2 layer (float32, FMA):

```
l2[j] = clamp(l2_biases[b][j] + sum_i l2_weights[b][i][j] * l1[i], 0, 1)     # i in 0..16
```

L3 layer (float32, FMA, 16-lane tree reduction then halving):

```
sums[l]  = sum over i in {l, l+16}: l3_weights[b][i]*l2[i]   # 16 lanes l=0..15
halving reduction: sums[i] += sums[i+8]; ... ; sums[0]
l3_out   = sums[0] + l3_biases[b]
score    = (l3_out * 380.0) as i32        # truncation toward zero
```

## 4. Quantized training conventions

To be exportable, the float model must respect:

| Tensor               | Type  | Scale                      | Float range |
| -------------------- | ----- | -------------------------- | ----------- |
| `ft_piece_weights`   | `i16` | `w_int = round(255*w)`     | ±128        |
| `ft_threat_weights`  | `i8`  | `w_int = clamp(round(255*w), -127, 127)` | ±0.498 |
| `ft_biases`          | `i16` | `b_int = round(255*b)`     | ±128        |
| `l1_weights`         | `i8`  | `w_int = clamp(round(64*w), -127, 127)`  | ±1.98 |
| `l1_biases`          | `f32` | direct                     | —           |
| `l2/l3_weights`      | `f32` | direct                     | —           |
| `l2/l3_biases`       | `f32` | direct                     | —           |

Piece and threat weights share the same float scale (`255 = 1.0` activation
unit). Because the engine sums both raw integer parts, the float scale factor
**must be identical** for both. Threat weights are 8-bit, so training must
keep them within `[-0.498, 0.498]` (the exporter clamps).

## 5. Binary layout

The file is exactly **84,349,504 bytes** (compile-time asserted in
`src/nnue.rs`). All offsets are multiples of 64 (fields are `repr(align(64))`).
Little-endian throughout.

| Offset      | Size      | Type     | Shape                 | Name                |
| ----------- | --------- | -------- | --------------------- | ------------------- |
| 0           | 68,468,736 | `i8`    | `[66864][1024]`       | `ft_threat_weights` |
| 68,468,736  | 15,728,640 | `i16`   | `[7680][1024]`        | `ft_piece_weights`  |
| 84,197,376  | 2,048     | `i16`    | `[1024]`              | `ft_biases`         |
| 84,199,424  | 131,072   | `i8`     | `[8][16*1024]`        | `l1_weights`        |
| 84,330,496  | 512       | `f32`    | `[8][16]`             | `l1_biases`         |
| 84,331,008  | 16,384    | `f32`    | `[8][16][32]`         | `l2_weights`        |
| 84,347,392  | 1,024     | `f32`    | `[8][32]`             | `l2_biases`         |
| 84,348,416  | 1,024     | `f32`    | `[8][32]`             | `l3_weights`        |
| 84,349,440  | 32 (+32 pad) | `f32` | `[8]`                 | `l3_biases`         |

`l1_weights` layout: `w[(chunk * 16 + output) * 4 + k]` where `chunk` is the
4-byte input chunk index (`0..256`).

## 6. Compatibility checks implemented

* `training/eightchess_nnue/format.py` asserts all offsets and the total size.
* `training/eightchess_nnue/verify_engine.py` compares the replica against the
  engine on arbitrary FEN batches (exact integer match requirement).
* `training/eightchess_nnue/export.py` performs an end-to-end round trip:
  floats → quantized ints → binary → parse back → quantized forward, and
  reports the quantization error.
* The engine rejects files whose size differs from 84,349,504 bytes.

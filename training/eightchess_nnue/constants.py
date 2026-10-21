"""Architecture constants for the 8Chess (Reckless-lineage) NNUE.

These MUST stay in sync with `src/nnue.rs` in the 8Chess engine.
See docs/nnue_format.md for the full binary layout specification.
"""

# Feature transformer
L1_SIZE = 1024
L2_SIZE = 16
L3_SIZE = 32

INPUT_BUCKETS = 10
OUTPUT_BUCKETS = 8

# Threat feature space (66864 = total sized from piece interaction tables)
THREAT_FEATURES = 66864
PIECE_FEATURES = INPUT_BUCKETS * 768  # 7680

# Quantization (must match src/nnue.rs)
FT_QUANT = 255
L1_QUANT = 64
FT_SHIFT = 9  # scalar inference uses (left * right) >> FT_SHIFT

DEQUANT_MULTIPLIER = float(1 << FT_SHIFT) / float(FT_QUANT * FT_QUANT * L1_QUANT)

NETWORK_SCALE = 380

# Input bucket layout indexed by square (a1=0 .. h8=63)
INPUT_BUCKETS_LAYOUT = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
]

# Output bucket layout indexed by piece count
OUTPUT_BUCKETS_LAYOUT = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7, 7,
]

# Exact serialized size of the Parameters struct (bytes)
NETWORK_SIZE = 84_349_504


def output_bucket(piece_count: int) -> int:
    return OUTPUT_BUCKETS_LAYOUT[piece_count]


# ---------------------------------------------------------------------------
# Threat feature indexing tables (port of src/nnue/accumulator/threats/
# threat_index.rs). Built once on import.
# ---------------------------------------------------------------------------

# Rows/columns: PieceType (P,N,B,R,Q,K). -1 == excluded interaction.
PIECE_INTERACTION_MAP = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
]

PIECE_TARGET_COUNT = [6, 10, 8, 8, 10, 8]


def _build_attack_tables():
    """Attack sets (as 64-bit masks) for offsets 0..63, occupancy 0.

    Mirrors the generated tables in the engine for the purpose of
    (a) counting target squares per (piece, square) and
    (b) computing the "number of attacked squares below `to`" indices.
    """
    knight = [0] * 64
    king = [0] * 64
    pawn = [[0] * 64 for _ in range(2)]  # [color][square]
    ray_diag = [[0] * 64 for _ in range(4)]
    ray_orth = [[0] * 64 for _ in range(4)]

    knight_deltas = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    king_deltas = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    diag_deltas = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    orth_deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def add_deltas(deltas, single_step):
        table = [0] * 64
        for sq in range(64):
            f, r = sq & 7, sq >> 3
            bb = 0
            for df, dr in deltas:
                nf, nr = f, r
                while True:
                    nf, nr = nf + df, nr + dr
                    if not (0 <= nf < 8 and 0 <= nr < 8):
                        break
                    bb |= 1 << (nr * 8 + nf)
                    if single_step:
                        break
            table[sq] = bb
        return table

    knight = add_deltas(knight_deltas, True)
    king = add_deltas(king_deltas, True)
    ray_diag = add_deltas(diag_deltas, False)
    ray_orth = add_deltas(orth_deltas, False)

    for sq in range(64):
        f, r = sq & 7, sq >> 3
        # white pawn attacks (towards higher ranks)
        w = 0
        for df in (-1, 1):
            nf, nr = f + df, r + 1
            if 0 <= nf < 8 and 0 <= nr < 8:
                w |= 1 << (nr * 8 + nf)
        pawn[0][sq] = w
        b = 0
        for df in (-1, 1):
            nf, nr = f + df, r - 1
            if 0 <= nf < 8 and 0 <= nr < 8:
                b |= 1 << (nr * 8 + nf)
        pawn[1][sq] = b

    return {
        "knight": knight,
        "king": king,
        "pawn": pawn,
        "ray_diag": ray_diag,
        "ray_orth": ray_orth,
    }


_ATTACKS = _build_attack_tables()


def attacks_empty(piece: int, square: int) -> int:
    """Attack mask with empty occupancy. piece: 0..11 (type*2 + color)."""
    ptype = piece >> 1
    color = piece & 1
    if ptype == 0:
        return _ATTACKS["pawn"][color][square]
    if ptype == 1:
        return _ATTACKS["knight"][square]
    if ptype == 2:
        return _ATTACKS["ray_diag"][square]
    if ptype == 3:
        return _ATTACKS["ray_orth"][square]
    if ptype == 4:
        return _ATTACKS["ray_diag"][square] | _ATTACKS["ray_orth"][square]
    if ptype == 5:
        return _ATTACKS["king"][square]
    return 0


def _build_threat_lookups():
    piece_offset = [0] * 12
    offset_table = [0] * 12

    offset = 0
    for color in (0, 1):
        for ptype in range(6):
            piece = ptype * 2 + color
            count = 0
            for sq in range(64):
                if ptype != 0 or 8 <= sq < 56:
                    count += bin(attacks_empty(piece, sq)).count("1")
            piece_offset[piece] = count
            offset_table[piece] = offset
            offset += PIECE_TARGET_COUNT[ptype] * count

    total = offset
    assert total == THREAT_FEATURES, (total, THREAT_FEATURES)

    # pair base lookup + exclusion flags
    pair_base = [[0] * 12 for _ in range(12)]
    pair_excluded = [[False] * 12 for _ in range(12)]
    pair_semi = [[False] * 12 for _ in range(12)]

    for attacking in range(12):
        atk_type, atk_color = attacking >> 1, attacking & 1
        for attacked in range(12):
            def_type, def_color = attacked >> 1, attacked & 1
            m = PIECE_INTERACTION_MAP[atk_type][def_type]
            base = (
                offset_table[attacking]
                + (def_color * (PIECE_TARGET_COUNT[atk_type] // 2) + m) * piece_offset[attacking]
            )
            enemy = atk_color != def_color
            semi = atk_type == def_type and (enemy or atk_type != 0)
            pair_base[attacking][attacked] = base
            pair_excluded[attacking][attacked] = m < 0
            pair_semi[attacking][attacked] = semi

    piece_offsets = [None] * 12
    attack_index = [None] * 12
    for piece in range(12):
        piece_offsets[piece] = [0] * 64
        count = 0
        for sq in range(64):
            piece_offsets[piece][sq] = count
            if (piece >> 1) != 0 or 8 <= sq < 56:
                count += bin(attacks_empty(piece, sq)).count("1")
        assert count == piece_offset[piece]

        table = [0] * 4096
        for frm in range(64):
            atk = attacks_empty(piece, frm)
            for to in range(64):
                table[frm * 64 + to] = bin(atk & ((1 << to) - 1)).count("1")
        attack_index[piece] = table

    return {
        "pair_base": pair_base,
        "pair_excluded": pair_excluded,
        "pair_semi": pair_semi,
        "piece_offsets": piece_offsets,
        "attack_index": attack_index,
    }


THREAT_LOOKUPS = _build_threat_lookups()

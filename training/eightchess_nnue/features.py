"""Exact feature extraction for the 8Chess NNUE (port of the Rust code).

Provides piece (PSQT) and threat feature indices for a python-chess board,
matching `src/nnue/accumulator/psq.rs` and
`src/nnue/accumulator/threats/threat_index.rs` bit for bit.
"""

from __future__ import annotations

from typing import Iterable, Optional

import chess

from .constants import (
    INPUT_BUCKETS_LAYOUT,
    PIECE_FEATURES,
    THREAT_FEATURES,
    THREAT_LOOKUPS,
    attacks_empty,
)

_PIECE_CODE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.PAWN, chess.BLACK): 1,
    (chess.KNIGHT, chess.WHITE): 2,
    (chess.KNIGHT, chess.BLACK): 3,
    (chess.BISHOP, chess.WHITE): 4,
    (chess.BISHOP, chess.BLACK): 5,
    (chess.ROOK, chess.WHITE): 6,
    (chess.ROOK, chess.BLACK): 7,
    (chess.QUEEN, chess.WHITE): 8,
    (chess.QUEEN, chess.BLACK): 9,
    (chess.KING, chess.WHITE): 10,
    (chess.KING, chess.BLACK): 11,
}

_PTYPE_CODE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_DIAG_DELTAS = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
_ORTH_DELTAS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _ray_attacks(sq: int, deltas, occ: int) -> int:
    bb = 0
    f, r = sq & 7, sq >> 3
    for df, dr in deltas:
        nf, nr = f, r
        while True:
            nf, nr = nf + df, nr + dr
            if not (0 <= nf < 8 and 0 <= nr < 8):
                break
            s = nr * 8 + nf
            bb |= 1 << s
            if (occ >> s) & 1:
                break
    return bb


def attacks_with_occupancy(piece: int, square: int, occ: int) -> int:
    ptype = piece >> 1
    if ptype in (2, 4):
        bb = _ray_attacks(square, _DIAG_DELTAS, occ)
        if ptype == 4:
            bb |= _ray_attacks(square, _ORTH_DELTAS, occ)
        return bb
    if ptype == 3:
        return _ray_attacks(square, _ORTH_DELTAS, occ)
    return attacks_empty(piece, square)


def iter_bits(bb: int) -> Iterable[int]:
    while bb:
        lsb = bb & -bb
        yield lsb.bit_length() - 1
        bb ^= lsb


def piece_feature_indices(board: chess.Board) -> list[list[int]]:
    """Returns [pov0_features, pov1_features] where pov matches side-to-move order.

    pov0 == White, pov1 == Black: the engine refreshes accumulators for both
    colors; the activation then selects by side to move.
    """
    result: list[list[int]] = []
    for pov in (chess.WHITE, chess.BLACK):
        king = board.king(pov)
        assert king is not None
        kingside = (king & 7) >= 4
        flip = (7 if kingside else 0) ^ (56 if pov == chess.BLACK else 0)
        bucket = INPUT_BUCKETS_LAYOUT[king ^ flip]

        feats = []
        for sq in chess.scan_reversed(board.occupied):
            piece = board.piece_at(sq)
            assert piece is not None
            color = 0 if piece.color == chess.WHITE else 1
            ptype = _PTYPE_CODE[piece.piece_type]
            pov_code = 0 if pov == chess.WHITE else 1
            idx = bucket * 768 + 384 * (color != pov_code) + 64 * ptype + (sq ^ flip)
            assert 0 <= idx < PIECE_FEATURES
            feats.append(idx)
        result.append(feats)
    return result


def threat_index(piece: int, frm: int, attacked: int, to: int, mirrored: bool, pov: int) -> Optional[int]:
    """Port of `threat_index` in Rust. Returns None when the pair/square
    ordering is excluded (symmetric interactions are stored once)."""
    if pov == 1:
        frm ^= 56
        to ^= 56
    if mirrored:
        frm ^= 7
        to ^= 7

    attacking = piece ^ pov
    attacked_p = attacked ^ pov

    look = THREAT_LOOKUPS
    if look["pair_excluded"][attacking][attacked_p]:
        return None
    if look["pair_semi"][attacking][attacked_p] and frm < to:
        return None

    base = look["pair_base"][attacking][attacked_p]
    idx = base + look["piece_offsets"][attacking][frm] + look["attack_index"][attacking][frm * 64 + to]
    assert 0 <= idx < THREAT_FEATURES, idx
    return idx


def threat_feature_indices(board: chess.Board) -> list[list[int]]:
    """Returns [pov0_features, pov1_features] with pov = White/Black."""
    occ = board.occupied
    result: list[list[int]] = []

    for pov in (chess.WHITE, chess.BLACK):
        king = board.king(pov)
        assert king is not None
        mirrored = (king & 7) >= 4
        pov_code = 0 if pov == chess.WHITE else 1

        feats = []
        for sq in chess.scan_reversed(occ):
            piece_obj = board.piece_at(sq)
            assert piece_obj is not None
            piece = _PIECE_CODE[(piece_obj.piece_type, piece_obj.color)]
            targets = attacks_with_occupancy(piece, sq, occ) & occ
            for to in iter_bits(targets):
                target = board.piece_at(to)
                assert target is not None
                attacked = _PIECE_CODE[(target.piece_type, target.color)]
                idx = threat_index(piece, sq, attacked, to, mirrored, pov_code)
                if idx is not None:
                    feats.append(idx)
        result.append(feats)
    return result


def full_feature_indices(board: chess.Board):
    """(piece_features, threat_features), each [pov0, pov1]."""
    return piece_feature_indices(board), threat_feature_indices(board)

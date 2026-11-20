use bullet_lib::game::{
    formats::bulletformat::ChessBoard,
    inputs::SparseInputType,
};
use montyformat::chess::{Attacks, Side};

const INPUT_BUCKETS_LAYOUT: [usize; 64] = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
];

const OUTPUT_BUCKETS_LAYOUT: [usize; 33] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7,
];

fn king_square(pos: &ChessBoard, white: bool) -> usize {
    for (pc, sq) in pos.into_iter() {
        if (pc & 7) == 5 && ((pc & 8) == 0) == white {
            return sq as usize;
        }
    }
    0
}

/// Returns (white_pov_indices, black_pov_indices) for piece features.
pub fn piece_indices(pos: &ChessBoard) -> (Vec<usize>, Vec<usize>) {
    let wk = king_square(pos, true);
    let bk = king_square(pos, false);

    let mut white_feats: Vec<usize> = Vec::new();
    let mut black_feats: Vec<usize> = Vec::new();

    for pov_white in [true, false] {
        let king = if pov_white { wk } else { bk };
        let kingside = (king & 7) >= 4;
        let flip = (if kingside { 7usize } else { 0usize }) ^ (if pov_white { 0usize } else { 56usize });
        let bucket = INPUT_BUCKETS_LAYOUT[king ^ flip];
        let target_list = if pov_white { &mut white_feats } else { &mut black_feats };

        for (pc, sq) in pos.into_iter() {
            let color_white = (pc & 8) == 0;
            let ptype = (pc & 7) as usize;
            let color_differs = color_white != pov_white;
            let idx = bucket * 768 + 384 * (color_differs as usize) + 64 * ptype + (sq as usize ^ flip);
            target_list.push(idx);
        }
    }
    (white_feats, black_feats)
}

/// Returns (white_pov_indices, black_pov_indices) for threat features.
pub fn threat_indices(pos: &ChessBoard) -> (Vec<usize>, Vec<usize>) {
    eightchess::init_threat_indexing();
    let wk = king_square(pos, true);
    let bk = king_square(pos, false);

    let mut bb2 = [0u64; 8];
    for (pc, sq) in pos.into_iter() {
        let bit = 1u64 << sq;
        let side_idx = if pc & 8 == 0 { 0usize } else { 1usize };
        let ptype = (pc & 7) as usize;
        bb2[side_idx] |= bit;
        bb2[2 + ptype] |= bit;
    }
    let occ = bb2[0] | bb2[1];

    let mut pieces = [usize::MAX; 64];
    for (pc, sq) in pos.into_iter() {
        let side = if pc & 8 == 0 { 0usize } else { 1usize };
        let ptype = (pc & 7) as usize;
        pieces[sq as usize] = ptype * 2 + side;
    }

    let mut white_feats: Vec<usize> = Vec::new();
    let mut black_feats: Vec<usize> = Vec::new();

    for (pov_white, target_list) in [(true, &mut white_feats), (false, &mut black_feats)] {
        let king = if pov_white { wk } else { bk };
        let mirrored = (king & 7) >= 4;
        for side in [Side::WHITE, Side::BLACK] {
            let side_idx = if side == Side::WHITE { 0usize } else { 1usize };
            for ptype in 0..6usize {
                let mut bb = bb2[side_idx] & bb2[2 + ptype];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    let attacks_bb = match ptype {
                        0 => Attacks::pawn(sq, side),
                        1 => Attacks::knight(sq),
                        2 => Attacks::bishop(sq, occ),
                        3 => Attacks::rook(sq, occ),
                        4 => Attacks::queen(sq, occ),
                        5 => Attacks::king(sq),
                        _ => 0,
                    } & occ;
                    let mut t = attacks_bb;
                    while t != 0 {
                        let dest = t.trailing_zeros() as usize;
                        t &= t - 1;
                        let target_engine = pieces[dest];
                        if target_engine == usize::MAX { continue; }
                        let attacking_engine = ptype * 2 + side_idx;
                        let idx = eightchess::threat_index_raw(
                            attacking_engine as u8,
                            sq as u8,
                            target_engine as u8,
                            dest as u8,
                            mirrored,
                            pov_white,
                        );
                        if idx >= 0 {
                            target_list.push(idx as usize);
                        }
                    }
                }
            }
        }
    }
    (white_feats, black_feats)
}

/// Returns stm_is_white
pub fn stm_is_white(pos: &ChessBoard) -> bool {
    let our_ksq = pos.our_ksq() as usize;
    for (pc, sq) in pos.into_iter() {
        if sq as usize == our_ksq && (pc & 7) == 5 {
            return (pc & 8) == 0;
        }
    }
    true
}

pub fn output_bucket(pos: &ChessBoard) -> usize {
    let count = pos.into_iter().count();
    let idx = count.clamp(2, 32);
    OUTPUT_BUCKETS_LAYOUT[idx]
}

#[derive(Clone, Copy)]
pub struct PieceFeatures8Chess;

impl SparseInputType for PieceFeatures8Chess {
    type RequiredDataType = ChessBoard;
    fn num_inputs(&self) -> usize { 7680 }
    fn max_active(&self) -> usize { 32 }
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let (w, b) = piece_indices(pos);
        let stm_w = stm_is_white(pos);
        let (stm_list, ntm_list) = if stm_w { (w, b) } else { (b, w) };
        let n = stm_list.len().min(ntm_list.len());
        for i in 0..n {
            f(stm_list[i], ntm_list[i]);
        }
    }
    fn shorthand(&self) -> String { "piece".to_string() }
    fn description(&self) -> String { "8Chess piece features".to_string() }
}

#[derive(Clone, Copy)]
pub struct ThreatFeatures8Chess;

impl SparseInputType for ThreatFeatures8Chess {
    type RequiredDataType = ChessBoard;
    fn num_inputs(&self) -> usize { 66864 }
    fn max_active(&self) -> usize { 256 }
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let (w, b) = threat_indices(pos);
        let stm_w = stm_is_white(pos);
        let (stm_list, ntm_list) = if stm_w { (w, b) } else { (b, w) };
        let n = stm_list.len().min(ntm_list.len());
        for i in 0..n {
            f(stm_list[i], ntm_list[i]);
        }
    }
    fn shorthand(&self) -> String { "threat".to_string() }
    fn description(&self) -> String { "8Chess threat features".to_string() }
}

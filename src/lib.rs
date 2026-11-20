#![allow(unsafe_op_in_unsafe_fn)]
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::redundant_clone)]
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports))]

mod board;
mod evaluation;
mod history;
pub mod lookup;
mod misc;
mod movepick;
pub mod nnue;
mod numa;
mod parameters;
mod search;
mod setwise;
mod stack;
mod thread;
mod threadpool;
mod time;
mod transposition;
pub mod types;

mod tools;

#[cfg(not(target_arch = "wasm32"))]
mod uci;

#[cfg(feature = "syzygy")]
mod tb;

#[cfg(feature = "syzygy")]
#[allow(warnings)]
mod bindings;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(not(target_arch = "wasm32"))]
pub fn run(buffer: std::collections::VecDeque<String>) {
    lookup::initialize();
    nnue::initialize();
    uci::message_loop(buffer);
}


/// Training helpers exposed for external NNUE trainers (e.g. Bullet).
/// Initializes the lookup tables required by `threat_index_raw`.
pub fn init_threat_indexing() {
    lookup::initialize();
    nnue::accumulator::threats::initialize();
}

/// Returns the 8Chess threat feature index for a single (attacker, target) pair,
/// or -1 if the pair is excluded. `attacking`/`attacked` are engine Piece indices
/// (piece_type*2 + color). `from`/`to` are squares (0..64). `mirrored` is whether
/// the king is on the kingside. `pov_white` selects the accumulator POV.
pub fn threat_index_raw(attacking: u8, from: u8, attacked: u8, to: u8, mirrored: bool, pov_white: bool) -> isize {
    let piece = types::Piece::from_index(attacking as usize);
    let attacked_piece = types::Piece::from_index(attacked as usize);
    let from_sq = types::Square::new(from);
    let to_sq = types::Square::new(to);
    let pov = if pov_white { types::Color::White } else { types::Color::Black };
    nnue::accumulator::threats::threat_index(piece, from_sq, attacked_piece, to_sq, mirrored, pov)
}

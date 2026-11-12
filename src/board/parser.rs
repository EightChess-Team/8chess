use super::Board;
use crate::{
    lookup::between,
    types::{
        Bitboard, CastlingKind, Color, HOME_RANK, KING_TO_FILE, Piece, PieceType, ROOK_TO_FILE, Square,
    },
};

#[derive(Debug)]
pub enum ParseFenError {
    /// The FEN string is missing piece placement data.
    MissingPlacementData,
    /// The piece placement data is malformed (wrong rank/file count).
    InvalidPlacementData,
    /// The FEN string contains an invalid piece type character.
    InvalidPieceType,
    /// The FEN string contains an invalid active color.
    InvalidActiveColor,
    /// The FEN string does not contain exactly one king for the given color.
    MissingKing(Color),
    /// The FEN string places a pawn on the first or last rank.
    InvalidPawnPlacement,
}

impl Board {
    /// Parses a [Forsyth–Edwards Notation][fen] string into a `Board`.
    ///
    /// The parser accepts the common shorthand forms with missing optional
    /// fields, but rejects placement data that would otherwise index outside
    /// the board or produce a position without exactly one king per side.
    ///
    /// [fen]: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    pub fn from_fen(fen: &str) -> Result<Self, ParseFenError> {
        let mut board = Self::default();
        let mut parts = fen.split_whitespace();

        let rows = parts.next().ok_or(ParseFenError::MissingPlacementData)?.split('/');
        let mut rank_count = 0;

        for (rank, row) in rows.rev().enumerate() {
            if rank >= 8 {
                return Err(ParseFenError::InvalidPlacementData);
            }
            rank_count += 1;

            let mut file = 0u8;

            for symbol in row.chars() {
                if let Some(skip) = symbol.to_digit(10) {
                    if !(1..=8).contains(&skip) {
                        return Err(ParseFenError::InvalidPlacementData);
                    }
                    file += skip as u8;
                } else {
                    if file >= 8 {
                        return Err(ParseFenError::InvalidPlacementData);
                    }

                    let piece = symbol.try_into().map_err(|()| ParseFenError::InvalidPieceType)?;
                    let square = Square::from_rank_file(rank as u8, file);

                    board.add_piece(piece, square);
                    board.state.material += piece.value();
                    file += 1;
                }

                if file > 8 {
                    return Err(ParseFenError::InvalidPlacementData);
                }
            }

            if file != 8 {
                return Err(ParseFenError::InvalidPlacementData);
            }
        }

        if rank_count != 8 {
            return Err(ParseFenError::InvalidPlacementData);
        }

        for color in [Color::White, Color::Black] {
            if board.colored_pieces(color, PieceType::King).popcount() != 1 {
                return Err(ParseFenError::MissingKing(color));
            }
        }

        if !(board.pieces(PieceType::Pawn) & Bitboard::BOTH_HOME_ROWS).is_empty() {
            return Err(ParseFenError::InvalidPawnPlacement);
        }

        let side_to_move = match parts.next() {
            Some("w") => Color::White,
            Some("b") => Color::Black,
            _ => return Err(ParseFenError::InvalidActiveColor),
        };

        board.set_castling(parts.next().unwrap_or("-"));

        let expected_en_passant_rank = if side_to_move == Color::White { 5 } else { 2 };
        board.state.en_passant = parts
            .next()
            .and_then(|value| Square::try_from(value).ok())
            .filter(|square| square.rank() as u8 == expected_en_passant_rank)
            .unwrap_or(Square::None);

        board.state.fiftymove_clock = parts.next().unwrap_or_default().parse().unwrap_or_default();
        let fullmove_number: usize = parts.next().unwrap_or_default().parse().unwrap_or_default();
        board.halfmove_number =
            fullmove_number.max(1).saturating_mul(2).saturating_add(side_to_move as usize);

        board.update_threats();
        board.update_hash_keys();
        board.validate_en_passant();

        Ok(board)
    }

    fn set_castling(&mut self, rights: &str) {
        for right in rights.chars() {
            let upper = right.to_ascii_uppercase();
            if !matches!(upper, 'A'..='H' | 'K' | 'Q') {
                continue;
            }

            let color = if right.is_uppercase() { Color::White } else { Color::Black };
            let king_from = self.king_square(color);

            if king_from.rank() as u8 != HOME_RANK[color] as u8 {
                continue;
            }

            let rooks = self.colored_pieces(color, PieceType::Rook) & Bitboard::rank(HOME_RANK[color]);

            let rook_from = if upper == 'K' {
                rooks.into_iter().filter(|rook| *rook > king_from).min_by_key(|rook| *rook as u8)
            } else if upper == 'Q' {
                rooks.into_iter().filter(|rook| *rook < king_from).max_by_key(|rook| *rook as u8)
            } else {
                let file = upper as u8 - b'A';
                rooks.into_iter().find(|rook| rook.file() as u8 == file)
            };

            let Some(rook_from) = rook_from else {
                continue;
            };

            let king_side = (rook_from > king_from) as usize;
            let rights = CastlingKind::KINDS[color][king_side];

            // Ignore duplicate or contradictory castling rights.
            if self.state.castling.is_allowed(rights) {
                continue;
            }

            let king_to = Square::from_rank_file(HOME_RANK[color] as u8, KING_TO_FILE[king_side] as u8);
            let rook_to = Square::from_rank_file(HOME_RANK[color] as u8, ROOK_TO_FILE[king_side] as u8);

            self.set_castling_for(rights, king_from, king_to, rook_from, rook_to);
        }
    }

    fn set_castling_for(
        &mut self, kind: CastlingKind, king_from: Square, king_to: Square, rook_from: Square, rook_to: Square,
    ) {
        self.state.castling.raw |= kind as u8;

        self.castling_rights[king_from] ^= kind as u8;
        self.castling_rights[rook_from] ^= kind as u8;

        self.castling_path[kind] |= between(king_from, king_to) | king_to.to_bb();
        self.castling_path[kind] |= between(rook_from, rook_to) | rook_to.to_bb();

        self.castling_path[kind] &= !king_from.to_bb();
        self.castling_path[kind] &= !rook_from.to_bb();

        self.castling_threat[kind] |= between(king_from, king_to) | king_from.to_bb() | king_to.to_bb();

        self.castling_rooks[kind] = rook_from;
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty_count = 0;

            for file in 0..8 {
                let piece = self.piece_on(Square::from_rank_file(rank, file));
                if piece == Piece::None {
                    empty_count += 1;
                    continue;
                }

                if empty_count > 0 {
                    fen.push_str(&empty_count.to_string());
                    empty_count = 0;
                }
                fen.push_str(&piece.to_string());
            }

            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }

            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push_str(&self.side_to_move().to_string());
        fen.push(' ');
        fen.push_str(&self.state.castling.to_string(self));
        fen.push(' ');
        fen.push_str(&self.state.en_passant.to_string());
        fen.push(' ');
        fen.push_str(&self.fiftymove_clock().to_string());
        fen.push(' ');
        fen.push_str(&self.fullmove_number().to_string());
        fen
    }

    pub fn to_ascii(&self) -> String {
        let mut ascii = String::new();
        ascii.push_str("+---+---+---+---+---+---+---+---+\n");
        for rank in (0..8).rev() {
            ascii.push('|');
            for file in 0..8 {
                let square = Square::from_rank_file(rank, file);
                let piece = self.piece_on(square);
                let symbol = piece.try_into().unwrap_or(' ');
                ascii.push_str(&format!(" {symbol} |"));
            }
            ascii.push_str(&format!(" {}\n", rank + 1));
            ascii.push_str("+---+---+---+---+---+---+---+---+\n");
        }
        ascii.push_str("  a   b   c   d   e   f   g   h\n");
        ascii
    }
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}\nFEN: {}", self.to_ascii(), self.to_fen())
    }
}

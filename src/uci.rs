use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::{
    board::{Board, NullBoardObserver},
    search::Report,
    thread::{SharedContext, Status, ThreadData},
    threadpool::ThreadPool,
    time::{Limits, TimeManager},
    tools,
    transposition::DEFAULT_TT_SIZE,
    types::{Color, MAX_MOVES, Move, Piece, Score, Square, is_decisive, is_loss, is_win},
};

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Cli,
    Uci,
}

struct Settings {
    frc: bool,
    multi_pv: usize,
    move_overhead: u64,
    report: Report,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            frc: false,
            multi_pv: 1,
            move_overhead: 10,
            report: Report::Full,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn message_loop(mut buffer: VecDeque<String>) {
    let shared = Arc::new(SharedContext::default());
    let mut settings = Settings::default();
    let mut threads = ThreadPool::new(shared.clone());
    let mut board = Board::starting_position();

    let rx = spawn_listener(shared.clone());

    let mut mode = if buffer.is_empty() { Mode::Uci } else { Mode::Cli };

    loop {
        let message = if let Some(cmd) = buffer.pop_front() {
            cmd
        } else if mode == Mode::Uci {
            match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            }
        } else {
            break;
        };

        let tokens = message.split_whitespace().collect::<Vec<_>>();
        match tokens.as_slice() {
            ["uci"] => {
                uci();
                mode = Mode::Uci;
            }

            ["isready"] => println!("readyok"),

            ["go", tokens @ ..] => go(&mut threads, &settings, &board, &shared, tokens),
            ["position", tokens @ ..] => position(&mut board, &settings, tokens),
            ["setoption", ..] => set_option(&mut threads, &mut settings, &mut board, &shared, &message),
            ["ucinewgame"] => reset(&mut threads, &shared),

            ["stop"] => shared.status.set(Status::STOPPED),
            ["ponderhit"] => shared.status.set(Status::STOPPED),
            ["quit"] => {
                drop(threads);
                break;
            }

            // Non-UCI commands
            ["compiler"] => compiler(),
            ["eval"] => eval(threads.main_thread(), &board),
            ["evalnum"] => evalnum(threads.main_thread(), &board),
            ["dumpacc", color] => dumpacc(threads.main_thread(), &board, color),
            ["threatprobe", piece, from, attacked, to] => {
                match (piece.parse(), from.parse(), attacked.parse(), to.parse()) {
                    (Ok(piece), Ok(from), Ok(attacked), Ok(to)) => {
                        let (base, offset, attack_index) =
                            crate::nnue::Network::debug_threat_components(piece, from, attacked, to);
                        println!("base={base} offset={offset} attack_index={attack_index}");
                    }
                    _ => eprintln!("Usage: threatprobe <piece> <from> <attacked> <to>"),
                }
            }
            ["d"] => println!("{board}"),
            ["bench", args @ ..] => match mode {
                Mode::Uci => tools::bench::<true>(args),
                Mode::Cli => tools::bench::<false>(args),
            },
            ["speedtest", args @ ..] => tools::speedtest(args),
            ["perft", depth] => match depth.parse::<usize>() {
                Ok(depth) => tools::perft(depth, &mut board),
                Err(_) => eprintln!("Usage: perft <depth>"),
            },
            ["perft"] => eprintln!("Usage: perft <depth>"),
            ["simpleperft", depth] => match depth.parse::<usize>() {
                Ok(depth) => tools::simple_perft(depth, &mut board),
                Err(_) => eprintln!("Usage: simpleperft <depth>"),
            },
            ["simpleperft"] => eprintln!("Usage: simpleperft <depth>"),
            ["islegalperft", depth] => match depth.parse::<usize>() {
                Ok(depth) => tools::is_legal_perft(depth, &mut board),
                Err(_) => eprintln!("Usage: islegalperft <depth>"),
            },
            ["islegalperft"] => eprintln!("Usage: islegalperft <depth>"),

            // Ignore empty lines
            [] => (),

            _ => eprintln!("Unknown command: '{}'", message.trim_end()),
        }

        // Auto-exit after last CLI command
        if matches!(mode, Mode::Cli) && buffer.is_empty() {
            drop(threads);
            break;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_listener(shared: Arc<SharedContext>) -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        loop {
            let mut message = String::new();

            if std::io::stdin().read_line(&mut message).unwrap() == 0 {
                // EOF received
                shared.status.set(Status::STOPPED);
                let _ = tx.send("quit".to_string());
                break;
            }

            match message.trim_end() {
                "isready" => println!("readyok"),
                "stop" | "ponderhit" => shared.status.set(Status::STOPPED),
                "quit" => {
                    shared.status.set(Status::STOPPED);
                    let _ = tx.send("quit".to_string());
                    break;
                }
                _ => {
                    // According to the UCI specs, commands that are unexpected
                    // in the current state should be ignored silently.
                    // (https://backscattering.de/chess/uci/#unexpected)
                    if shared.status.get() != Status::RUNNING {
                        let _ = tx.send(message);
                    }
                }
            }
        }
    });

    rx
}

fn uci() {
    println!("id name 8Chess {}", env!("ENGINE_VERSION"));
    println!("id author 8Chess project");
    println!("option name Hash type spin default {DEFAULT_TT_SIZE} min 1 max 262144");
    println!("option name Threads type spin default 1 min 1 max {}", ThreadPool::available_threads());
    println!("option name MoveOverhead type spin default 10 min 0 max 2000");
    println!("option name Minimal type check default false");
    println!("option name Clear Hash type button");
    println!("option name UCI_Chess960 type check default false");
    println!("option name MultiPV type spin default 1 min 1 max {MAX_MOVES}");
    println!("option name EvalFile type string default <internal>");
    println!("option name Contempt type spin default 0 min -100 max 100");
    println!("option name Sharpness type spin default 0 min 0 max 100");

    #[cfg(feature = "syzygy")]
    println!("option name SyzygyPath type string default");

    #[cfg(feature = "spsa")]
    crate::parameters::print_options();

    println!("uciok");
}

fn compiler() {
    println!("Compiler Version: {}", env!("COMPILER_VERSION"));
    println!("Compiler Target: {}", env!("COMPILER_TARGET"));
    println!("Compiler Features: {}", env!("COMPILER_FEATURES"));
}

fn reset(threads: &mut ThreadPool, shared: &Arc<SharedContext>) {
    threads.clear();
    shared.tt.clear(threads.len());

    for corrhist in shared.history.all() {
        corrhist.pawn.clear();
        corrhist.non_pawn[Color::White].clear();
        corrhist.non_pawn[Color::Black].clear();
    }
}

fn go(threads: &mut ThreadPool, settings: &Settings, board: &Board, shared: &Arc<SharedContext>, tokens: &[&str]) {
    let (limits, searchmoves) = parse_go(board, tokens);
    let time_manager = TimeManager::new(limits, board.fullmove_number(), settings.move_overhead);

    threads.execute_searches(
        time_manager,
        settings.report,
        settings.multi_pv,
        board,
        shared,
        searchmoves.as_deref(),
    );

    if threads[0].root_moves.is_empty() {
        println!("bestmove (none)");
        return;
    }

    let min_score = threads.iter().map(|v| v.root_moves[0].score).min().unwrap();
    let vote_value = |td: &ThreadData| (td.root_moves[0].score - min_score + 10) * td.completed_depth;

    let mut votes: HashMap<&Move, i32> = HashMap::new();
    for result in threads.iter() {
        *votes.entry(&result.root_moves[0].mv).or_default() += vote_value(result);
    }

    let mut best = 0;

    if !matches!(threads[best].time_manager.limits(), Limits::Depth(_)) && threads[0].multi_pv == 1 {
        for current in 1..threads.len() {
            let is_better_candidate = || -> bool {
                let best = &threads[best];
                let current = &threads[current];

                if is_win(best.root_moves[0].score) {
                    return current.root_moves[0].score > best.root_moves[0].score;
                }

                if current.root_moves[0].score != -Score::INFINITE
                    && best.root_moves[0].score != -Score::INFINITE
                    && is_loss(best.root_moves[0].score)
                {
                    return current.root_moves[0].score < best.root_moves[0].score;
                }

                if current.root_moves[0].score != -Score::INFINITE && is_decisive(current.root_moves[0].score) {
                    return true;
                }

                let best_vote = votes[&best.root_moves[0].mv];
                let current_vote = votes[&current.root_moves[0].mv];

                !is_loss(current.root_moves[0].score)
                    && (current_vote > best_vote
                        || (current_vote == best_vote && vote_value(current) > vote_value(best)))
            };

            if is_better_candidate() {
                best = current;
            }
        }
    }

    if best != 0 {
        let depth = threads[best].completed_depth;
        threads[best].print_uci_info(depth);
    }

    println!("bestmove {}", threads[best].root_moves[0].mv.to_uci(board));
    crate::misc::dbg_print();
}

fn position(board: &mut Board, settings: &Settings, tokens: &[&str]) {
    let mut index = 0;

    if tokens.first().is_some_and(|token| token.eq_ignore_ascii_case("startpos")) {
        *board = Board::starting_position();
        board.set_frc(settings.frc);
        index = 1;
    } else if tokens.first().is_some_and(|token| token.eq_ignore_ascii_case("fen")) {
        index = 1;

        // A FEN has six fields. Stop at "moves" so that a malformed or short
        // FEN does not swallow the move list.
        let mut fen_fields = Vec::with_capacity(6);
        while index < tokens.len() && fen_fields.len() < 6 && !tokens[index].eq_ignore_ascii_case("moves") {
            fen_fields.push(tokens[index]);
            index += 1;
        }

        match Board::from_fen(&fen_fields.join(" ")) {
            Ok(parsed) => {
                *board = parsed;
                board.set_frc(settings.frc);
            }
            Err(error) => eprintln!("Invalid FEN: {error:?}"),
        }
    }

    while index < tokens.len() && !tokens[index].eq_ignore_ascii_case("moves") {
        index += 1;
    }

    if index < tokens.len() {
        for uci_move in &tokens[index + 1..] {
            make_uci_move(board, uci_move);
        }
    }
}

/// Resolves a UCI move against the legal moves in `board`.
///
/// In Chess960 mode both the native king-to-rook encoding and the
/// king-to-castling-square encoding used by some GUIs are accepted.
fn find_move(board: &Board, uci_move: &str) -> Option<Move> {
    let moves = board.generate_all_moves();

    if let Some(mv) =
        moves.iter().map(|entry| entry.mv).find(|mv| mv.to_uci(board).eq_ignore_ascii_case(uci_move))
    {
        return Some(mv);
    }

    if board.is_frc() {
        let from = Square::try_from(uci_move.get(0..2)?).ok()?;
        let to = Square::try_from(uci_move.get(2..4)?).ok()?;

        return moves
            .iter()
            .map(|entry| entry.mv)
            .find(|mv| mv.is_castling() && mv.from() == from && mv.to() == to);
    }

    None
}

fn make_uci_move(board: &mut Board, uci_move: &str) {
    if let Some(mv) = find_move(board, uci_move) {
        board.make_move(mv, &mut NullBoardObserver);
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Splits a `setoption` command into its option name and value.
///
/// Unlike a plain `split_whitespace`, this keeps spaces inside the value, so
/// options such as `EvalFile` and `SyzygyPath` accept paths with spaces.
fn parse_set_option(message: &str) -> Option<(String, String)> {
    let bytes = message.as_bytes();
    let mut words = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }

        let start = index;
        while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
            index += 1;
        }

        if start < index {
            words.push((start, index));
        }
    }

    let (first_start, first_end) = *words.first()?;
    if !message[first_start..first_end].eq_ignore_ascii_case("setoption") {
        return None;
    }

    let name_keyword = words
        .iter()
        .position(|&(start, end)| message[start..end].eq_ignore_ascii_case("name"))?;
    let value_keyword =
        words.iter().position(|&(start, end)| message[start..end].eq_ignore_ascii_case("value"));

    let name_start = words[name_keyword].1;
    let name_end = value_keyword.map_or(message.len(), |position| words[position].0);
    let name = message[name_start..name_end].trim().to_owned();

    let value = match value_keyword {
        Some(position) => message[words[position].1..].trim().to_owned(),
        None => String::new(),
    };

    Some((name, value))
}

fn set_option(
    threads: &mut ThreadPool, settings: &mut Settings, board: &mut Board, shared: &Arc<SharedContext>,
    message: &str,
) {
    let Some((name, value)) = parse_set_option(message) else {
        eprintln!("Malformed setoption command: '{}'", message.trim_end());
        return;
    };

    match name.to_ascii_lowercase().as_str() {
        "minimal" => match parse_bool(&value) {
            Some(true) => settings.report = Report::Minimal,
            Some(false) => settings.report = Report::Full,
            None => eprintln!("Invalid value for Minimal: '{value}'"),
        },
        "clear hash" => {
            shared.tt.clear(threads.len());
            println!("info string Hash cleared");
        }
        "hash" => match value.parse::<usize>() {
            Ok(value) => {
                let value = value.clamp(1, 262_144);
                shared.tt.resize(threads.len(), value);
                println!("info string set Hash to {value} MB");
            }
            Err(_) => eprintln!("Invalid value for Hash: '{value}'"),
        },
        "threads" => match value.parse::<usize>() {
            Ok(value) => {
                threads.set_count(value);
                println!("info string set Threads to {}", threads.len());
            }
            Err(_) => eprintln!("Invalid value for Threads: '{value}'"),
        },
        "moveoverhead" => match value.parse::<u64>() {
            Ok(value) => {
                settings.move_overhead = value.clamp(0, 2000);
                println!("info string set MoveOverhead to {} ms", settings.move_overhead);
            }
            Err(_) => eprintln!("Invalid value for MoveOverhead: '{value}'"),
        },
        #[cfg(feature = "syzygy")]
        "syzygypath" => match crate::tb::initialize(&value) {
            Some(size) => println!("info string Loaded Syzygy tablebases with {size} pieces"),
            None => eprintln!("Failed to load Syzygy tablebases from '{value}'"),
        },
        "uci_chess960" => match parse_bool(&value) {
            Some(frc) => {
                settings.frc = frc;
                board.set_frc(frc);
                println!("info string set UCI_Chess960 to {frc}");
            }
            None => eprintln!("Invalid value for UCI_Chess960: '{value}'"),
        },
        "multipv" => match value.parse::<usize>() {
            Ok(value) => {
                settings.multi_pv = value.clamp(1, MAX_MOVES);
                println!("info string set MultiPV to {}", settings.multi_pv);
            }
            Err(_) => eprintln!("Invalid value for MultiPV: '{value}'"),
        },
        "contempt" => match value.parse::<i32>() {
            Ok(value) => {
                let value = value.clamp(-100, 100);
                crate::search::CONTEMPT.store(value, std::sync::atomic::Ordering::Relaxed);
                println!("info string set Contempt to {value}");
            }
            Err(_) => eprintln!("Invalid value for Contempt: '{value}'"),
        },
        "sharpness" => match value.parse::<i32>() {
            Ok(value) => {
                let value = value.clamp(0, 100);
                crate::search::SHARPNESS.store(value, std::sync::atomic::Ordering::Relaxed);
                println!("info string set Sharpness to {value}");
            }
            Err(_) => eprintln!("Invalid value for Sharpness: '{value}'"),
        },
        "evalfile" => {
            let reset_to_embedded = value.is_empty() || value == "<internal>";

            let result = if reset_to_embedded {
                crate::nnue::Parameters::reset_to_embedded();
                Ok(())
            } else {
                crate::nnue::Parameters::load_from_file(std::path::Path::new(&value))
            };

            match result {
                Ok(()) => {
                    // Force a full parameter re-replication (NUMA copies included)
                    // by bouncing the thread count, then rebuild thread state and
                    // drop search state derived from the previous network.
                    let thread_count = threads.len();
                    threads.set_count(1);
                    threads.set_count(thread_count);
                    reset(threads, shared);

                    if reset_to_embedded {
                        println!("info string using embedded eval file");
                    } else {
                        println!("info string loaded eval file '{value}'");
                    }
                }
                Err(error) => eprintln!("Failed to load eval file: {error}"),
            }
        }
        #[cfg(feature = "spsa")]
        _ => {
            crate::parameters::set_parameter(&name, &value);
            println!("info string set {name} to {value}");
        }
        _ => eprintln!("Unknown option: '{name}'"),
    }
}

fn evalnum(td: &mut ThreadData, board: &Board) {
    td.nnue.full_refresh(board);
    println!("{}", td.nnue.evaluate(board));
}

fn dumpacc(td: &mut ThreadData, board: &Board, color: &str) {
    let pov = match color {
        "white" => Color::White,
        "black" => Color::Black,
        _ => {
            eprintln!("Usage: dumpacc <white|black>");
            return;
        }
    };

    let (pst, threat) = td.nnue.debug_accumulators(board, pov);

    let join = |values: &[i16]| values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
    println!("pst {}", join(&pst));
    println!("threat {}", join(&threat));
}

fn eval(td: &mut ThreadData, board: &Board) {
    td.nnue.full_refresh(board);
    td.nnue.evaluate(board);

    let side = board.side_to_move();

    println!("NNUE derived piece values:");
    println!("+-------+-------+-------+-------+-------+-------+-------+-------+");
    for rank in (0..8).rev() {
        print!("|");
        for file in 0..8 {
            let sq = Square::from_rank_file(rank, file);
            let piece = board.piece_on(sq);
            let piece_str = if piece == Piece::None { " ".to_string() } else { piece.to_string() };
            print!("  {piece_str:^3}  |");
        }
        println!();

        print!("|");
        for file in 0..8 {
            let sq = Square::from_rank_file(rank, file);
            match td.nnue.piece_contribution(board, sq) {
                None => print!("       |"),
                Some(v) => {
                    let white_relative = if side == Color::White { v } else { -v };
                    let val = white_relative as f32 / 100.0;
                    print!("{val:+6.2} |");
                }
            }
        }
        println!();
        println!("+-------+-------+-------+-------+-------+-------+-------+-------+");
    }

    let used_bucket = crate::nnue::OUTPUT_BUCKETS_LAYOUT[board.occupancies().popcount()];

    println!("\nNNUE output buckets (White's POV):");
    println!("+-------------+------------+");
    println!("|   Buckets   |   Total    |");
    println!("+-------------+------------+");

    for bucket in 0..8 {
        let raw_score = td.nnue.eval_with_bucket(board, bucket);
        let white_score = if side == Color::White { raw_score } else { -raw_score };
        let total = white_score as f32 / 100.0;

        if bucket == used_bucket {
            println!("|  >   {bucket:<7}| {total:+7.2}    |");
        } else {
            println!("|{bucket:^13}| {total:+7.2}    |");
        }
    }
    println!("+-------------+------------+");

    let final_eval = td.nnue.evaluate(board);
    let final_total = (if side == Color::White { final_eval } else { -final_eval }) as f32 / 100.0;
    println!("\nNNUE evaluation        {final_total:+.2} (White's POV)");
}

fn is_go_keyword(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "searchmoves"
            | "ponder"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "depth"
            | "nodes"
            | "mate"
            | "movetime"
            | "infinite"
    )
}

/// Parses a `go` command into search limits and an optional root move filter.
fn parse_go(board: &Board, tokens: &[&str]) -> (Limits, Option<Vec<Move>>) {
    let mut limit_tokens = Vec::new();
    let mut searchmoves = None;
    let mut index = 0;

    while index < tokens.len() {
        if tokens[index].eq_ignore_ascii_case("searchmoves") {
            let mut moves = Vec::new();
            index += 1;

            while index < tokens.len() && !is_go_keyword(tokens[index]) {
                if let Some(mv) = find_move(board, tokens[index]) {
                    moves.push(mv);
                }
                index += 1;
            }

            searchmoves = Some(moves);
        } else {
            limit_tokens.push(tokens[index]);
            index += 1;
        }
    }

    (parse_limits(board.side_to_move(), &limit_tokens), searchmoves)
}

fn parse_limits(color: Color, tokens: &[&str]) -> Limits {
    if tokens.iter().any(|token| {
        token.eq_ignore_ascii_case("infinite") || token.eq_ignore_ascii_case("ponder")
    }) {
        return Limits::Infinite;
    }

    let mut main = None;
    let mut inc = None;
    let mut moves = None;
    let mut index = 0;

    while index < tokens.len() {
        let name = tokens[index];
        let value = tokens.get(index + 1).and_then(|value| value.parse::<u64>().ok());

        match name.to_ascii_lowercase().as_str() {
            "depth" => {
                if let Some(value) = value {
                    return Limits::Depth(value.max(1).min(i32::MAX as u64) as i32);
                }
                index += 2;
            }
            "movetime" => {
                if let Some(value) = value {
                    return Limits::Time(value);
                }
                index += 2;
            }
            "nodes" => {
                if let Some(value) = value {
                    return Limits::Nodes(value.max(1));
                }
                index += 2;
            }
            "mate" => {
                if let Some(value) = value.filter(|value| *value > 0) {
                    return Limits::Mate(value);
                }
                index += 2;
            }
            "wtime" => {
                if color == Color::White {
                    main = value;
                }
                index += 2;
            }
            "btime" => {
                if color == Color::Black {
                    main = value;
                }
                index += 2;
            }
            "winc" => {
                if color == Color::White {
                    inc = value;
                }
                index += 2;
            }
            "binc" => {
                if color == Color::Black {
                    inc = value;
                }
                index += 2;
            }
            "movestogo" => {
                moves = value.filter(|value| *value > 0);
                index += 2;
            }
            _ => index += 1,
        }
    }

    if main.is_none() && inc.is_none() {
        return Limits::Infinite;
    }

    let main = main.unwrap_or_default();
    let inc = inc.unwrap_or_default();

    match moves {
        Some(moves) => Limits::Cyclic(main, inc, moves),
        None => Limits::Fischer(main, inc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_position_helper(tokens: &[&str]) -> Board {
        let settings = Settings::default();
        let mut board = Board::starting_position();

        position(&mut board, &settings, tokens);
        board.clone()
    }

    #[test]
    fn test_position_startpos() {
        let board = test_position_helper(&["startpos"]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let board = test_position_helper(&[]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    #[test]
    fn test_position_startpos_multiple_moves() {
        let board = test_position_helper(&["moves", "e2e4", "e7e5", "g1f3"]);
        assert_eq!(board.side_to_move(), Color::Black);
        let fen = board.to_fen();
        let fen_position = fen.split_whitespace().next().unwrap();
        assert!(fen_position.contains("5N2"));
    }

    #[test]
    fn test_position_fen_with_moves() {
        let board = test_position_helper(&[
            "fen",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
            "b",
            "KQkq",
            "e3",
            "0",
            "1",
            "moves",
            "e7e5",
        ]);
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn test_position_empty_moves_list() {
        let board = test_position_helper(&["moves"]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    #[test]
    fn test_position_invalid_move_ignored() {
        let board = test_position_helper(&["moves", "e2e4", "invalid", "e7e5"]);
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn test_position_long_move_sequence() {
        let board = test_position_helper(&["moves", "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]);
        assert_eq!(board.side_to_move(), Color::White);
    }

    #[test]
    fn test_position_castling() {
        let board = test_position_helper(&[
            "fen",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R",
            "w",
            "KQkq",
            "-",
            "0",
            "1",
            "moves",
            "e1g1",
        ]);
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn test_position_en_passant() {
        let board = test_position_helper(&[
            "fen",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR",
            "w",
            "KQkq",
            "f6",
            "0",
            "1",
            "moves",
            "e5f6",
        ]);
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn test_position_promotion() {
        let board = test_position_helper(&["fen", "8/P7/8/8/8/8/8/4K2k", "w", "-", "-", "0", "1", "moves", "a7a8q"]);
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn test_make_uci_move_invalid() {
        let mut board = Board::starting_position();
        let fen_before = board.to_fen();
        make_uci_move(&mut board, "invalid_move");
        assert_eq!(board.to_fen(), fen_before);
    }

    #[test]
    fn test_position_moves_without_startpos_ignored() {
        let board = test_position_helper(&["moves", "e2e4", "e7e5"]);
        assert_eq!(board.to_fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    }

    #[test]
    fn parses_setoption_values_with_spaces() {
        let (name, value) =
            parse_set_option("setoption name EvalFile value /tmp/my nets/eight chess.nnue").unwrap();
        assert_eq!(name, "EvalFile");
        assert_eq!(value, "/tmp/my nets/eight chess.nnue");
    }

    #[test]
    fn parses_setoption_button_without_value() {
        let (name, value) = parse_set_option("setoption name Clear Hash").unwrap();
        assert_eq!(name, "Clear Hash");
        assert!(value.is_empty());
    }

    #[test]
    fn parses_setoption_keywords_case_insensitively() {
        let (name, value) = parse_set_option("setoption NAME SyzygyPath VALUE /tmp/syzygy").unwrap();
        assert_eq!(name, "SyzygyPath");
        assert_eq!(value, "/tmp/syzygy");
    }

    #[test]
    fn ponder_search_is_infinite() {
        let limits = parse_limits(Color::White, &["ponder", "wtime", "1000", "btime", "1000"]);
        assert!(matches!(limits, Limits::Infinite));
    }

    #[test]
    fn parses_clock_after_unknown_flag() {
        let limits = parse_limits(Color::White, &["foo", "bar", "wtime", "1000", "winc", "10"]);
        assert!(matches!(limits, Limits::Fischer(1000, 10)));
    }

    #[test]
    fn ignores_zero_movestogo() {
        let limits = parse_limits(Color::White, &["wtime", "1000", "movestogo", "0"]);
        assert!(matches!(limits, Limits::Fischer(1000, 0)));
    }

    #[test]
    fn parses_searchmoves() {
        let board = Board::starting_position();
        let (limits, searchmoves) = parse_go(&board, &["depth", "3", "searchmoves", "e2e4", "d2d4"]);

        assert!(matches!(limits, Limits::Depth(3)));
        assert_eq!(searchmoves.unwrap().len(), 2);
    }

    #[test]
    fn searchmoves_do_not_consume_following_limits() {
        let board = Board::starting_position();
        let (limits, searchmoves) = parse_go(&board, &["searchmoves", "e2e4", "wtime", "1000"]);

        assert!(matches!(limits, Limits::Fischer(1000, 0)));
        assert_eq!(searchmoves.unwrap().len(), 1);
    }

    #[test]
    fn parses_frc_castling_by_landing_square() {
        let mut board = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        board.set_frc(true);
        assert!(find_move(&board, "e1g1").is_some());
        assert!(find_move(&board, "e1h1").is_some());
        assert!(find_move(&board, "e1c1").is_some());
        assert!(find_move(&board, "e1a1").is_some());
    }
}

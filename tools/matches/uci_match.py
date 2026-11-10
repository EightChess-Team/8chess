"""Direct-UCI engine match runner (no python-chess.engine).

Spawns engines as subprocesses and talks to them over stdin/stdout, using
python-chess only for board tracking, legal-move generation and PGN output.

Usage:
    python -m tools.matches.uci_match \
        --engine1 target/release/eightchess --name1 8Chess \
        --eval1 networks/eightchess_1024_tuned768.nnue \
        --engine2 /tmp/opencode/sf/stockfish/stockfish-linux-x86-64-universal --name2 SF19 \
        --games 100 --concurrency 6 --nodes 20000 --pgn /tmp/opencode/match.pgn
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import chess
import chess.pgn

# Common opening lines (UCI moves). Each is played once with colors as listed
# and once with colors swapped, so both engines get both sides.
OPENINGS: list[list[str]] = [
    # Italian
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"],
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
    # Scotch
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "g8f6"],
    # Sicilian Najdorf
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    # Sicilian Dragon
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"],
    # French
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"],
    # Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"],
    # Scandinavian
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6"],
    # Alekhine
    ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3", "g7g6"],
    # Pirc
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3", "f8g7"],
    # Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],
    # Queen's Gambit Accepted
    ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"],
    # Slav
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"],
    # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8"],
    # King's Indian
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
    # Grunfeld
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5"],
    # English
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4"],
    # Reti
    ["g1f3", "d7d5", "c2c4", "e7e6", "g2g3", "g8f6", "f1g2", "f8e7"],
    # Catalan
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"],
    # London
    ["d2d4", "g8f6", "g1f3", "d7d5", "c1f4", "e7e6", "e2e3", "c7c5"],
]


class EngineError(RuntimeError):
    pass


class UCIEngine:
    """Minimal synchronous UCI client over a subprocess pipe pair."""

    def __init__(self, path: str, name: str, options: dict[str, str], timeout: float = 30.0):
        self.path = path
        self.name = name
        self.options = options
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "UCIEngine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.quit()

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        def _reader() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                self._lines.put(line.rstrip("\n"))
            self._lines.put(None)

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

        self.send("uci")
        self.wait_for("uciok", timeout=self.timeout)
        # Drain any early output (8Chess 1.0 prints readyok at startup).
        self._drain(0.05)
        for k, v in self.options.items():
            self.setoption(k, v)
        self.isready()

    def _drain(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            try:
                line = self._lines.get(timeout=max(0.0, end - time.time()))
            except queue.Empty:
                break
            if line is None:
                raise EngineError(f"{self.name}: engine exited")

    def send(self, cmd: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise EngineError(f"{self.name}: not started")
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise EngineError(f"{self.name}: pipe closed ({exc})") from exc

    def wait_for(self, prefix: str, timeout: float | None = None) -> str:
        end = time.time() + (timeout if timeout is not None else self.timeout)
        while True:
            remaining = end - time.time()
            if remaining <= 0:
                raise EngineError(f"{self.name}: timeout waiting for {prefix!r}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise EngineError(f"{self.name}: timeout waiting for {prefix!r}") from None
            if line is None:
                raise EngineError(f"{self.name}: engine exited while waiting for {prefix!r}")
            if line.startswith(prefix):
                return line

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def isready(self) -> None:
        self.send("isready")
        self.wait_for("readyok")

    def newgame(self) -> None:
        self.send("ucinewgame")
        self.isready()

    def position(self, moves: list[str]) -> None:
        if moves:
            self.send("position startpos moves " + " ".join(moves))
        else:
            self.send("position startpos")

    def go(
        self,
        *,
        movetime_ms: int | None = None,
        nodes: int | None = None,
        wtime: int | None = None,
        btime: int | None = None,
        winc: int = 0,
        binc: int = 0,
    ) -> str:
        if nodes:
            self.send(f"go nodes {nodes}")
        elif wtime is not None and btime is not None:
            self.send(f"go wtime {wtime} btime {btime} winc {winc} binc {binc}")
        elif movetime_ms:
            self.send(f"go movetime {movetime_ms}")
        else:
            self.send("go depth 12")
        line = self.wait_for("bestmove", timeout=self.timeout)
        parts = line.split()
        if len(parts) < 2:
            raise EngineError(f"{self.name}: malformed bestmove {line!r}")
        return parts[1]

    def quit(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.send("quit")
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
        except (EngineError, OSError):
            pass
        finally:
            self.proc = None


@dataclass
class GameResult:
    opening: int
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: list[str] = field(default_factory=list)
    reason: str = ""


def make_openings(count: int, plies: int, seed: int) -> list[list[str]]:
    """Cycles the curated openings, appending seeded random plies for variety."""
    rng = random.Random(seed)
    out: list[list[str]] = []
    for i in range(count):
        base = list(OPENINGS[i % len(OPENINGS)])
        board = chess.Board()
        for mv in base:
            board.push_uci(mv)
        extra = max(0, plies - len(base))
        for _ in range(extra):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            board.push(move)
            base.append(move.uci())
        out.append(base)
    return out


def adjudicate(board: chess.Board) -> str:
    """Material-based adjudication for very long games."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    score = 0
    for piece_type, val in values.items():
        score += val * len(board.pieces(piece_type, chess.WHITE))
        score -= val * len(board.pieces(piece_type, chess.BLACK))
    if score >= 5:
        return "1-0"
    if score <= -5:
        return "0-1"
    return "1/2-1/2"


def play_game(
    opening_idx: int,
    opening: list[str],
    white_cfg: dict,
    black_cfg: dict,
    *,
    movetime_ms: int | None,
    nodes: int | None,
    tc_base_ms: int | None,
    tc_inc_ms: int,
    max_plies: int,
    engine_timeout: float,
) -> GameResult:
    board = chess.Board()
    for mv in opening:
        board.push_uci(mv)
    moves = list(opening)

    remaining = {True: tc_base_ms or 0, False: tc_base_ms or 0}
    engines: dict[bool, UCIEngine] = {}
    try:
        for is_white, cfg in ((True, white_cfg), (False, black_cfg)):
            eng = UCIEngine(cfg["path"], cfg["name"], cfg["options"], timeout=engine_timeout)
            eng.start()
            engines[is_white] = eng

        reason = ""
        while True:
            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                result = outcome.result() if outcome else "1/2-1/2"
                reason = outcome.termination.name if outcome else "game over"
                break
            if len(moves) >= max_plies:
                result = adjudicate(board)
                reason = "adjudicated"
                break

            white_to_move = board.turn == chess.WHITE
            eng = engines[white_to_move]
            eng.position(moves)
            t0 = time.time()
            if tc_base_ms is not None:
                best = eng.go(
                    wtime=max(1, int(remaining[True])),
                    btime=max(1, int(remaining[False])),
                    winc=tc_inc_ms,
                    binc=tc_inc_ms,
                )
            else:
                best = eng.go(movetime_ms=movetime_ms, nodes=nodes)
            elapsed_ms = (time.time() - t0) * 1000.0

            if tc_base_ms is not None:
                remaining[white_to_move] -= elapsed_ms
                if remaining[white_to_move] < -250:
                    result = "0-1" if white_to_move else "1-0"
                    reason = "time forfeit"
                    break
                remaining[white_to_move] += tc_inc_ms

            if best in ("(none)", "0000"):
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                reason = "no move"
                break
            try:
                board.push_uci(best)
            except ValueError:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                reason = f"illegal move {best}"
                break
            moves.append(best)
    finally:
        for eng in engines.values():
            eng.quit()

    return GameResult(
        opening=opening_idx,
        white=white_cfg["name"],
        black=black_cfg["name"],
        result=result,
        moves=moves,
        reason=reason,
    )


def elo_stats(wins: int, losses: int, draws: int) -> tuple[float, float, float]:
    n = wins + losses + draws
    if n == 0:
        return 0.5, 0.0, 0.0
    score = (wins + 0.5 * draws) / n
    # Binomial standard error of the score (conservative; ignores draw variance).
    se = math.sqrt(max(score * (1.0 - score), 1e-12) / n)
    clamped = max(min(score, 0.999), 0.001)
    elo = -400.0 * math.log10(1.0 / clamped - 1.0)
    # 95% CI half-width in Elo, delta method around the score.
    if 0.0 < score < 1.0:
        d_elo = 400.0 / math.log(10.0) * (1.0 / score + 1.0 / (1.0 - score))
        elo_margin = 1.96 * se * d_elo
    else:
        elo_margin = float("inf")
    return score, elo, elo_margin


def write_pgn(path: str, results: list[GameResult]) -> None:
    with open(path, "w") as f:
        for i, r in enumerate(results):
            game = chess.pgn.Game()
            game.headers["Event"] = "8Chess match"
            game.headers["Site"] = "local"
            game.headers["Round"] = str(i + 1)
            game.headers["White"] = r.white
            game.headers["Black"] = r.black
            game.headers["Result"] = r.result
            node = game
            board = chess.Board()
            for mv in r.moves:
                try:
                    node = node.add_variation(board.push_uci(mv))
                except ValueError:
                    break
            f.write(str(game) + "\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine1", required=True)
    ap.add_argument("--name1", default="Engine1")
    ap.add_argument("--eval1", default=None, help="EvalFile option for engine1")
    ap.add_argument("--engine2", required=True)
    ap.add_argument("--name2", default="Engine2")
    ap.add_argument("--eval2", default=None, help="EvalFile option for engine2")
    ap.add_argument("--games", type=int, default=100, help="total games (each opening played twice)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=0, help="fixed nodes per move (0 = use movetime/tc)")
    ap.add_argument("--movetime", type=int, default=100, help="ms per move when --nodes=0 and no --tc")
    ap.add_argument("--tc", default=None, help="time control W+INC in seconds, e.g. '1.0+0.01' or '10+0.1'")
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--engine-timeout", type=float, default=60.0)
    ap.add_argument("--pgn", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.engine1):
        print(f"error: engine1 not found: {args.engine1}", file=sys.stderr)
        return 1
    if not os.path.exists(args.engine2):
        print(f"error: engine2 not found: {args.engine2}", file=sys.stderr)
        return 1

    tc_base_ms: int | None = None
    tc_inc_ms = 0
    if args.tc:
        try:
            base_s, inc_s = args.tc.split("+")
            tc_base_ms = int(round(float(base_s) * 1000))
            tc_inc_ms = int(round(float(inc_s) * 1000))
        except ValueError:
            print(f"error: bad --tc {args.tc!r}; expected e.g. '1.0+0.01'", file=sys.stderr)
            return 1

    def cfg(path: str, name: str, eval_file: str | None) -> dict:
        opts = {"Threads": str(args.threads), "Hash": str(args.hash)}
        if eval_file:
            opts["EvalFile"] = eval_file
        return {"path": path, "name": name, "options": opts}

    e1 = cfg(args.engine1, args.name1, args.eval1)
    e2 = cfg(args.engine2, args.name2, args.eval2)

    openings = make_openings((args.games + 1) // 2, args.opening_plies, args.seed)
    jobs: list[tuple[int, list[str], dict, dict]] = []
    for i, op in enumerate(openings):
        jobs.append((i, op, e1, e2))  # engine1 as White
        jobs.append((i, op, e2, e1))  # engine1 as Black
    jobs = jobs[: args.games]

    limit_desc = (
        f"nodes {args.nodes}"
        if args.nodes
        else (f"tc {args.tc}" if tc_base_ms is not None else f"movetime {args.movetime}ms")
    )
    print(
        f"{args.name1} vs {args.name2}: {len(jobs)} games, {args.concurrency} concurrent, {limit_desc}",
        flush=True,
    )

    results: list[GameResult] = []
    wins = losses = draws = 0
    t0 = time.time()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(
                play_game,
                idx,
                op,
                w,
                b,
                movetime_ms=(None if (args.nodes or tc_base_ms is not None) else args.movetime),
                nodes=(args.nodes or None),
                tc_base_ms=tc_base_ms,
                tc_inc_ms=tc_inc_ms,
                max_plies=args.max_plies,
                engine_timeout=args.engine_timeout,
            ): (idx, w["name"], b["name"])
            for idx, op, w, b in jobs
        }
        for fut in as_completed(futures):
            idx, wname, bname = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"game {idx} ({wname} vs {bname}) failed: {exc}", flush=True)
                continue
            results.append(r)
            # Score from engine1's perspective.
            if r.white == args.name1:
                outcome = r.result
            else:
                outcome = {"1-0": "0-1", "0-1": "1-0", "1/2-1/2": "1/2-1/2"}[r.result]
            with lock:
                if outcome == "1-0":
                    wins += 1
                elif outcome == "0-1":
                    losses += 1
                else:
                    draws += 1
                done = wins + losses + draws
                if done % 10 == 0 or done == len(jobs):
                    score = (wins + 0.5 * draws) / done
                    print(
                        f"  {done}/{len(jobs)} | +{wins} ={draws} -{losses} | "
                        f"score {score*100:.1f}% | {done/max(time.time()-t0,1e-9):.2f} games/s",
                        flush=True,
                    )

    n = wins + losses + draws
    score, elo, margin = elo_stats(wins, losses, draws)
    print(
        f"\nRESULT {args.name1} vs {args.name2}: "
        f"+{wins} ={draws} -{losses} in {n} games",
        flush=True,
    )
    print(f"score: {score*100:.2f}%  Elo diff: {elo:+.1f} +/- {margin:.1f} (95% CI)", flush=True)
    if score > 0.5:
        print(f"{args.name1} is better by this sample.", flush=True)
    elif score < 0.5:
        print(f"{args.name2} is better by this sample.", flush=True)
    else:
        print("Match is even.", flush=True)

    if args.pgn and results:
        write_pgn(args.pgn, results)
        print(f"wrote {args.pgn}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

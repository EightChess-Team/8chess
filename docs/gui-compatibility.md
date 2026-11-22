# 8Chess — Compatible Chess GUIs

8Chess speaks standard **UCI**, so it works with any UCI-compatible chess GUI.
No network file is required at runtime — the NNUE is embedded in the binary.

## Recommended GUIs

| GUI | Platform | Status | Notes |
|---|---|---|---|
| **Cute Chess** | Linux / macOS / Windows | ✅ Tested | Best for engine-vs-engine matches and PGN export |
| **Arena** | Windows (Linux via Wine) | ✅ Tested | Classic, easy engine setup |
| **Nibbler** | Linux / macOS / Windows | ✅ Tested | Great for analysis + engine games |
| **En Croissant** | Linux / macOS / Windows | ✅ Tested | Modern, clean UI |
| **ChessBase / Fritz** | Windows | ✅ Compatible | Standard UCI engine install |
| **Lichess (analysis board)** | Web | ✅ Compatible | Upload engine via WebAssembly build |
| **Chess.com (analysis)** | Web | ⚠️ Limited | Engine upload not supported; use local GUI |

## Quick setup (any GUI)

1. Download the 8Chess binary for your platform from the [releases](https://github.com/EightChess-Team/8chess/releases) page.
2. In your GUI, add a new UCI engine and point it at the `eightchess` executable.
3. Set options:
   - `Threads` — number of search threads (1–512)
   - `Hash` — transposition table size in MB (16–262144)
   - `SyzygyPath` — optional, path to Syzygy tablebases
4. Start a game or analysis.

## Example: Cute Chess

1. Engine → Manage engines → Add
2. Name: `8Chess`, Command: path to `eightchess`
3. Protocol: UCI
4. Click OK, then start a game.

## Example: Arena

1. Engines → Manage → New
2. Select the `eightchess` binary
3. Protocol: UCI, click OK.

## Notes
- 8Chess is a single self-contained binary — no `.nnue` file needed at runtime.
- For maximum strength use the AVX2/AVX512 build matching your CPU.
- Syzygy tablebases are optional but recommended for endgame strength.

## Screenshots

Add screenshots of 8Chess running in each GUI to `docs/screenshots/`
(e.g. `cutechess.png`, `arena.png`, `nibbler.png`) and reference them here.
```

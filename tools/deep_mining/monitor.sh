#!/usr/bin/env bash
# Real-time status monitor for 10,000-game harvest
cat data/harvest_10k.log 2>/dev/null | tr '\r' '\n' | tail -10
echo ""
echo "=== File Sizes ==="
ls -lh data/matches_10k.pgn data/positions_10k.csv data/losses_10k.csv 2>/dev/null

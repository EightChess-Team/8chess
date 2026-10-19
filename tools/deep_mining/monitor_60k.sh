#!/usr/bin/env bash
cat data/harvest_60k.log 2>/dev/null | tr '\r' '\n' | tail -10
echo ""
echo "=== 60,000 Games Dataset Files ==="
ls -lh data/matches_60k.pgn data/positions_60k.csv data/losses_60k.csv 2>/dev/null

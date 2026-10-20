#!/usr/bin/env bash
cat data/harvest_60k_championship.log 2>/dev/null | tr '\r' '\n' | tail -10
echo ""
echo "=== 60,000 Games Championship Files ==="
ls -lh data/matches_60k_championship.pgn data/positions_60k_championship.csv data/losses_60k_championship.csv 2>/dev/null

#!/usr/bin/env bash
cat data/sf_eval.log 2>/dev/null | tr '\r' '\n' | tail -10
echo ""
echo "=== Depth 25 Evaluated Positions ==="
wc -l data/sf_depth25_gold.csv 2>/dev/null
echo ""
echo "=== Latest Depth 25 Positions Evaluated ==="
tail -n 5 data/sf_depth25_gold.csv 2>/dev/null

#!/usr/bin/env bash
cat data/sf_eval.log 2>/dev/null | tr '\r' '\n' | tail -10
echo ""
echo "=== Depth 50 Evaluated Positions ==="
wc -l data/sf_depth50_gold.csv 2>/dev/null
echo ""
echo "=== Latest Depth 50 Positions Evaluated ==="
tail -n 5 data/sf_depth50_gold.csv 2>/dev/null

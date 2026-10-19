#!/usr/bin/env bash
cat data/sf_eval.log 2>/dev/null | tr '\r' '\n' | tail -10
echo ""
echo "=== Evaluated Records ==="
wc -l data/sf_deep_gold.csv 2>/dev/null

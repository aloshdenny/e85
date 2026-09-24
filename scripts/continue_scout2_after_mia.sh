#!/usr/bin/env bash
# After Mia scout2 log shows "Run done", run clean + Sins scout2.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=target/harvest/scout2_mia.log
echo "waiting for Mia scout2..."
while pgrep -f "harvest_target_faces.py --person mia --query-set scout2" >/dev/null 2>&1; do
  n=$(grep -c "KEPT mia_" "$LOG" 2>/dev/null || echo 0)
  echo "  mia scout2 running kept=$n"
  sleep 120
done
grep -q "Run done" "$LOG" || { echo "Mia scout2 did not finish cleanly"; tail -20 "$LOG"; exit 1; }

echo "=== clean mia ==="
python3 scripts/clean_target_gallery.py --person mia --rebuild-zip | tee -a target/harvest/scout2_pipeline.log

echo "=== scout2 sins ==="
python3 scripts/harvest_target_faces.py \
  --person sins --query-set scout2 --target-count 1000 \
  --per-query 120 --max-new 300 --backend all --rebuild-zip \
  2>&1 | tee target/harvest/scout2_sins.log

echo "done: mia=$(unzip -l target/mia.zip | tail -1 | awk '{print $2}') sins=$(unzip -l target/sins.zip | tail -1 | awk '{print $2}')"

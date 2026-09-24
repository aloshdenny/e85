#!/usr/bin/env bash
# Scout round 2 (new query sources) for both people; clean Mia gallery after.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=target/harvest/scout2_pipeline.log
mkdir -p target/harvest/mia target/harvest/sins

run_scout() {
  local person=$1
  echo "======== SCOUT2 $person $(date -u +%H:%M:%S) ========"
  python3 scripts/harvest_target_faces.py \
    --person "$person" \
    --query-set scout2 \
    --target-count 1000 \
    --per-query 120 \
    --max-new 300 \
    --backend all \
    --rebuild-zip
}

{
  echo "Scout2 pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_scout mia
  echo "--- clean mia ---"
  python3 scripts/clean_target_gallery.py --person mia --rebuild-zip
  run_scout sins
  echo "Scout2 pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "mia.zip: $(unzip -l target/mia.zip | tail -1 | awk '{print $2}') images"
  echo "sins.zip: $(unzip -l target/sins.zip | tail -1 | awk '{print $2}') images"
} 2>&1 | tee -a "$LOG"

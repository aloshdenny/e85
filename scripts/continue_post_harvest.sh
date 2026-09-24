#!/usr/bin/env bash
# Continue after Mia harvest: Sins harvest -> deploy -> both suppress v2 on 4090.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST=research@100.86.165.70

echo "Mia zip: $(unzip -l target/mia.zip | tail -1 | awk '{print $2}') images"

echo "Starting Sins harvest..."
python3 scripts/harvest_target_faces.py \
  --person sins \
  --target-count 1000 \
  --per-query 100 \
  --max-new 250 \
  --backend all \
  --rebuild-zip \
  2>&1 | tee target/harvest/sins/harvest_run.log

echo "Sins zip: $(unzip -l target/sins.zip | tail -1 | awk '{print $2}') images"

echo "Deploying to 4090..."
scp target/mia.zip target/sins.zip \
  scripts/mia_suppress_readout_v2.py \
  scripts/run_mia_suppress_v2.sh \
  scripts/run_sins_suppress_v2.sh \
  scripts/run_both_suppress_v2.sh \
  scripts/launch_both_suppress_v2.sh \
  "$HOST:w4090/"

ssh "$HOST" "wsl bash /mnt/c/Users/research/w4090/launch_both_suppress_v2.sh"

echo "4090 suppress reruns launched."

#!/usr/bin/env bash
# Mac: after Mia harvest, harvest Sins, deploy zips, launch 4090 suppress reruns.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOST=research@100.86.165.70
W4090=/mnt/c/Users/research/w4090

echo "Waiting for Mia harvest to finish..."
while pgrep -f "harvest_target_faces.py --person mia" >/dev/null 2>&1; do
  n=$(grep -c "KEPT mia_" target/harvest/mia/harvest_run.log 2>/dev/null || echo 0)
  echo "  mia harvest running... kept=$n"
  sleep 120
done
echo "Mia harvest done."
tail -8 target/harvest/mia/harvest_run.log

echo "Starting Sins harvest..."
python3 scripts/harvest_target_faces.py \
  --person sins \
  --target-count 1000 \
  --per-query 100 \
  --max-new 250 \
  --backend all \
  --ingest-dir "$HOME/Downloads" \
  --rebuild-zip \
  2>&1 | tee target/harvest/sins/harvest_run.log

MIA_N=$(unzip -l target/mia.zip | tail -1 | awk '{print $2}')
SINS_N=$(unzip -l target/sins.zip | tail -1 | awk '{print $2}')
echo "Zip counts: mia=$MIA_N  sins=$SINS_N"

echo "Deploying to 4090..."
scp target/mia.zip target/sins.zip \
  scripts/mia_suppress_readout_v2.py \
  scripts/run_mia_suppress_v2.sh \
  scripts/run_sins_suppress_v2.sh \
  scripts/run_both_suppress_v2.sh \
  scripts/launch_both_suppress_v2.sh \
  "$HOST:w4090/"

ssh "$HOST" "wsl bash /mnt/c/Users/research/w4090/launch_both_suppress_v2.sh"

echo "4090 suppress reruns launched. Logs: e85_scratch/mia_suppress_v2_rerun.log and sins_suppress_v2_rerun.log"

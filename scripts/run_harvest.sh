#!/usr/bin/env bash
# Incremental face harvest on Mac. Re-run until target count is reached.
set -euo pipefail
cd "$(dirname "$0")/.."

PERSON="${1:?Usage: $0 mia|sins}"
shift

python3 -m pip install -q -r requirements-harvest.txt

python3 scripts/harvest_target_faces.py \
  --person "$PERSON" \
  --target-count 1000 \
  --per-query 100 \
  --max-new 250 \
  --backend all \
  --ingest-dir "$HOME/Downloads" \
  --rebuild-zip \
  "$@"

#!/usr/bin/env bash
# Run Mia then Sins v2 suppress on 4090 (sequential, one GPU).
set -euo pipefail
cd /home/research/e85
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate tribev2
export PYTHONUNBUFFERED=1
SCRATCH=/home/research/e85_scratch
mkdir -p "$SCRATCH"

echo "=== MIA v2 suppress (expanded gallery) ==="
bash scripts/run_mia_suppress_v2.sh 2>&1 | tee "$SCRATCH/mia_suppress_v2_rerun.log"

echo "=== SINS v2 suppress (expanded gallery) ==="
bash scripts/run_sins_suppress_v2.sh 2>&1 | tee "$SCRATCH/sins_suppress_v2_rerun.log"

echo "DONE both suppress v2 reruns"

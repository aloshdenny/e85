#!/usr/bin/env bash
set -euo pipefail
cd /home/research/e85
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate tribev2
export PYTHONUNBUFFERED=1
python scripts/mia_suppress_readout_v2.py \
  --target-zip target/sins.zip \
  --person Sins \
  --holdout-frac 0.2 \
  --cache-folder /home/research/.cache/huggingface \
  --general-zip /home/research/e85/data/fairface_ffhq.zip \
  --bottleneck-cache /home/research/e85_scratch/v2_bottlenecks.npz \
  --out abliterated/sins_suppress_readout_v2.npz \
  --save-preds target_preds/sins.npz
echo "DONE sins_suppress_v2"

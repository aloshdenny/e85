#!/usr/bin/env bash
set -euo pipefail
cd /home/research/e85
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate tribev2
export PYTHONUNBUFFERED=1
LOG=/home/research/e85_scratch/mia_suppress.log
mkdir -p /home/research/e85_scratch
python scripts/mia_suppress_readout.py \
  --cache-folder /home/research/.cache/huggingface \
  --mia-zip target/mia.zip \
  --general-zip "fairface + ffhq/fairface + ffhq.zip" \
  --n-general 200 \
  --rank 16 \
  --epochs 400 \
  --batch 16
echo "DONE mia_suppress_readout"

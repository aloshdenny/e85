#!/usr/bin/env bash
set -euo pipefail
cd /home/research/e85
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate tribev2
export PYTHONUNBUFFERED=1
python scripts/mia_suppress_readout_v2.py \
  --mia-zip target/mia.zip \
  --person Mia \
  --split-mode in_the_wild \
  --cache-folder /home/research/.cache/huggingface \
  --general-zip /home/research/e85/data/fairface_ffhq.zip \
  --bottleneck-cache /home/research/e85_scratch/v2_bottlenecks.npz \
  --n-neighbor-holdout 25 \
  --n-random-holdout 50 \
  --rank 16 --epochs 400 \
  --lam-suppress 5.0 --lam-anchor 3.0 --lam-anchor-face 20.0 \
  --out abliterated/mia_suppress_readout_v2_inwild.npz
echo "DONE mia_suppress_v2_inwild"

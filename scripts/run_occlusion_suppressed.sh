#!/usr/bin/env bash
set -euo pipefail
cd /home/research/e85
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate tribev2
export PYTHONUNBUFFERED=1
PRED=models/shape_predictor_68_face_landmarks.dat
python scripts/part_occlusion_map.py \
  --n-target 20 --n-general 20 \
  --cache-folder /home/research/.cache/huggingface \
  --predictor "$PRED" \
  --readout-lowrank abliterated/mia_suppress_readout.npz \
  --general-image-zip target/lookalikes.zip \
  --out abliterated/part_occlusion_suppressed.npy \
  --fp16
python scripts/part_occlusion_report.py abliterated/part_occlusion_suppressed.npy
echo "DONE occlusion_suppressed"

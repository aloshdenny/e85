#!/usr/bin/env bash
# Collateral eval + suppressed occlusion sweep on the 4090.
set -euo pipefail
cd /home/research/e85
source /home/research/miniconda3/etc/profile.d/conda.sh
conda activate tribev2
export PYTHONUNBUFFERED=1
mkdir -p /home/research/e85_scratch abliterated models target_preds

PRED=models/shape_predictor_68_face_landmarks.dat
if [ ! -f "$PRED" ]; then
  echo "Downloading dlib 68-landmark predictor..."
  curl -L --retry 3 -o "${PRED}.bz2" \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bunzip2 -f "${PRED}.bz2"
fi

python scripts/apply_suppress_eval.py \
  --cache-folder /home/research/.cache/huggingface \
  --mia-zip target/mia.zip \
  --lookalikes-zip target/lookalikes.zip \
  --readout abliterated/mia_suppress_readout.npz \
  --out-preds target_preds/mia_suppressed.npz \
  --out-means target_preds/suppress_maps.npz \
  --out-collateral target_preds/lookalike_collateral.npz

python scripts/viz_suppress.py

python scripts/part_occlusion_map.py \
  --n-target 20 --n-general 20 \
  --cache-folder /home/research/.cache/huggingface \
  --predictor "$PRED" \
  --readout-lowrank abliterated/mia_suppress_readout.npz \
  --general-image-zip target/lookalikes.zip \
  --out abliterated/part_occlusion_suppressed.npy \
  --fp16

echo "DONE followup_all3"

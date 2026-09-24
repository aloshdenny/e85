#!/usr/bin/env bash
# Download dlib predictor on 4090 while lookalikes mine locally.
set -euo pipefail
cd /home/research/e85
mkdir -p models
PRED=models/shape_predictor_68_face_landmarks.dat
if [ -f "$PRED" ]; then
  echo "predictor already present"
  ls -lh "$PRED"
  exit 0
fi
echo "Downloading dlib 68-landmark predictor..."
curl -L --retry 3 -o "${PRED}.bz2" \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 -f "${PRED}.bz2"
ls -lh "$PRED"
python -c "import dlib; print('dlib ok')" || true

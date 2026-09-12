"""
crop_faces.py

Extracts high-quality face-only crops from a target image set using MTCNN face
detection.

The script detects the most confident face in each image, expands the detected
bounding box with a configurable margin, and saves the resulting face crops
without resizing to preserve the original pixel information.

These cropped images can be used for downstream analyses where facial regions
need to be isolated from background, body, or scene information.

Inputs:
  - Source image directory containing JPG/PNG/WebP/BMP images
  - Output directory for extracted face crops

The detector uses GPU acceleration when available and skips images where a
reliable face cannot be detected.
"""

import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
from pathlib import Path

# ---- Config ----
SRC_DIR = "./target"
MARGIN = 40          # extra pixels around detected face box
MIN_FACE_SIZE = 40   # skip tiny/false detections

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

mtcnn = MTCNN(
    keep_all=False,
    min_face_size=MIN_FACE_SIZE,
    device=device,
)

exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
src_path = Path(SRC_DIR)
files = [p for p in src_path.rglob("*") if p.suffix.lower() in exts]

skipped = 0

for path in tqdm(files, desc="Detecting & cropping faces"):
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Could not open {path.name}: {e}")
        skipped += 1
        continue

    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        skipped += 1
        continue

    # Take the highest-confidence face
    best_idx = probs.argmax()
    x1, y1, x2, y2 = boxes[best_idx]

    w, h = img.size
    x1 = max(0, int(x1) - MARGIN)
    y1 = max(0, int(y1) - MARGIN)
    x2 = min(w, int(x2) + MARGIN)
    y2 = min(h, int(y2) + MARGIN)

    face = img.crop((x1, y1, x2, y2))

    try:
        ext = path.suffix.lower()

        if ext in {".jpg", ".jpeg"}:
            face.save(path, quality=100, subsampling=0)
        elif ext == ".png":
            face.save(path, optimize=True)
        elif ext == ".webp":
            face.save(path, quality=100, lossless=True)
        elif ext == ".bmp":
            face.save(path)
        else:
            face.save(path)

    except Exception as e:
        print(f"Could not save {path.name}: {e}")
        skipped += 1

print(
    f"\nDone. {len(files) - skipped}/{len(files)} images replaced, "
    f"{skipped} skipped (no face / unreadable / save error)."
)
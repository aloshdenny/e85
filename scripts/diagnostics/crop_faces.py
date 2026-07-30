import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

from pathlib import Path

# ---- Config ----
SRC_DIR = "./target"
DST_DIR = "./target_faces"
MARGIN = 40          # extra pixels around detected face box
MIN_FACE_SIZE = 40   # skip tiny/false detections

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

mtcnn = MTCNN(
    keep_all=False,
    min_face_size=MIN_FACE_SIZE,
    device=device,
)

os.makedirs(DST_DIR, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
src_path = Path(SRC_DIR)
files = [p for p in src_path.rglob("*") if p.suffix.lower() in exts]

skipped = 0
for path in tqdm(files, desc="Detecting & cropping faces"):
    fname = path.name
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Could not open {fname}: {e}")
        skipped += 1
        continue

    boxes, probs = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        skipped += 1
        continue

    # take the highest-confidence face
    best_idx = probs.argmax()
    x1, y1, x2, y2 = boxes[best_idx]

    w, h = img.size
    x1 = max(0, int(x1) - MARGIN)
    y1 = max(0, int(y1) - MARGIN)
    x2 = min(w, int(x2) + MARGIN)
    y2 = min(h, int(y2) + MARGIN)

    face = img.crop((x1, y1, x2, y2))  # no resize — keep native crop dimensions

    out_name = os.path.splitext(fname)[0] + "_face.jpg"
    face.save(os.path.join(DST_DIR, out_name), quality=100, subsampling=0)  # max quality, no chroma subsampling

print(f"\nDone. {len(files) - skipped}/{len(files)} faces extracted, {skipped} skipped (no face / unreadable).")
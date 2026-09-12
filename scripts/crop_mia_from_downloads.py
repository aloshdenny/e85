"""Crop Mia Khalifa faces from today's Downloads images.

Uses the existing target/mia.zip gallery as the identity template so
podcast guests and background people are dropped. Multiple detections
in one file (split-screen, two shots of her) are saved as separate crops.
"""

from __future__ import annotations

import io
import os
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

DOWNLOADS = Path("/Users/aoxo/Downloads")
GALLERY_ZIP = Path("/Users/aoxo/vscode/e85/target/mia.zip")
OUT_DIR = Path("/Users/aoxo/vscode/e85/target/mia_in_the_wild")
REJECT_DIR = OUT_DIR / "_rejected_other_faces"

TODAY_NAMES = [
    "mia-khalifa-11.jpg",
    "images (9).jpeg",
    "miakhalf_(1).webp",
    "images (8).jpeg",
    "mia-khalifa.jpg",
    "27mag-interview-khalifa-02-googleFourByThree.jpg",
    "mia-khalifa-interview.png",
    "3233eefcf8.webp",
    "images (7).jpeg",
    "Mia_Khalifa_interview_question.jpg",
    "miaa-1-1024x614.jpg",
    "x1080.jpeg",
    "images (6).jpeg",
    "52dbb2dfd691b3126dd3b00badcdbfd91723775587748785_original.webp",
    "images (5).jpeg",
    "_1eb21422-be86-11e9-9bc9-c6f10a5dc6e3.avif",
    "maxresdefault (1).jpg",
    "images (4).jpeg",
    "images (3).jpeg",
    "images (2).jpeg",
    "images (1).jpeg",
    "images.jpeg",
    "Mia_Khalifa_podcast.webp",
    "maxresdefault.jpg",
]

THRESH = 0.38  # insightface cosine; guests typically <0.25
MARGIN = 0.45
MIN_DET = 0.35
GALLERY_MAX = 80


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_bgr(path: Path):
    data = path.read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    tmp = Path("/tmp") / (path.stem + "_conv.jpg")
    r = subprocess.run(
        ["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
        capture_output=True,
    )
    if r.returncode == 0 and tmp.exists():
        img = cv2.imread(str(tmp))
        tmp.unlink(missing_ok=True)
        return img
    return None


def crop_box(img, bbox, margin=MARGIN):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * (1.0 + 2 * margin)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1i = int(round(cx + side / 2))
    y1i = int(round(cy + side / 2))
    x0, y0 = max(0, x0), max(0, y0)
    x1i, y1i = min(w, x1i), min(h, y1i)
    if x1i - x0 < 16 or y1i - y0 < 16:
        return None
    return img[y0:y1i, x0:x1i]


def save_jpg(path: Path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"encode failed {path}")
    path.write_bytes(buf.tobytes())


def gallery_centroid(app: FaceAnalysis):
    embs = []
    with zipfile.ZipFile(GALLERY_ZIP) as zf:
        names = [
            n for n in zf.namelist()
            if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            and not n.startswith("__MACOSX/")
            and not Path(n).name.startswith("._")
        ]
        rng = np.random.default_rng(0)
        if len(names) > GALLERY_MAX:
            names = list(rng.choice(names, GALLERY_MAX, replace=False))
        for n in names:
            raw = zf.read(n)
            arr = np.frombuffer(raw, dtype=np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if im is None:
                continue
            faces = app.get(im)
            if not faces:
                continue
            faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
            embs.append(unit(faces[0].normed_embedding.astype(np.float64)))
    if len(embs) < 8:
        raise SystemExit(f"gallery too small: {len(embs)}")
    c = unit(np.mean(embs, axis=0))
    print(f"gallery: {len(embs)} embeddings from mia.zip")
    return c


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REJECT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*.jpg"):
        p.unlink()
    for p in REJECT_DIR.glob("*.jpg"):
        p.unlink()

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    centroid = gallery_centroid(app)

    kept = 0
    rejected = 0
    empty = []
    print(f"\n{'file':<55s} {'faces':>5s} {'kept':>5s}  scores")
    print("-" * 90)

    for name in TODAY_NAMES:
        path = DOWNLOADS / name
        if not path.exists():
            print(f"{name}: MISSING")
            continue
        img = load_bgr(path)
        if img is None:
            print(f"{name}: UNREADABLE")
            empty.append(name)
            continue
        faces = app.get(img)
        if not faces:
            app.prepare(ctx_id=-1, det_size=(1024, 1024))
            faces = app.get(img)
            app.prepare(ctx_id=-1, det_size=(640, 640))
        scores = []
        n_kept = 0
        stem = Path(name).stem.replace(" ", "_")[:40]
        for i, f in enumerate(sorted(faces, key=lambda x: float(x.bbox[0]))):
            if float(f.det_score) < MIN_DET:
                continue
            sim = float(np.dot(unit(f.normed_embedding.astype(np.float64)), centroid))
            scores.append(sim)
            crop = crop_box(img, f.bbox)
            if crop is None:
                continue
            if sim >= THRESH:
                n_kept += 1
                kept += 1
                out = OUT_DIR / f"{stem}_mia{n_kept:02d}.jpg"
                save_jpg(out, crop)
            else:
                rejected += 1
                out = REJECT_DIR / f"{stem}_other{rejected:02d}_sim{sim:.2f}.jpg"
                save_jpg(out, crop)
        print(f"{name:<55s} {len(faces):5d} {n_kept:5d}  "
              + (" ".join(f"{s:+.2f}" for s in scores) or "-"))
        if n_kept == 0:
            empty.append(name)

    print("-" * 90)
    print(f"kept {kept} Mia crops -> {OUT_DIR}")
    print(f"rejected {rejected} other faces -> {REJECT_DIR}")
    if empty:
        print("no Mia crop from:", ", ".join(empty))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()

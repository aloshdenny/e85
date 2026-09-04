"""Crop and upright-align Michael Jackson faces from ~/Downloads/mj.

Same approach as crop_sins_from_downloads.py: clusters InsightFace embeddings
so any other person's face in the folder gets dropped, rotates each keep so
the eyes sit on a horizontal line, writes square 256px crops.

Usage:
  python scripts/crop_mj_from_downloads.py
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

SRC_DIRS = [Path("/Users/aoxo/Downloads/mj"), Path("/Users/aoxo/Downloads/mj2")]
OUT_DIR = Path("/Users/aoxo/vscode/e85/target/mj_crops")
REJECT_DIR = OUT_DIR / "_rejected_other_faces"
ZIP_OUT = Path("/Users/aoxo/vscode/e85/target/mj.zip")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".heic"}
OUT_SIZE = 256
MARGIN = 0.70
MIN_DET = 0.30
CLUSTER_SIM = 0.32


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
    tmp = Path("/tmp") / (path.stem[:40] + "_conv.jpg")
    r = subprocess.run(
        ["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
        capture_output=True,
    )
    if r.returncode == 0 and tmp.exists():
        img = cv2.imread(str(tmp))
        tmp.unlink(missing_ok=True)
        return img
    return None


def source_images():
    """Pooled across every SRC_DIRS entry that exists, deduped by content
    hash -- mj and mj2 were collected separately and may overlap."""
    seen_hashes = set()
    out = []
    for src in SRC_DIRS:
        if not src.exists():
            print(f"  [skip] {src} does not exist")
            continue
        for p in sorted(src.iterdir()):
            if not (p.is_file() and p.suffix.lower() in EXTS):
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h in seen_hashes:
                print(f"  [dup]  {p.relative_to(src.parent)} (exact duplicate, skipped)")
                continue
            seen_hashes.add(h)
            out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def upright_square(img, bbox, kps, out_size=OUT_SIZE, margin=MARGIN):
    """Rotate so the eyes are level, then take a square crop around the face.

    Padding first so a tilted head is not clipped. More stable on 3/4 views
    than warping onto the 112px ArcFace template.
    """
    le, re = np.asarray(kps[0], np.float32), np.asarray(kps[1], np.float32)
    angle = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    side = max(bw, bh) * (1.0 + 2 * margin)

    pad = int(max(img.shape[0], img.shape[1], side) * 0.6)
    fill = tuple(int(v) for v in img.reshape(-1, 3).mean(0))
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill)
    cx_p, cy_p = cx + pad, cy + pad
    R = cv2.getRotationMatrix2D((cx_p, cy_p), angle, 1.0)
    rot = cv2.warpAffine(
        padded, R, (padded.shape[1], padded.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
    )
    xa = int(round(cx_p - side / 2)); ya = int(round(cy_p - side / 2))
    xb = int(round(cx_p + side / 2)); yb = int(round(cy_p + side / 2))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(rot.shape[1], xb), min(rot.shape[0], yb)
    crop = rot[ya:yb, xa:xb]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def save_jpg(path: Path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"encode failed {path}")
    path.write_bytes(buf.tobytes())


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    paths = source_images()
    print(f"{len(paths)} unique images across {[str(d) for d in SRC_DIRS]}")
    if not paths:
        raise SystemExit("no images found")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    detections = []  # (path, img, face, emb)
    for p in paths:
        img = load_bgr(p)
        if img is None:
            print(f"  UNREADABLE {p.name}")
            continue
        faces = app.get(img)
        if not faces and min(img.shape[:2]) > 400:
            app.prepare(ctx_id=-1, det_size=(1024, 1024))
            faces = app.get(img)
            app.prepare(ctx_id=-1, det_size=(640, 640))
        if not faces:
            print(f"  no face  {p.name}")
            continue
        for f in faces:
            if float(f.det_score) < MIN_DET:
                continue
            detections.append((p, img, f, unit(np.asarray(f.normed_embedding, dtype=np.float64))))
        print(f"  {len(faces):2d} face(s)  {p.name}")

    if len(detections) < 3:
        raise SystemExit(f"only {len(detections)} detections")

    E = np.stack([e for *_, e in detections])
    S = E @ E.T
    counts = (S >= CLUSTER_SIM).sum(1)
    seed = int(np.argmax(counts))
    centroid = unit(E[counts >= counts[seed] * 0.8].mean(0) if (counts >= counts[seed] * 0.8).sum() >= 3
                    else E[seed])
    sims = E @ centroid
    keep_mask = sims >= CLUSTER_SIM
    print(f"\nmajority cluster: {keep_mask.sum()}/{len(detections)} faces  "
          f"(seed neighbors={int(counts[seed])}, sim {sims[keep_mask].min():+.2f}..{sims[keep_mask].max():+.2f})")

    if OUT_DIR.exists():
        for old in OUT_DIR.rglob("*.jpg"):
            old.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REJECT_DIR.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    for i, (p, img, f, emb) in enumerate(detections):
        sim = float(sims[i])
        crop = None
        kps = getattr(f, "kps", None)
        if kps is not None and len(kps) >= 2:
            crop = upright_square(img, f.bbox, kps)
        if crop is None:
            continue
        stem = p.stem.replace(" ", "_")[:36]
        if keep_mask[i]:
            kept += 1
            save_jpg(OUT_DIR / f"{stem}_mj{kept:02d}.jpg", crop)
        else:
            rejected += 1
            save_jpg(REJECT_DIR / f"{stem}_other{rejected:02d}_sim{sim:.2f}.jpg", crop)

    print(f"kept {kept} -> {OUT_DIR}")
    print(f"rejected {rejected} -> {REJECT_DIR}")

    crops = sorted(OUT_DIR.glob("*.jpg"))
    if len(crops) < 5:
        raise SystemExit("too few crops to zip")
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for c in crops:
            zf.write(c, f"mj/{c.name}")
    print(f"wrote {ZIP_OUT} ({len(crops)} images)")


if __name__ == "__main__":
    main()

"""
infer_ffhq_bulk.py

Bulk TribeV2 inference over a flat folder of FFHQ-70k (256x256) images,
batched for GPU throughput on a 24GB card.

DIFFERENT FROM infer_fairface_bulk.py in two structural ways:

  1. FFHQ has no demographic category buckets -- it's one flat folder of
     ~70,000 images. Instead of per-category zips, this saves ONE .npz per
     IMAGE ({out_dir}/{image_stem}.npz, containing that image's mean
     prediction vector alone). Resumability is skip-if-exists per image --
     already-completed images are never re-decoded/re-encoded on rerun. No
     shard-level bookkeeping needed. Tradeoff worth knowing: many small file
     writes has more filesystem overhead than a handful of larger files,
     particularly on network-mounted paths (e.g. WSL's /mnt/c/...).

  2. --duration defaults to 3.0s (not 1.0s). Per the TRIBE v2 paper: TRIBE
     was trained on naturalistic video/audio/text, not isolated still
     images. Applying it to a static image means holding one frame for a
     short synthetic clip, resampling to the target frequency, and
     truncating to a single timestep -- exactly what this pipeline already
     does. The paper's own validated standard for this is 3 seconds, not
     the 1.0s floor we derived empirically by probing the minimum duration
     that returned a non-empty prediction. 1.0s technically "worked" in the
     sense of returning output, but 3.0s is the setting the paper's authors
     actually validated this out-of-distribution use case against.

Confirmed via --probe-batch (carried over from infer_fairface_bulk.py):
  - model.predict() accepts multiple "Video" rows in one events dataframe.
  - Returned Segment list preserves row order and tags each timestep with
    that row's `timeline` value.
  - So: batch B images per predict() call, one row per image with a unique
    timeline id, then group the returned preds by segment.timeline and mean
    within each group to get one vector per image.

Usage:
  # Confirm your setup is behaving before running the full ~70k images:
  python infer_ffhq_bulk.py --probe 3
  python infer_ffhq_bulk.py --probe-batch 2

  # Full run
  python scripts/infer_ffhq_bulk.py --images-dir /path/to/ffhq70k-256 --out-dir ./ffhq_preds \
      --batch-size 64
"""

import os
import sys
import shutil
import tempfile
import warnings
import logging
import argparse
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import cv2
import pandas as pd

from tribev2.demo_utils import TribeModel
from chunk_utils import save_npz, npz_exists

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Paths ────────────────────────────────────────────────────────────────────

IMAGES_DIR = Path("./ffhq70k-256")   # flat folder of FFHQ images
OUT_DIR    = Path("./ffhq_preds")    # output directory for sharded .npz files
CACHE_DIR  = Path("./cache")


# ── Temp root: prefer RAM-backed /dev/shm if available ───────────────────────

def get_tmp_root():
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


# ── Flat-folder image discovery ───────────────────────────────────────────────

def discover_images(images_dir: Path):
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def decode_image_from_disk(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


# ── Write a short static clip for one image ───────────────────────────────────

def write_static_clip(img, out_path: Path, duration: float, fps: int):
    h, w = img.shape[:2]
    n_frames = max(1, int(round(duration * fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(img)
    writer.release()


def make_multi_row_df(rows, duration: float) -> pd.DataFrame:
    """rows: list of (clip_path, timeline_id)"""
    records = []
    for clip_path, timeline_id in rows:
        records.append({
            "type":      "Video",
            "start":     0.0,
            "duration":  duration,
            "timeline":  timeline_id,
            "subject":   "default",
            "session":   "",
            "task":      "",
            "run":       "",
            "filepath":  str(clip_path.resolve()),
            "frequency": 60.0,
            "offset":    0.0,
            "stop":      duration,
            "context":   float("nan"),
        })
    return pd.DataFrame(records)


def group_preds_by_timeline(preds, segments):
    """Group preds rows by segments[i].timeline, preserving first-seen order.
    Returns dict: timeline_id -> mean pred vector (n_rois,)"""
    groups = defaultdict(list)
    for i, seg in enumerate(segments):
        groups[seg.timeline].append(preds[i])
    return {tl: np.mean(np.stack(vecs, axis=0), axis=0) for tl, vecs in groups.items()}


# ── Diagnostics (carried over from infer_fairface_bulk.py) ──────────────────

def probe_batch_row_identity(model, tmp_dir: Path, duration: float, fps: int, n_images: int = 2):
    rows = []
    for i in range(n_images):
        img = np.full((256, 256, 3), fill_value=(i * 40) % 255, dtype=np.uint8)
        clip_path = tmp_dir / f"probe_batch_{i}.mp4"
        write_static_clip(img, clip_path, duration=duration, fps=fps)
        rows.append((clip_path, f"probe_{i}"))
    df = make_multi_row_df(rows, duration=duration)
    preds, segments = model.predict(events=df)
    print(f"[probe_batch] preds.shape={preds.shape}")
    print(f"[probe_batch] n segments={len(segments)}")
    for s in segments:
        print(f"  {s}")
    return preds, segments


def probe_single_images(model, image_paths, tmp_dir: Path, duration: float, fps: int, n: int):
    for idx, path in enumerate(image_paths[:n]):
        img = decode_image_from_disk(path)
        clip_path = tmp_dir / f"probe_{idx}.mp4"
        write_static_clip(img, clip_path, duration=duration, fps=fps)
        df = make_multi_row_df([(clip_path, f"probe_{idx}")], duration=duration)
        preds, segments = model.predict(events=df)
        print(f"  [probe] {path.name}: preds.shape={preds.shape} "
              f"segments sample={segments[:1] if len(segments) else segments}")


# ── Sharded batch processing over the flat folder ────────────────────────────

def process_batch(model, batch_paths, out_dir: Path, tmp_root: Path,
                   duration: float, fps: int):
    """
    One npz per image: {out_dir}/{image_stem}.npz containing that image's
    mean prediction vector alone. Images already having an output file are
    skipped before ever being decoded/encoded -- simpler resumability than
    shard-level bookkeeping, at the cost of many small file writes (worth
    knowing on network-mounted paths, e.g. WSL's /mnt/c/..., where many
    small-file operations can be slow).

    Returns (n_processed, n_skipped, n_failed) for this batch.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_batch_", dir=tmp_root))
    n_processed = n_skipped = n_failed = 0

    try:
        rows = []
        timeline_to_path = {}

        for i, img_path in enumerate(batch_paths):
            out_path = out_dir / f"{img_path.stem}.npz"
            if npz_exists(out_path):
                n_skipped += 1
                continue
            timeline_id = f"img_{i}"
            try:
                img = decode_image_from_disk(img_path)
                clip_path = tmp_dir / f"{timeline_id}.mp4"
                write_static_clip(img, clip_path, duration=duration, fps=fps)
                rows.append((clip_path, timeline_id))
                timeline_to_path[timeline_id] = img_path
            except Exception as e:
                print(f"  [ERROR building clip] {img_path.name}: {e}")
                n_failed += 1

        if not rows:
            return n_processed, n_skipped, n_failed

        df = make_multi_row_df(rows, duration=duration)
        try:
            preds, segments = model.predict(events=df)
        except Exception as e:
            print(f"  [ERROR predict()] batch: {e}")
            n_failed += len(timeline_to_path)
            for clip_path, _ in rows:
                clip_path.unlink(missing_ok=True)
            return n_processed, n_skipped, n_failed

        grouped = group_preds_by_timeline(preds, segments)

        for timeline_id, img_path in timeline_to_path.items():
            vec = grouped.get(timeline_id)
            if vec is None:
                print(f"  [WARN] no prediction returned for {img_path.name}")
                n_failed += 1
                continue
            out_path = out_dir / f"{img_path.stem}.npz"
            save_npz(out_path, preds=vec, filename=img_path.name)
            n_processed += 1

        for clip_path, _ in rows:
            clip_path.unlink(missing_ok=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return n_processed, n_skipped, n_failed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batched FFHQ-70k -> TribeV2 inference.")
    parser.add_argument("--images-dir", default=IMAGES_DIR, type=Path,
                        help="Flat folder containing all FFHQ images.")
    parser.add_argument("--out-dir", default=OUT_DIR, type=Path)
    parser.add_argument("--cache-folder", default=CACHE_DIR, type=Path)
    parser.add_argument("--duration", type=float, default=3.0,
                         help="Synthetic static-clip duration (s). TRIBE v2 paper's "
                              "validated standard for the still-image-as-video "
                              "out-of-distribution case is 3.0s (not the 1.0s floor "
                              "we derived empirically by probing for minimum non-empty "
                              "output -- that worked, but wasn't validated by the authors).")
    parser.add_argument("--fps", type=int, default=2,
                         help="Synthetic clip frame rate. Kept low since content is static.")
    parser.add_argument("--batch-size", type=int, default=64,
                         help="Images per predict() call. Tune up/down based on VRAM headroom.")
    parser.add_argument("--probe", type=int, default=0,
                         help="Run this many single-image predict() calls and print shapes. "
                              "No files saved.")
    parser.add_argument("--probe-batch", type=int, default=0,
                         help="Diagnostic: multi-row predict() call with this many dummy "
                              "images, prints segment structure. No files saved.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = get_tmp_root()
    print(f"Using temp root: {tmp_root} "
          f"({'RAM-backed' if str(tmp_root) == '/dev/shm' else 'disk-backed, not /dev/shm'})")

    print("Loading TribeV2...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=Path(args.cache_folder))

    if args.probe_batch:
        tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_probe_batch_", dir=tmp_root))
        try:
            probe_batch_row_identity(model, tmp_dir, duration=args.duration,
                                      fps=args.fps, n_images=args.probe_batch)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    all_images = discover_images(images_dir)
    if not all_images:
        print(f"No images found in {images_dir}")
        sys.exit(1)
    print(f"Discovered {len(all_images)} images in {images_dir}")

    if args.probe:
        tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_probe_", dir=tmp_root))
        try:
            probe_single_images(model, all_images, tmp_dir,
                                 duration=args.duration, fps=args.fps, n=args.probe)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    n_batches = (len(all_images) + args.batch_size - 1) // args.batch_size
    print(f"Processing {len(all_images)} images in {n_batches} batches of up to "
          f"{args.batch_size} (one .npz per image, resumable via skip-if-exists)")

    total_processed = total_skipped = total_failed = 0
    for batch_idx in range(n_batches):
        batch_paths = all_images[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        n_proc, n_skip, n_fail = process_batch(model, batch_paths, out_dir, tmp_root,
                                               duration=args.duration, fps=args.fps)
        total_processed += n_proc
        total_skipped += n_skip
        total_failed += n_fail
        done = min((batch_idx + 1) * args.batch_size, len(all_images))
        print(f"  ...{done}/{len(all_images)}  "
              f"(processed={total_processed} skipped={total_skipped} failed={total_failed})")

    print(f"\nAll images processed.")
    print(f"  Processed: {total_processed}")
    print(f"  Skipped (already done): {total_skipped}")
    print(f"  Failed:    {total_failed}")
    print(f"Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
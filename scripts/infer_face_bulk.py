"""
infer_fairface_bulk.py

Bulk TribeV2 inference over FairFace category zips (fairface/*.zip),
batched for GPU throughput on a 24GB card.

Confirmed via --probe-batch:
  - model.predict() accepts multiple "Video" rows in one events dataframe.
  - Returned Segment list preserves row order and tags each timestep with
    that row's `timeline` value (e.g. 10 Segments per row for a 10s clip
    at 1Hz, all sharing that row's timeline string).
  - So: batch B images per predict() call, one row per image with a unique
    timeline id, then group the returned preds by segment.timeline and mean
    within each group to get one vector per image.

Speed profile vs. the original video pipeline:
  - Images decoded straight from zip bytes (cv2.imdecode) -- never hit disk.
  - No whisperx/audio parsing.
  - No PlotBrain rendering -- this pass only produces pred_mean vectors for
    the difference-in-means step.
  - One predict() call per batch of --batch-size images instead of per image
    -- the GPU forward pass is now the dominant cost, not Python/IO overhead.
  - Synthetic clips are written to /dev/shm when available (RAM), else system
    temp; the whole temp dir for a category is deleted once that category
    finishes.

Usage:
  # Confirm your setup is behaving before running the full 97k images:
  python infer_fairface_bulk.py --probe 3
  python infer_fairface_bulk.py --probe-batch 2

  # Full run
  python infer_fairface_bulk.py --zips-dir ./fairface --out-dir ./fairface_preds --batch-size 32
"""

import os
import sys
import shutil
import tempfile
import warnings
import logging
import argparse
import zipfile
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import cv2
import pandas as pd

from tribev2.demo_utils import TribeModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Temp root: prefer RAM-backed /dev/shm if available ───────────────────────

def get_tmp_root():
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


# ── Category name -> (age, gender, race) ─────────────────────────────────────

def parse_category(name):
    parts = name.split("_")
    age = parts[0]
    gender = parts[1]
    race = "_".join(parts[2:])
    return age, gender, race


# ── In-memory image decode ────────────────────────────────────────────────────

def decode_image_from_zip(zf: zipfile.ZipFile, name: str):
    data = zf.read(name)          # bytes, never touches disk
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {name}")
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
    """
    rows: list of (clip_path, timeline_id)
    """
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
    """
    Group preds rows by segments[i].timeline, preserving first-seen order.
    Returns dict: timeline_id -> mean pred vector (n_rois,)
    """
    groups = defaultdict(list)
    for i, seg in enumerate(segments):
        groups[seg.timeline].append(preds[i])

    return {tl: np.mean(np.stack(vecs, axis=0), axis=0) for tl, vecs in groups.items()}


# ── Diagnostics (kept from earlier probing) ──────────────────────────────────

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


def probe_single_images(model, zf: zipfile.ZipFile, members, tmp_dir: Path,
                         duration: float, fps: int, n: int):
    for idx, name in enumerate(members[:n]):
        img = decode_image_from_zip(zf, name)
        clip_path = tmp_dir / f"probe_{idx}.mp4"
        write_static_clip(img, clip_path, duration=duration, fps=fps)
        df = make_multi_row_df([(clip_path, f"probe_{idx}")], duration=duration)
        preds, segments = model.predict(events=df)
        print(f"  [probe] {name}: preds.shape={preds.shape} "
              f"segments sample={segments[:1] if len(segments) else segments}")


# ── Batched category processing ───────────────────────────────────────────────

def process_fairface_zips(model, zip_path: Path, out_dir: Path, tmp_root: Path,
                          duration: float, fps: int, batch_size: int):
    category = zip_path.stem
    out_path = out_dir / f"{category}.npz"
    if out_path.exists():
        print(f"[SKIP] {category} (already done)")
        return

    age, gender, race = parse_category(category)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tribe_{category}_", dir=tmp_root))
    print(f"[{category}] temp dir: {tmp_dir}")

    all_names = []
    all_means = []
    failed = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS]
            print(f"[{category}] {len(members)} images, batch_size={batch_size}")

            for batch_start in range(0, len(members), batch_size):
                batch_members = members[batch_start:batch_start + batch_size]

                rows = []               # (clip_path, timeline_id)
                timeline_to_name = {}   # timeline_id -> original filename

                for i, name in enumerate(batch_members):
                    timeline_id = f"img_{batch_start + i}"
                    try:
                        img = decode_image_from_zip(zf, name)
                        clip_path = tmp_dir / f"{timeline_id}.mp4"
                        write_static_clip(img, clip_path, duration=duration, fps=fps)
                        rows.append((clip_path, timeline_id))
                        timeline_to_name[timeline_id] = name
                    except Exception as e:
                        print(f"  [ERROR building clip] {name}: {e}")
                        failed.append(name)

                if not rows:
                    continue

                df = make_multi_row_df(rows, duration=duration)
                try:
                    preds, segments = model.predict(events=df)
                except Exception as e:
                    print(f"  [ERROR predict()] batch at {batch_start}: {e}")
                    failed.extend(timeline_to_name.values())
                    for clip_path, _ in rows:
                        clip_path.unlink(missing_ok=True)
                    continue

                grouped = group_preds_by_timeline(preds, segments)

                for timeline_id, name in timeline_to_name.items():
                    vec = grouped.get(timeline_id)
                    if vec is None:
                        print(f"  [WARN] no prediction returned for {name}")
                        failed.append(name)
                        continue
                    all_names.append(name)
                    all_means.append(vec)

                for clip_path, _ in rows:
                    clip_path.unlink(missing_ok=True)

                done = min(batch_start + batch_size, len(members))
                print(f"  ...{done}/{len(members)}")

        if all_means:
            means_arr = np.stack(all_means, axis=0)
            np.savez_compressed(
                out_path,
                preds=means_arr,
                filenames=np.array(all_names),
                failed=np.array(failed),
                age=age, gender=gender, race=race,
            )
            print(f"[{category}] saved {means_arr.shape} -> {out_path} "
                  f"({len(failed)} failed)")
        else:
            print(f"[{category}] no successful predictions -- nothing saved")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batched FairFace -> TribeV2 inference.")
    parser.add_argument("--zips-dir", default="./fairface")
    parser.add_argument("--out-dir", default="./fairface_preds")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--duration", type=float, default=1.0,
                         help="Synthetic static-clip duration (s). Confirmed floor: "
                              "1.0s @ fps=2 gives a single clean valid timestep with "
                              "minimal encode cost. Do not go below this without "
                              "re-probing -- offset behavior differs from earlier "
                              "assumption, but very short clips risk 0 frames.")
    parser.add_argument("--fps", type=int, default=2,
                         help="Synthetic clip frame rate. Kept low since content is static.")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Images per predict() call. Tune up/down based on VRAM headroom.")
    parser.add_argument("--probe", type=int, default=0,
                         help="Run this many single-image predict() calls from the FIRST "
                              "zip and print shapes. No files saved.")
    parser.add_argument("--probe-batch", type=int, default=0,
                         help="Diagnostic: multi-row predict() call with this many dummy "
                              "images, prints segment structure. No files saved.")
    args = parser.parse_args()

    zips_dir = Path(args.zips_dir)
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

    zip_files = sorted(zips_dir.glob("*.zip"))
    if not zip_files:
        print(f"No zips found in {zips_dir}")
        sys.exit(1)

    if args.probe:
        tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_probe_", dir=tmp_root))
        try:
            with zipfile.ZipFile(zip_files[0], "r") as zf:
                members = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS]
                probe_single_images(model, zf, members, tmp_dir,
                                     duration=args.duration, fps=args.fps, n=args.probe)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    print(f"Discovered {len(zip_files)} category zips")
    for zip_path in zip_files:
        process_fairface_zips(model, zip_path, out_dir, tmp_root,
                              duration=args.duration, fps=args.fps,
                              batch_size=args.batch_size)

    print("\nAll categories processed.")
    print(f"Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
"""
infer_face_bulk.py

Bulk TribeV2 inference over FairFace category zips (category_zips/*.zip),
built for speed:

  - Images are read directly out of each zip in memory (zipfile + cv2.imdecode)
    and are NEVER extracted to a persistent path on disk.
  - No whisperx/audio transcription (reuses the make_video_only_df trick from
    infer_bulk.py — builds the minimal events row directly).
  - No PlotBrain rendering — this pass is purely to get per-ROI prediction
    vectors for the difference-in-means step, not brain-map PNGs.
  - Per-image synthetic clips are written to a short-lived temp dir (prefers
    /dev/shm if present, i.e. RAM-backed, else falls back to system temp) and
    deleted immediately after each image is processed. The whole temp dir for
    a category is torn down once that category finishes.

IMPORTANT — read before running at scale:
  model.predict() is called ONCE PER IMAGE by default (not batched across
  images) because I don't have visibility into exactly how the returned
  `segments` array demarcates multiple video rows within a single predict()
  call. Guessing at that mapping risks silently mis-attributing a prediction
  to the wrong image, which would quietly corrupt your dataset. Run with
  --probe first (see below) to inspect segments on a handful of images; if
  you confirm the structure, batching multiple images per predict() call is
  a further speedup this script can be extended for.

  Also: FmriExtractor.offset=5.0 (hemodynamic delay) means very short clips
  may return zero valid post-offset timesteps. Default --duration is set to
  10s (well past the offset) rather than the 2s used in the earlier
  single-image script — verify with --probe that you're getting a sane
  preds.shape before running the full ~97k images.

Usage:
  # Sanity check first — run 3 images from the first zip, print shapes/segments
  python infer_face_bulk.py --zips-dir ./category_zips --probe 3

  # Full run
  python infer_face_bulk.py --zips-dir ./category_zips --out-dir ./fairface_preds
"""

import os
import sys
import io
import shutil
import tempfile
import warnings
import logging
import argparse
import zipfile
from pathlib import Path

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
    # e.g. "20-29_female_east_asian" -> ("20-29", "female", "east_asian")
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


# ── Minimal single-row events dataframe (bypasses whisperx, same as infer_bulk.py) ──

def make_video_only_df(video_path: Path, duration: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "type":      "Video",
        "start":     0.0,
        "duration":  duration,
        "timeline":  "default",
        "subject":   "default",
        "session":   "",
        "task":      "",
        "run":       "",
        "filepath":  str(video_path.resolve()),
        "frequency": 60.0,
        "offset":    0.0,
        "stop":      duration,
        "context":   float("nan"),
    }])


# ── Per-image inference ────────────────────────────────────────────────────────

def infer_one_image(model, img, tmp_dir: Path, tag: str, duration: float, fps: int, verbose=False):
    clip_path = tmp_dir / f"{tag}.mp4"
    write_static_clip(img, clip_path, duration=duration, fps=fps)

    df = make_video_only_df(clip_path, duration=duration)
    preds, segments = model.predict(events=df)

    if verbose:
        print(f"    [probe] preds.shape={preds.shape} "
              f"segments type={type(segments)} "
              f"segments sample={segments[:1] if len(segments) else segments}")

    try:
        clip_path.unlink()
    except OSError:
        pass

    if preds.shape[0] == 0:
        return None  # no valid post-offset timesteps — flag upstream

    return preds.mean(axis=0)


# ── Category processing ────────────────────────────────────────────────────────

def process_category_zip(model, zip_path: Path, out_dir: Path, tmp_root: Path,
                          duration: float, fps: int, probe: int = 0):
    category = zip_path.stem
    out_path = out_dir / f"{category}.npz"
    if out_path.exists() and probe == 0:
        print(f"[SKIP] {category} (already done)")
        return

    age, gender, race = parse_category(category)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tribe_{category}_", dir=tmp_root))
    print(f"[{category}] temp dir: {tmp_dir}")

    means = []
    names = []
    failed = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS]
            if probe:
                members = members[:probe]

            print(f"[{category}] {len(members)} images")

            for idx, name in enumerate(members):
                try:
                    img = decode_image_from_zip(zf, name)
                    mean_pred = infer_one_image(
                        model, img, tmp_dir, tag=f"img_{idx:06d}",
                        duration=duration, fps=fps, verbose=bool(probe),
                    )
                    if mean_pred is None:
                        print(f"  [WARN] no valid timesteps: {name}")
                        failed.append(name)
                        continue
                    means.append(mean_pred)
                    names.append(name)
                except Exception as e:
                    print(f"  [ERROR] {name}: {e}")
                    failed.append(name)

                if (idx + 1) % 50 == 0:
                    print(f"  ...{idx + 1}/{len(members)}")

        if probe:
            print(f"[{category}] probe complete — inspect the printed shapes above "
                  f"before running the full set.")
            return

        if means:
            means_arr = np.stack(means, axis=0)
            np.savez_compressed(
                out_path,
                preds=means_arr,
                filenames=np.array(names),
                failed=np.array(failed),
                age=age, gender=gender, race=race,
            )
            print(f"[{category}] saved {means_arr.shape} -> {out_path} "
                  f"({len(failed)} failed)")
        else:
            print(f"[{category}] no successful predictions — nothing saved")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bulk FairFace -> TribeV2 inference.")
    parser.add_argument("--zips-dir", default="./category_zips")
    parser.add_argument("--out-dir", default="./fairface_preds")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--duration", type=float, default=10.0,
                         help="Synthetic static-clip duration (s). Must clear "
                              "FmriExtractor offset=5.0 to get valid timesteps.")
    parser.add_argument("--fps", type=int, default=2,
                         help="Synthetic clip frame rate. Kept low since content is static.")
    parser.add_argument("--probe", type=int, default=0,
                         help="If >0, only run this many images from the FIRST zip "
                              "and print preds/segments shapes for inspection. "
                              "No files are saved in probe mode.")
    args = parser.parse_args()

    zips_dir = Path(args.zips_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = get_tmp_root()
    print(f"Using temp root: {tmp_root} "
          f"({'RAM-backed' if str(tmp_root) == '/dev/shm' else 'disk-backed, not /dev/shm'})")

    print("Loading TribeV2...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=Path(args.cache_folder))

    zip_files = sorted(zips_dir.glob("*.zip"))
    if not zip_files:
        print(f"No zips found in {zips_dir}")
        sys.exit(1)

    if args.probe:
        print(f"PROBE MODE — running {args.probe} image(s) from {zip_files[0].name} only")
        process_category_zip(model, zip_files[0], out_dir, tmp_root,
                              duration=args.duration, fps=args.fps, probe=args.probe)
        return

    print(f"Discovered {len(zip_files)} category zips")
    for zip_path in zip_files:
        process_category_zip(model, zip_path, out_dir, tmp_root,
                              duration=args.duration, fps=args.fps)

    print("\nAll categories processed.")
    print(f"Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
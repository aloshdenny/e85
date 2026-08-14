"""
infer_fairface_bulk.py

Bulk TribeV2 inference over a general-population face zip (or a directory of
category zips), batched for GPU throughput.

Output format matches the merged FairFace+FFHQ layout used by the rest of the
pipeline: ONE .npz PER IMAGE with
  preds      shape (20484,)
  filename   zip member name (str)

Defaults point at the merged zip / preds tree. Pass --zips-dir to run the
legacy multi-category FairFace dump instead (still writes per-image npzs).

Usage:
  python scripts/infer_fairface_bulk.py --probe 3
  python scripts/infer_fairface_bulk.py --probe-batch 2
  python scripts/infer_fairface_bulk.py --batch-size 64

  # Legacy category-zip dump:
  python scripts/infer_fairface_bulk.py --zips-dir ./fairface --out-dir ./fairface_preds
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
from chunk_utils import save_npz, npz_exists, ensure_fused_zip, resolve_zip_member

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Paths ────────────────────────────────────────────────────────────────────

ZIP_PATH  = Path("./fairface + ffhq/fairface + ffhq.zip")
OUT_DIR   = Path("./fairface + ffhq preds")
CACHE_DIR = Path("./cache")


def get_tmp_root():
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


def decode_image_from_zip(zf: zipfile.ZipFile, name: str):
    member = resolve_zip_member(zf, name)
    data = zf.read(member)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {name}")
    return img


def write_static_clip(img, out_path: Path, duration: float, fps: int):
    h, w = img.shape[:2]
    n_frames = max(1, int(round(duration * fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(img)
    writer.release()


def make_multi_row_df(rows, duration: float) -> pd.DataFrame:
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
    groups = defaultdict(list)
    for i, seg in enumerate(segments):
        groups[seg.timeline].append(preds[i])
    return {tl: np.mean(np.stack(vecs, axis=0), axis=0) for tl, vecs in groups.items()}


def member_out_stem(member: str) -> str:
    """Stable per-image npz stem from a zip member path."""
    p = Path(member)
    if len(p.parts) == 1:
        return p.stem
    return "_".join(p.with_suffix("").parts)


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


def process_zip(model, zip_path: Path, out_dir: Path, tmp_root: Path,
                duration: float, fps: int, batch_size: int):
    """
    One .npz per zip member. Already-done images are skipped via npz_exists.
    Returns (n_processed, n_skipped, n_failed).
    """
    zip_path = ensure_fused_zip(zip_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tribe_{zip_path.stem}_", dir=tmp_root))
    n_processed = n_skipped = n_failed = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS]
            print(f"[{zip_path.name}] {len(members)} images, batch_size={batch_size}")

            for batch_start in range(0, len(members), batch_size):
                batch_members = members[batch_start:batch_start + batch_size]
                rows = []
                timeline_to_name = {}

                for i, name in enumerate(batch_members):
                    out_path = out_dir / f"{member_out_stem(name)}.npz"
                    if npz_exists(out_path):
                        n_skipped += 1
                        continue
                    timeline_id = f"img_{batch_start + i}"
                    try:
                        img = decode_image_from_zip(zf, name)
                        clip_path = tmp_dir / f"{timeline_id}.mp4"
                        write_static_clip(img, clip_path, duration=duration, fps=fps)
                        rows.append((clip_path, timeline_id))
                        timeline_to_name[timeline_id] = name
                    except Exception as e:
                        print(f"  [ERROR building clip] {name}: {e}")
                        n_failed += 1

                if not rows:
                    continue

                df = make_multi_row_df(rows, duration=duration)
                try:
                    preds, segments = model.predict(events=df)
                except Exception as e:
                    print(f"  [ERROR predict()] batch at {batch_start}: {e}")
                    n_failed += len(timeline_to_name)
                    for clip_path, _ in rows:
                        clip_path.unlink(missing_ok=True)
                    continue

                grouped = group_preds_by_timeline(preds, segments)
                for timeline_id, name in timeline_to_name.items():
                    vec = grouped.get(timeline_id)
                    if vec is None:
                        print(f"  [WARN] no prediction returned for {name}")
                        n_failed += 1
                        continue
                    out_path = out_dir / f"{member_out_stem(name)}.npz"
                    # Canonical per-image schema: 1D preds + singular filename
                    save_npz(out_path, preds=np.asarray(vec, dtype=np.float32),
                               filename=Path(name).name)
                    n_processed += 1

                for clip_path, _ in rows:
                    clip_path.unlink(missing_ok=True)

                done = min(batch_start + batch_size, len(members))
                print(f"  ...{done}/{len(members)}  "
                      f"(processed={n_processed} skipped={n_skipped} failed={n_failed})")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return n_processed, n_skipped, n_failed


def main():
    parser = argparse.ArgumentParser(
        description="Batched FairFace / merged-zip -> TribeV2 inference "
                    "(one .npz per image)."
    )
    parser.add_argument(
        "--zip", default=None, type=Path,
        help=f"Single image zip (default: {ZIP_PATH}). Chunked zips are fused.",
    )
    parser.add_argument(
        "--zips-dir", default=None, type=Path,
        help="Directory of category .zip files (legacy). Overrides --zip.",
    )
    parser.add_argument("--out-dir", default=OUT_DIR, type=Path)
    parser.add_argument("--cache-folder", default=CACHE_DIR, type=Path)
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Synthetic static-clip duration (s).")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe", type=int, default=0)
    parser.add_argument("--probe-batch", type=int, default=0)
    args = parser.parse_args()

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

    if args.zips_dir is not None:
        zip_files = sorted(Path(args.zips_dir).glob("*.zip"))
        if not zip_files:
            print(f"No zips found in {args.zips_dir}")
            sys.exit(1)
    else:
        zip_path = Path(args.zip) if args.zip is not None else ZIP_PATH
        zip_files = [zip_path]

    if args.probe:
        tmp_dir = Path(tempfile.mkdtemp(prefix="tribe_probe_", dir=tmp_root))
        try:
            zp = ensure_fused_zip(zip_files[0])
            with zipfile.ZipFile(zp, "r") as zf:
                members = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS]
                probe_single_images(model, zf, members, tmp_dir,
                                     duration=args.duration, fps=args.fps, n=args.probe)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    print(f"Discovered {len(zip_files)} zip(s); writing per-image npzs to {out_dir}")
    total_p = total_s = total_f = 0
    for zip_path in zip_files:
        p, s, f = process_zip(model, zip_path, out_dir, tmp_root,
                              duration=args.duration, fps=args.fps,
                              batch_size=args.batch_size)
        total_p += p
        total_s += s
        total_f += f

    print("\nAll zips processed.")
    print(f"  Processed: {total_p}")
    print(f"  Skipped (already done): {total_s}")
    print(f"  Failed:    {total_f}")
    print(f"Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

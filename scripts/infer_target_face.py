"""
python scripts/infer_target_face.py
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

import cv2
import numpy as np
import pandas as pd

from tribev2.demo_utils import TribeModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Paths ────────────────────────────────────────────────────────────────────

TARGET_DIR   = Path("./target")        # target image zips, discovered recursively (/**/*.zip)
OUT_DIR      = Path("./target_preds")  # output directory for per-zip .npz files
CACHE_DIR    = Path("./cache")


def get_tmp_root():
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


def decode_image_from_zip(zf, name):
    data = zf.read(name)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Could not decode {name}")

    return img


def write_static_clip(img, out_path, duration, fps):
    h, w = img.shape[:2]
    n_frames = max(1, int(round(duration * fps)))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for _ in range(n_frames):
        writer.write(img)

    writer.release()


def make_events(rows, duration):
    records = []

    for clip_path, timeline in rows:
        records.append(
            {
                "type": "Video",
                "start": 0.0,
                "duration": duration,
                "timeline": timeline,
                "subject": "default",
                "session": "",
                "task": "",
                "run": "",
                "filepath": str(clip_path.resolve()),
                "frequency": 60.0,
                "offset": 0.0,
                "stop": duration,
                "context": float("nan"),
            }
        )

    return pd.DataFrame(records)


def group_predictions(preds, segments):
    grouped = defaultdict(list)

    for pred, seg in zip(preds, segments):
        grouped[seg.timeline].append(pred)

    return {
        k: np.mean(np.stack(v), axis=0)
        for k, v in grouped.items()
    }


def process_zip(
    model,
    zip_path,
    out_dir,
    tmp_root,
    duration,
    fps,
    batch_size,
):
    out_file = out_dir / f"{zip_path.stem}.npz"

    if out_file.exists():
        print(f"[SKIP] {zip_path.name}")
        return

    print(f"\nProcessing {zip_path.name}")

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix="tribe_",
            dir=tmp_root,
        )
    )

    vectors = []
    filenames = []
    failed = []

    try:
        with zipfile.ZipFile(zip_path) as zf:

            members = [
                m
                for m in zf.namelist()
                if Path(m).suffix.lower() in IMAGE_EXTS
            ]

            print(f"{len(members)} images")

            for batch_start in range(0, len(members), batch_size):

                batch = members[batch_start : batch_start + batch_size]

                rows = []
                mapping = {}

                for i, name in enumerate(batch):

                    timeline = f"img_{batch_start+i}"

                    try:
                        img = decode_image_from_zip(zf, name)

                        clip = tmp_dir / f"{timeline}.mp4"

                        write_static_clip(
                            img,
                            clip,
                            duration,
                            fps,
                        )

                        rows.append((clip, timeline))
                        mapping[timeline] = name

                    except Exception as e:
                        print(name, e)
                        failed.append(name)

                if not rows:
                    continue

                df = make_events(rows, duration)

                try:
                    preds, segments = model.predict(events=df)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("Prediction failed:", e)

                    failed.extend(mapping.values())

                    for clip, _ in rows:
                        clip.unlink(missing_ok=True)

                    continue

                grouped = group_predictions(preds, segments)

                for timeline, name in mapping.items():

                    vec = grouped.get(timeline)

                    if vec is None:
                        failed.append(name)
                        continue

                    filenames.append(name)
                    vectors.append(vec)

                for clip, _ in rows:
                    clip.unlink(missing_ok=True)

                print(
                    f"{min(batch_start+batch_size,len(members))}/{len(members)}"
                )

        if vectors:

            vectors = np.stack(vectors)

            np.savez_compressed(
                out_file,
                preds=vectors,
                filenames=np.array(filenames),
                failed=np.array(failed),
            )

            print(
                f"Saved {vectors.shape} -> {out_file}"
            )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-dir",
        default=TARGET_DIR,
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        default=OUT_DIR,
        type=Path,
    )

    parser.add_argument(
        "--cache-folder",
        default=CACHE_DIR,
        type=Path,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_root = get_tmp_root()

    print("Loading TribeV2...")

    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=Path(args.cache_folder),
    )

    zips = sorted(target_dir.rglob("*.zip"))

    if not zips:
        print("No zip files found.")
        sys.exit(1)

    for z in zips:

        process_zip(
            model=model,
            zip_path=z,
            out_dir=out_dir,
            tmp_root=tmp_root,
            duration=args.duration,
            fps=args.fps,
            batch_size=args.batch_size,
        )

    print("\nDone.")
    print(f"Outputs saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
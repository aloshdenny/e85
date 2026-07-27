"""
validation.py

Answers the real question after face abliteration runs: did the surgery
suppress target face identity SPECIFICALLY, or did it also dampen OFA/FFA
response to any strongly-presented face?

Reads images directly from a flat val/ folder -- no npz files involved.
Group membership comes straight from the filename: anything matching
--target-prefix (default "mia") is the target group, everything else is
the general/other-identity group (asian1_face.jpg, ebony1_face.jpg,
euro1_face.jpg, middle-eastern1_face.jpg, etc).

Efficiency fix from the previous version: the JPEG->MP4 encode happened
twice (once per pass) even though the images never change between passes --
only the model weights do. Now the MP4 is encoded ONCE per image.

CACHING BUG (same one hit in the earlier video-based project): TribeModel's
video/image feature extractors cache their output keyed by input filepath
(confirmed in the printed config: infra mode='cached', keep_in_ram=True).
Reusing identical clip paths across both predict() calls made the second
pass silently return the FIRST pass's cached features -- the abliterated
weights were loaded correctly but never actually got exercised, which is
why every delta came back as exactly 0.00000. Fix: the post-surgery pass
runs on HARDLINKED duplicates of the same clip files (new filepath -> new
cache key, same bytes -> no re-encode cost).

  1. Build all MP4 clips ONCE.
  2. Run baseline inference on them.
  3. Swap in the abliterated weights (in the same model instance).
  4. Hardlink each clip to a new filename (cache-busting, ~free).
  5. Run inference on the hardlinked duplicates.
  6. Delete all clips (originals + duplicates) once, at the end.

What a clean result looks like:
  Target group:  y drops sharply (large negative delta)
  General group: y barely moves (small delta, could go either direction)

What a confounded result looks like:
  Both groups drop comparably -- the ablation removed "strong face activation"
  in general, not the target person specifically.

Usage:
  python scripts/validation.py --val-dir ./val --target-prefix mia
"""

import os, sys, warnings, logging, argparse, tempfile, shutil
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from infer_fairface_bulk import get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline
from tribev2.demo_utils import TribeModel

CHECKPOINT = Path("./abliterated/vjepa2_face_abliterated.pt")
MASK_PATH  = Path("./abliterated/masks/face_mask.npy")
CACHE_DIR  = Path("./cache")
VAL_DIR    = Path("./val")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_val_images(val_dir: Path, target_prefix: str):
    """Group images by filename: names starting with target_prefix -> target,
    everything else -> general."""
    target, general = [], []
    for p in sorted(val_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        group = target if p.name.lower().startswith(target_prefix.lower()) else general
        group.append(p)
    return target, general


def build_clips_once(image_paths, tmp_dir: Path, duration: float, fps: int):
    """
    Reads each image and writes ONE synthetic clip per image, used for BOTH
    predict() passes below. Returns list of (clip_path, timeline_id, display_name).
    """
    rows = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] failed to read {img_path.name}, skipping")
            continue
        tl = f"img_{i}"
        clip_path = tmp_dir / f"{tl}.mp4"
        write_static_clip(img, clip_path, duration=duration, fps=fps)
        rows.append((clip_path, tl, img_path.name))
    return rows


def make_cache_busting_duplicates(rows, tmp_dir: Path):
    """
    IMPORTANT: TribeModel's video/image feature extractors cache output keyed
    by input filepath (infra mode='cached', keep_in_ram=True -- confirmed in
    the printed model config). Reusing the exact same clip paths for a second
    predict() call after swapping in the abliterated weights will silently
    return the FIRST pass's cached features -- the forward pass never
    actually re-runs, so the weight change has zero measurable effect even
    though the swap itself succeeded. This is the same bug from the earlier
    video-based project.

    Fix: hardlink each clip to a new filename (same bytes, new path -> new
    cache key), so the second pass is forced to recompute. A hardlink is
    essentially free (no data copy, same inode) since these live on
    /dev/shm or a local tmp filesystem.
    """
    new_rows = []
    for clip_path, tl, name in rows:
        dup_path = tmp_dir / f"{tl}_pass2.mp4"
        try:
            os.link(clip_path, dup_path)
        except OSError:
            shutil.copy2(clip_path, dup_path)  # cross-device fallback
        new_rows.append((dup_path, tl, name))
    return new_rows


def run_predict_on_clips(model, rows, duration: float):
    """rows: list of (clip_path, timeline_id, display_name). Does NOT delete clips.
    Returns dict display_name -> mean_pred (20484,)."""
    df_rows = [(clip_path, tl) for clip_path, tl, _ in rows]
    df = make_multi_row_df(df_rows, duration=duration)
    preds, segments = model.predict(events=df)
    grouped = group_preds_by_timeline(preds, segments)
    return {name: grouped[tl] for _, tl, name in rows if tl in grouped}


def load_checkpoint_state_dict(checkpoint_path: Path):
    try:
        import chunk_utils
        if chunk_utils.check_chunked_exists(checkpoint_path):
            print("  Detected chunked checkpoint -- fusing via chunk_utils...")
            return chunk_utils.load_chunked(checkpoint_path, map_location="cpu")
    except ImportError:
        pass
    return torch.load(checkpoint_path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", default=VAL_DIR, type=Path,
                        help="Folder of flat validation images (default: ./val)")
    parser.add_argument("--target-prefix", default="mia",
                        help="Filename prefix identifying the target person (default: mia)")
    parser.add_argument("--checkpoint", default=CHECKPOINT, type=Path)
    parser.add_argument("--mask", default=MASK_PATH, type=Path)
    parser.add_argument("--cache-folder", default=CACHE_DIR, type=Path)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    mask = np.load(args.mask)
    print(f"Mask: {mask.sum()} vertices")

    print(f"\nScanning {args.val_dir} ...")
    target_paths, general_paths = discover_val_images(args.val_dir, args.target_prefix)
    print(f"  {len(target_paths)} target images (prefix='{args.target_prefix}'): "
          f"{[p.name for p in target_paths]}")
    print(f"  {len(general_paths)} general images: {[p.name for p in general_paths]}")

    if not target_paths:
        sys.exit(f"[FATAL] No images matched target prefix '{args.target_prefix}' in {args.val_dir}")

    tmp_root = get_tmp_root()
    tmp_dir = Path(tempfile.mkdtemp(prefix="validation_", dir=tmp_root))
    print(f"\nTemp dir (built once, reused across both passes): {tmp_dir}")

    all_paths = target_paths + general_paths
    try:
        print("Building MP4 clips (once)...")
        rows = build_clips_once(all_paths, tmp_dir, duration=args.duration, fps=args.fps)
        print(f"  {len(rows)} clips built")

        print("\nLoading TribeModel (baseline weights)...")
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)

        print("Running baseline inference...")
        preds_before = run_predict_on_clips(model, rows, duration=args.duration)

        print(f"\nLoading abliterated checkpoint: {args.checkpoint}")
        vjepa2_module = model.data.video_feature.image.model.model
        state_dict = load_checkpoint_state_dict(args.checkpoint)
        vjepa2_module.load_state_dict(state_dict)
        print("  Loaded (same model instance, weights swapped in place).")

        print("Creating cache-busting duplicate clips for pass 2 "
              "(hardlinks -- same bytes, new filepaths so the feature-extractor "
              "cache can't return stale pre-surgery features)...")
        rows_pass2 = make_cache_busting_duplicates(rows, tmp_dir)

        print("Running post-surgery inference (on duplicated clip paths)...")
        preds_after = run_predict_on_clips(model, rows_pass2, duration=args.duration)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nTemp clips deleted: {tmp_dir}")

    # ── Compare ───────────────────────────────────────────────────────────
    def summarize(group_name, paths):
        deltas, befores = [], []
        for p in paths:
            name = p.name
            if name not in preds_before or name not in preds_after:
                print(f"  [WARN] missing prediction for {name}, skipping")
                continue
            yb = float(preds_before[name][mask].mean())
            ya = float(preds_after[name][mask].mean())
            befores.append(yb)
            deltas.append(ya - yb)
            print(f"    {name:30s}  before={yb:+.5f}  after={ya:+.5f}  delta={ya-yb:+.5f}")
        deltas = np.array(deltas)
        befores = np.array(befores)
        n_decreased = int((deltas < 0).sum())
        print(f"\n{group_name} (n={len(deltas)}):")
        print(f"  mean y before: {befores.mean():+.5f}")
        print(f"  mean delta:    {deltas.mean():+.5f}  "
              f"({deltas.mean()/max(abs(befores.mean()),1e-9)*100:+.1f}% relative)")
        print(f"  images decreased: {n_decreased}/{len(deltas)} "
              f"({n_decreased/max(len(deltas),1)*100:.0f}%)")
        return deltas.mean()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    target_delta = summarize("TARGET", target_paths)
    general_delta = summarize("GENERAL", general_paths)

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    if target_delta < 0 and general_delta >= target_delta * 0.3:
        print(">> Looks selective: target dropped meaningfully more than general population.")
    elif target_delta < 0 and abs(general_delta) > abs(target_delta) * 0.6:
        print(">> Warning: general population dropped nearly as much as target.")
        print("   This suggests the ablation is suppressing 'strong face activation' broadly,")
        print("   not the target person specifically. Consider adding an explicit")
        print("   difference-of-means contrastive component to direction-finding,")
        print("   on top of the existing weighted-PCA approach.")
    else:
        print(">> Inconclusive / target didn't drop as expected -- check tolerance sign,")
        print("   checkpoint loading, and whether the mask matches what was ablated.")

    print(f"\nNote: with only {len(target_paths)} target and {len(general_paths)} general "
          f"images, treat this as a quick directional check, not a statistically powered "
          f"validation -- worth rerunning with more held-out images per group once you're "
          f"deciding on final tolerance/layer settings.")


if __name__ == "__main__":
    main()
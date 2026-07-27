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

WHY TWO SEPARATE MODEL INSTANCES (this replaces three earlier attempts at
busting a single shared instance's cache -- all of which returned an
identical, suspiciously-exact 0.00000 delta on every image):

  TribeModel's video_feature extractor uses exca's CacheDict, which is
  disk-backed, lock-coordinated (the repo's own README documents an
  "inflight.db" used to prevent deadlocks under concurrent access), and
  almost certainly holds an OPEN handle/connection to its backing store
  once first accessed. Deleting or redirecting files out from under an
  ALREADY-OPEN handle doesn't invalidate what that handle already sees --
  on POSIX, reads through an open fd keep working against the old inode
  regardless of what happens to the directory entry. That's consistent
  with every attempt failing identically: in-memory dict clearing, on-disk
  file deletion, and cache-folder redirection all operate on the same
  live object from outside, and none of them can un-open an open handle.

  Two fully independent TribeModel instances, each pointed at its OWN
  cache folder, share no Python object and no open file handle at all.
  There is nothing to bust because nothing is shared. This sidesteps the
  problem instead of continuing to fight exca's internals from outside.

  1. Build all MP4 clips ONCE (images never change between passes).
  2. Load model_before, run baseline inference.
  3. Load model_after (SEPARATE instance, separate cache folder), load the
     abliterated checkpoint into it, run inference on the SAME clip files.
  4. Delete clips once, at the end.

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
    model instances below. Returns list of (clip_path, timeline_id, display_name).
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
    parser.add_argument("--cache-folder", default=CACHE_DIR, type=Path,
                        help="Base cache folder. Each model instance gets its own "
                             "SEPARATE subfolder under here (see --no-shared-cache-root "
                             "to fully isolate them elsewhere instead).")
    parser.add_argument("--no-shared-cache-root", action="store_true",
                        help="Put each model's cache in an isolated temp location "
                             "instead of subfolders under --cache-folder. Use this if "
                             "you suspect any sharing at the --cache-folder root level "
                             "itself (e.g. lock files) could still cause interference.")
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
    print(f"\nTemp dir (clips built once, used by BOTH model instances): {tmp_dir}")

    all_paths = target_paths + general_paths

    if args.no_shared_cache_root:
        cache_before = Path(tempfile.mkdtemp(prefix="cache_before_", dir=tmp_root))
        cache_after = Path(tempfile.mkdtemp(prefix="cache_after_", dir=tmp_root))
    else:
        cache_before = args.cache_folder / "validation_before"
        cache_after = args.cache_folder / "validation_after"
        cache_before.mkdir(parents=True, exist_ok=True)
        cache_after.mkdir(parents=True, exist_ok=True)

    print(f"model_before cache: {cache_before}")
    print(f"model_after cache:  {cache_after}")
    print("(Two separate model instances, two separate cache folders -- no shared "
          "object, no shared open file handle, nothing for a stale cache to hide in.)")

    try:
        print("\nBuilding MP4 clips (once, shared by both passes)...")
        rows = build_clips_once(all_paths, tmp_dir, duration=args.duration, fps=args.fps)
        print(f"  {len(rows)} clips built")

        print("\nLoading model_before (baseline weights)...")
        model_before = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_before)
        print("Running baseline inference...")
        preds_before = run_predict_on_clips(model_before, rows, duration=args.duration)
        del model_before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nLoading model_after (fresh instance, separate cache)...")
        model_after = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_after)
        print(f"Loading abliterated checkpoint into model_after: {args.checkpoint}")
        vjepa2_module = model_after.data.video_feature.image.model.model
        state_dict = load_checkpoint_state_dict(args.checkpoint)
        load_result = vjepa2_module.load_state_dict(state_dict, strict=False)
        print(f"  missing_keys={len(load_result.missing_keys)} "
              f"unexpected_keys={len(load_result.unexpected_keys)}")
        print("Running post-surgery inference on the SAME clip files...")
        preds_after = run_predict_on_clips(model_after, rows, duration=args.duration)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\nTemp clips deleted: {tmp_dir}")
        if args.no_shared_cache_root:
            shutil.rmtree(cache_before, ignore_errors=True)
            shutil.rmtree(cache_after, ignore_errors=True)
            print("Isolated temp cache folders deleted.")

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
    elif target_delta == 0.0 and general_delta == 0.0:
        print(">> STILL ZERO. If this happens even with two fully independent model")
        print("   instances and separate cache folders, the issue is almost certainly")
        print("   NOT caching -- go back to smoke_test_vjepa2.py's result: weights and")
        print("   raw forward pass both differed there. Suspect instead: (a) the mask")
        print("   file doesn't match what was actually operated on, (b) tolerance sign/")
        print("   magnitude produces a change too small to survive the remaining ~35")
        print("   encoder layers + TRIBE's projector at 5-decimal precision (unlikely")
        print("   to be EXACTLY 0.00000 if so, but worth printing full float precision")
        print("   to rule out rounding), or (c) --checkpoint path is stale/wrong file.")
    else:
        print(">> Inconclusive / target didn't drop as expected -- check tolerance sign,")
        print("   checkpoint loading, and whether the mask matches what was ablated.")

    print(f"\nNote: with only {len(target_paths)} target and {len(general_paths)} general "
          f"images, treat this as a quick directional check, not a statistically powered "
          f"validation -- worth rerunning with more held-out images per group once you're "
          f"deciding on final tolerance/layer settings.")


if __name__ == "__main__":
    main()
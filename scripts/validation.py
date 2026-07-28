"""
validation.py

Answers the real question after face abliteration runs: did the surgery
suppress target face identity SPECIFICALLY, or did it also dampen OFA/FFA
response to any strongly-presented face?

THREE CONFIRMED FIXES COMBINED HERE, after a long debugging chain:

1. model_name must be redirected to a LOCAL HF-format checkpoint directory,
   not a raw state_dict loaded via load_state_dict(). Confirmed via
   smoke_test_model_identity.py + smoke_test_model_name_redirect.py:
   TribeModel's video_feature.image extractor reconstructs a fresh vjepa2
   model internally on every predict() call via
   AutoModel.from_pretrained(model_name) -- it never reuses any object we
   hold a reference to and mutate.

2. That redirect must happen via model_copy(update=...), escalating up the
   config tree (image -> video_feature -> data) until a level accepts plain
   assignment -- direct attribute mutation raises a frozen-pydantic
   RuntimeError, and this freeze was observed to kick in specifically AFTER
   a model instance's first predict() call (a standalone smoke test could
   set model_name freely before ever calling predict() once). So: use a
   FRESH model instance for the post-surgery pass and apply the redirect
   before that instance's first predict() call ever.

3. Pass 2 must use genuinely distinct events (filepath AND timeline),
   not just a different model. Confirmed by observing zero "Encoding video"
   progress bars in pass 2 despite the redirect succeeding -- a pure cache
   hit across every clip. If the cache key is built from the event itself
   rather than incorporating the extractor's config, sending the identical
   event twice returns the identical cached answer regardless of what
   changed about the model. Fix: hardlinked duplicate files (same bytes,
   zero re-encode cost) with new timeline IDs for pass 2.

Usage:
  python scripts/validation.py --val-dir ./val --target-prefix mia \
      --hf-checkpoint-dir ./abliterated_face/vjepa2_hf_checkpoint
"""

import os, sys, warnings, logging, argparse, tempfile, shutil
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import cv2

sys.path.append(str(Path(__file__).parent))
from infer_fairface_bulk import get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline
from tribev2.demo_utils import TribeModel

HF_CHECKPOINT_DIR = Path("./abliterated_face/vjepa2_hf_checkpoint")
MASK_PATH  = Path("./abliterated_face/masks/face_mask.npy")
CACHE_DIR  = Path("./cache")
VAL_DIR    = Path("./val")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_val_images(val_dir: Path, target_prefix: str):
    target, general = [], []
    for p in sorted(val_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        group = target if p.name.lower().startswith(target_prefix.lower()) else general
        group.append(p)
    return target, general


def build_clips_once(image_paths, tmp_dir: Path, duration: float, fps: int):
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
    """Pass 2 needs events that are genuinely distinct from pass 1's --
    new filepaths (hardlinks, zero re-encode cost) AND new timeline IDs,
    since either alone might be part of the cache key."""
    new_rows = []
    for clip_path, tl, name in rows:
        new_path = clip_path.with_name(clip_path.stem + "_pass2" + clip_path.suffix)
        os.link(clip_path, new_path)
        new_rows.append((new_path, tl + "_pass2", name))
    return new_rows


def run_predict_on_clips(model, rows, duration: float):
    df_rows = [(clip_path, tl) for clip_path, tl, _ in rows]
    df = make_multi_row_df(df_rows, duration=duration)
    preds, segments = model.predict(events=df)
    grouped = group_preds_by_timeline(preds, segments)
    return {name: grouped[tl] for _, tl, name in rows if tl in grouped}


def redirect_model_name(model, new_model_name: str):
    """model.data.video_feature.image is a frozen pydantic model -- direct
    attribute assignment raises RuntimeError. Escalate via model_copy
    (update=...) until a level accepts plain assignment."""
    image = model.data.video_feature.image
    new_image = image.model_copy(update={"model_name": new_model_name})

    try:
        model.data.video_feature.image = new_image
        print("  Redirected at level: video_feature.image (direct)")
        return
    except Exception as e1:
        pass

    video_feature = model.data.video_feature
    new_video_feature = video_feature.model_copy(update={"image": new_image})

    try:
        model.data.video_feature = new_video_feature
        print("  Redirected at level: video_feature (rebuilt with new .image)")
        return
    except Exception as e2:
        pass

    data = model.data
    new_data = data.model_copy(update={"video_feature": new_video_feature})

    try:
        model.data = new_data
        print("  Redirected at level: data (rebuilt with new .video_feature)")
        return
    except Exception as e3:
        raise RuntimeError(
            f"Could not redirect model_name at any level.\n"
            f"  image-level error:         {e1}\n"
            f"  video_feature-level error: {e2}\n"
            f"  data-level error:          {e3}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", default=VAL_DIR, type=Path)
    parser.add_argument("--target-prefix", default="mia")
    parser.add_argument("--hf-checkpoint-dir", default=HF_CHECKPOINT_DIR, type=Path)
    parser.add_argument("--mask", default=MASK_PATH, type=Path)
    parser.add_argument("--cache-folder", default=CACHE_DIR, type=Path)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    if not args.hf_checkpoint_dir.exists():
        sys.exit(f"[FATAL] {args.hf_checkpoint_dir} does not exist. Run "
                 f"convert_to_hf_checkpoint.py first.")

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
    print(f"\nTemp dir: {tmp_dir}")

    all_paths = target_paths + general_paths

    try:
        print("\nBuilding MP4 clips (once)...")
        rows = build_clips_once(all_paths, tmp_dir, duration=args.duration, fps=args.fps)
        print(f"  {len(rows)} clips built")

        cache_before = args.cache_folder / "validation_before"
        cache_after = args.cache_folder / "validation_after"
        cache_before.mkdir(parents=True, exist_ok=True)
        cache_after.mkdir(parents=True, exist_ok=True)

        print(f"\nLoading model_before (baseline weights, cache: {cache_before})...")
        model_before = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_before)
        print("Running baseline inference...")
        preds_before = run_predict_on_clips(model_before, rows, duration=args.duration)
        del model_before

        redirect_path = str(args.hf_checkpoint_dir.resolve())
        print(f"\nLoading model_after (FRESH instance, cache: {cache_after})...")
        model_after = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_after)
        print(f"Redirecting model_name -> {redirect_path}")
        print("(Applied BEFORE this instance's first predict() call -- matches the")
        print(" condition where the standalone smoke test succeeded without hitting")
        print(" a frozen-config error.)")
        redirect_model_name(model_after, redirect_path)

        print("\nBuilding cache-busting duplicates for pass 2 (new filepaths + new")
        print("timeline IDs -- the previous attempt showed ZERO 'Encoding video' bars")
        print("in pass 2, meaning it was a pure cache hit on the event itself,")
        print("independent of model_name. This closes that gap.)")
        rows_pass2 = make_cache_busting_duplicates(rows, tmp_dir)

        print("Running post-surgery inference on genuinely distinct events "
              "(watch for 'Encoding video' bars below -- their PRESENCE this time "
              "is the real signal that computation actually happened)...")
        preds_after = run_predict_on_clips(model_after, rows_pass2, duration=args.duration)

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
        print("   The ablation may be suppressing 'strong face activation' broadly,")
        print("   not the target person specifically.")
    elif target_delta == 0.0 and general_delta == 0.0:
        print(">> STILL ZERO despite all three fixes. At this point, go back to")
        print("   convert_to_hf_checkpoint.py's checksum verification output --")
        print("   if the 'before vs after load_state_dict' checksum was identical")
        print("   there, the HF checkpoint never contained your abliteration in the")
        print("   first place, and nothing in this script could have fixed that.")
    else:
        print(">> Non-zero result -- check sign/magnitude against tolerance and mask.")

    print(f"\nNote: with only {len(target_paths)} target and {len(general_paths)} general "
          f"images, treat this as a quick directional check, not a statistically "
          f"powered validation.")


if __name__ == "__main__":
    main()
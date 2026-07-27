"""
validation.py

Answers the real question after face abliteration runs: did the surgery
suppress target face identity SPECIFICALLY, or did it also dampen OFA/FFA
response to any strongly-presented face?

Loads TribeModel twice -- once with original weights, once with the
abliterated vjepa2 state dict loaded in -- and compares mask-averaged brain
response for the SAME images before vs after surgery, separately for the
target person and a general-population holdout.

What a clean result looks like:
  Target group:  y drops sharply (large negative delta)
  General group: y barely moves (small delta, could go either direction)

What a confounded result looks like:
  Both groups drop comparably -- the ablation removed "strong face activation"
  in general, not the target person specifically.

Usage:
  python scripts/validation.py
"""

import os, sys, warnings, logging, argparse, zipfile, random, tempfile, shutil
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import cv2
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent))
from infer_fairface_bulk import (
    get_tmp_root, decode_image_from_zip, write_static_clip, make_multi_row_df,
    group_preds_by_timeline,
)
from tribev2.demo_utils import TribeModel

# ── Paths ────────────────────────────────────────────────────────────────────

GENERAL_ZIPS_DIR  = Path("./fairface")              # general-population category zips
GENERAL_PREDS_DIR = Path("./fairface_preds")         # general-population category npzs
TARGET_DIR        = Path("./target")                 # target image zips (/**/*.zip)
TARGET_PREDS_DIR  = Path("./target_preds")           # target npz predictions (/**/*.npz)
CHECKPOINT        = Path("./abliterated/vjepa2_face_abliterated.pt")
MASK_PATH         = Path("./abliterated/masks/face_mask.npy")
CACHE_DIR         = Path("./cache")


def run_predict_batch(model, images_with_names, tmp_dir, duration=1.0, fps=2):
    """images_with_names: list of (img_bgr, name). Returns dict name -> mean_pred (20484,)."""
    rows = []
    name_by_timeline = {}
    for i, (img, name) in enumerate(images_with_names):
        tl = f"val_{i}"
        clip_path = tmp_dir / f"{tl}.mp4"
        write_static_clip(img, clip_path, duration=duration, fps=fps)
        rows.append((clip_path, tl))
        name_by_timeline[tl] = name

    df = make_multi_row_df(rows, duration=duration)
    preds, segments = model.predict(events=df)
    grouped = group_preds_by_timeline(preds, segments)

    for clip_path, _ in rows:
        clip_path.unlink(missing_ok=True)

    return {name_by_timeline[tl]: vec for tl, vec in grouped.items()}


def load_target_holdout(target_zip, target_preds_npz, n):
    preds_path = Path(target_preds_npz)
    if preds_path.is_dir():
        npz_files = sorted(preds_path.rglob("*.npz"))
    elif preds_path.is_file():
        npz_files = [preds_path]
    else:
        npz_files = sorted(Path(".").rglob(str(preds_path)))
    if not npz_files:
        raise FileNotFoundError(f"No target .npz files found matching {target_preds_npz}")

    zip_path = Path(target_zip)
    if zip_path.is_dir():
        zip_files = sorted(zip_path.rglob("*.zip"))
    elif zip_path.is_file():
        zip_files = [zip_path]
    else:
        zip_files = sorted(Path(".").rglob(str(target_zip)))
    if not zip_files:
        raise FileNotFoundError(f"No target .zip files found matching {target_zip}")

    all_filenames = []
    all_preds = []
    for npz_file in npz_files:
        data = np.load(npz_file)
        all_filenames.extend(list(data["filenames"]))
        all_preds.append(data["preds"])
    all_preds = np.concatenate(all_preds, axis=0)

    idxs = list(range(len(all_filenames)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    images = []
    for i in idxs:
        name = str(all_filenames[i])
        # find matching zip by stem or fallback to first zip
        for zf_path in zip_files:
            try:
                with zipfile.ZipFile(zf_path, "r") as zf:
                    img = decode_image_from_zip(zf, name)
                images.append((img, f"target/{name}"))
                break
            except Exception:
                continue
    return images


def load_general_holdout(general_zips_dir, general_preds_dir, n_total, seed):
    rng = random.Random(seed)
    npz_files = sorted(Path(general_preds_dir).glob("*.npz"))
    per_cat = max(1, n_total // len(npz_files))
    images = []
    for npz_path in npz_files:
        zip_path = Path(general_zips_dir) / f"{npz_path.stem}.zip"
        if not zip_path.exists():
            continue
        data = np.load(npz_path)
        filenames = list(data["filenames"])
        if not filenames:
            continue
        idxs = rng.sample(range(len(filenames)), min(per_cat, len(filenames)))
        with zipfile.ZipFile(zip_path, "r") as zf:
            for i in idxs:
                name = str(filenames[i])
                try:
                    img = decode_image_from_zip(zf, name)
                    images.append((img, f"{npz_path.stem}/{name}"))
                except Exception:
                    continue
        if len(images) >= n_total:
            break
    return images[:n_total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-zip", default=TARGET_DIR, type=Path,
                        help="Zip file or directory containing target raw images (default: ./target)")
    parser.add_argument("--target-preds-npz", default=TARGET_PREDS_DIR, type=Path,
                        help="Target person .npz file or directory (default: ./target_preds)")
    parser.add_argument("--general-zips-dir", default=GENERAL_ZIPS_DIR, type=Path,
                        help="Folder of general-population .zip files (default: ./fairface)")
    parser.add_argument("--general-preds-dir", default=GENERAL_PREDS_DIR, type=Path,
                        help="Folder of general-population .npz files (default: ./fairface_preds)")
    parser.add_argument("--checkpoint", default=CHECKPOINT, type=Path,
                        help="Abliterated checkpoint (plain .pt or chunked base path, default: ./abliterated/vjepa2_face_abliterated.pt)")
    parser.add_argument("--mask", default=MASK_PATH, type=Path,
                        help="Face mask .npy file (default: ./abliterated/masks/face_mask.npy)")
    parser.add_argument("--cache-folder", default=CACHE_DIR, type=Path)
    parser.add_argument("--n-target", type=int, default=175,
                        help="Number of target images to evaluate (default: all available).")
    parser.add_argument("--n-general", type=int, default=200)
    parser.add_argument("--seed", type=int, default=999,
                        help="Different default seed from abliteration.py's sampling "
                             "to reduce overlap with the general images used in surgery.")
    args = parser.parse_args()

    random.seed(args.seed)
    mask = np.load(args.mask)
    print(f"Mask: {mask.sum()} vertices")

    print("\nLoading holdout images...")
    target_images = load_target_holdout(args.target_zip, args.target_preds_npz, args.n_target)
    general_images = load_general_holdout(args.general_zips_dir, args.general_preds_dir,
                                          args.n_general, seed=args.seed)
    print(f"  {len(target_images)} target, {len(general_images)} general")
    all_images = target_images + general_images

    tmp_root = get_tmp_root()

    # ── Baseline (original weights) ──────────────────────────────────────
    print("\nLoading TribeModel (baseline)...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)

    tmp_dir = Path(tempfile.mkdtemp(prefix="val_baseline_", dir=tmp_root))
    print("Running baseline predictions...")
    try:
        preds_before = run_predict_batch(model, all_images, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Load abliterated weights into the same model instance ───────────
    print(f"\nLoading abliterated checkpoint: {args.checkpoint}")
    vjepa2_module = model.data.video_feature.image.model.model
    try:
        import chunk_utils
        if chunk_utils.check_chunked_exists(args.checkpoint):
            print("  Detected chunked checkpoint -- fusing via chunk_utils...")
            state_dict = chunk_utils.load_chunked(args.checkpoint, map_location="cpu")
        else:
            state_dict = torch.load(args.checkpoint, map_location="cpu")
    except ImportError:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
    vjepa2_module.load_state_dict(state_dict)
    print("  Loaded.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="val_ablated_", dir=tmp_root))
    print("Running post-surgery predictions...")
    try:
        preds_after = run_predict_batch(model, all_images, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Compare ───────────────────────────────────────────────────────────
    def summarize(group_name, names):
        deltas, befores, afters = [], [], []
        for _, name in [(None, n) for n in names]:
            if name not in preds_before or name not in preds_after:
                continue
            yb = float(preds_before[name][mask].mean())
            ya = float(preds_after[name][mask].mean())
            befores.append(yb)
            afters.append(ya)
            deltas.append(ya - yb)
        deltas = np.array(deltas)
        befores = np.array(befores)
        afters = np.array(afters)
        n_decreased = int((deltas < 0).sum())
        print(f"\n{group_name} (n={len(deltas)}):")
        print(f"  mean y before: {befores.mean():+.5f}")
        print(f"  mean y after:  {afters.mean():+.5f}")
        print(f"  mean delta:    {deltas.mean():+.5f}  "
              f"({deltas.mean()/max(abs(befores.mean()),1e-9)*100:+.1f}% relative)")
        print(f"  images decreased: {n_decreased}/{len(deltas)} "
              f"({n_decreased/max(len(deltas),1)*100:.0f}%)")
        return deltas.mean()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    target_names = [n for _, n in target_images]
    general_names = [n for _, n in general_images]
    target_delta = summarize("TARGET (target person)", target_names)
    general_delta = summarize("GENERAL population", general_names)

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    if target_delta < 0 and general_delta >= target_delta * 0.3:
        print(">> Looks selective: target dropped meaningfully more than general population.")
    elif target_delta < 0 and abs(general_delta) > abs(target_delta) * 0.6:
        print(">> Warning: general population dropped nearly as much as target.")
        print("   This suggests the ablation is suppressing 'strong face activation' broadly,")
        print("   not the target person specifically. Consider adding an explicit difference-of-means")
        print("   contrastive component to the direction-finding step, on top of the")
        print("   existing weighted-PCA approach.")
    else:
        print(">> Inconclusive / target didn't drop as expected -- check tolerance sign,")
        print("   checkpoint loading, and whether the mask matches what was ablated.")

    print("\nReminder: target-group result is IN-SAMPLE (same images used in surgery) "
          "unless you pointed --target-zip/--target-preds-npz at a separate, unused photo set.")


if __name__ == "__main__":
    main()
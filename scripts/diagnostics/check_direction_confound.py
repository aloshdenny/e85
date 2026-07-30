"""
check_direction_confound.py

Tests whether the extracted contrastive direction (from
face_abliteration.py's find_directions_contrastive) is actually capturing
"Mia's facial identity" or "this photo looks different in low-level ways
(lighting/contrast/camera/compression)" -- a confound that would explain
why the general population drops as much or more than the target after
surgery, DESPITE using a true difference-of-means contrastive direction
(ruling out the weighted-PCA math as the problem).

Method: recompute simple pixel-level statistics (mean luminance, contrast)
directly from the raw images -- no vjepa2 involved -- in the SAME order
used during activation collection (same seed/sampling), then correlate
those against each image's projection onto the primary contrastive
direction (component 0 of directions_L{layer}.npy).

  |corr(luminance, projection)| large  -> the direction substantially
    entangles a photometric confound. Fix is on the DATA side: match/
    normalize luminance & contrast between target and general images
    before encoding, not further direction-finding math.

  |corr(luminance, projection)| small  -> confound isn't the (main) issue;
    something else is going on and worth a different diagnostic.

Requires face_abliteration.py's activation cache to already exist
(OUT_DIR/raw_activations/raw_X_L{layer}.npy, raw_y_L{layer}.npy) -- run
face_abliteration.py first if it doesn't.

Usage:
  python scripts/diagnostics/check_direction_confound.py \
      --general-preds-dir ./fairface_preds --general-zips-dir ./fairface \
      --target-preds-npz ./target_preds/mia.npz --target-zip ./target/mia.zip \
      --layer 6 --general-sample-size 3000 --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from abliteration import (
    load_target_images,
    sample_general_images,
    build_face_mask,
    OUT_DIR,
)


def luminance_contrast(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(gray.mean()), float(gray.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-preds-dir", required=True, type=Path)
    parser.add_argument("--general-zips-dir", required=True, type=Path)
    parser.add_argument("--target-preds-npz", required=True, type=Path)
    parser.add_argument("--target-zip", required=True, type=Path)
    parser.add_argument("--layer", type=int, required=True,
                        help="One of the layers face_abliteration.py operated on "
                             "(needs directions_L{layer}.npy and the raw activation "
                             "cache to exist for this layer).")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--general-sample-size", type=int, default=2000,
                        help="MUST match the value used in the face_abliteration.py "
                             "run you're diagnosing, so sampling order lines up with "
                             "the cached activations.")
    parser.add_argument("--seed", type=int, default=0,
                        help="MUST match the face_abliteration.py run being diagnosed.")
    parser.add_argument("--norm-enabled", dest="norm_enabled", action="store_true", default=True,
                        help="Whether the face_abliteration.py run being diagnosed had "
                             "photometric normalization ON (default) or you passed "
                             "--disable-photometric-norm there -- determines which "
                             "versioned cache directory to read from.")
    parser.add_argument("--norm-disabled", dest="norm_enabled", action="store_false")
    args = parser.parse_args()

    activation_cache_dir = OUT_DIR / f"raw_activations_{'norm' if args.norm_enabled else 'unnorm'}"
    x_path = activation_cache_dir / f"raw_X_L{args.layer}.npy"
    y_path = activation_cache_dir / f"raw_y_L{args.layer}.npy"
    dirs_path = OUT_DIR / f"directions_L{args.layer}.npy"

    for p in [x_path, y_path, dirs_path]:
        if not p.exists():
            sys.exit(f"[FATAL] {p} not found -- run face_abliteration.py first "
                     f"(with matching --general-sample-size/--seed).")

    print("Loading cached activations and directions...")
    X = np.load(x_path)
    y = np.load(y_path)
    dirs = np.load(dirs_path)
    primary_direction = dirs[0]
    print(f"  X.shape={X.shape}, primary direction norm={np.linalg.norm(primary_direction):.4f}")

    print("\nRebuilding face mask (for consistency with the run being diagnosed)...")
    mask = build_face_mask(args.include_secondary)

    print("\nReloading raw images in the SAME order used for activation collection "
          "(target_samples + general_samples)...")
    target_samples = load_target_images(args.target_preds_npz, args.target_zip, mask)
    general_samples = sample_general_images(
        args.general_preds_dir, args.general_zips_dir, mask,
        total_sample=args.general_sample_size, seed=args.seed,
    )
    all_samples = target_samples + general_samples
    n_target = len(target_samples)

    if len(all_samples) != len(X):
        print(f"  [WARN] sample count mismatch: {len(all_samples)} images reloaded vs "
              f"{len(X)} cached activations. --general-sample-size/--seed likely don't "
              f"match the original run -- results below may be misaligned. Proceed with "
              f"caution or re-verify parameters.")

    print("Computing luminance/contrast directly from raw pixels (no vjepa2 involved)...")
    luminances, contrasts = [], []
    for img, _ in all_samples:
        lum, con = luminance_contrast(img)
        luminances.append(lum)
        contrasts.append(con)
    luminances = np.array(luminances[:len(X)])
    contrasts = np.array(contrasts[:len(X)])

    projection = X @ primary_direction

    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"corr(luminance, projection onto primary direction) = {corr(luminances, projection):+.4f}")
    print(f"corr(contrast,  projection onto primary direction) = {corr(contrasts, projection):+.4f}")
    print(f"corr(luminance, y [mask-averaged brain response])  = {corr(luminances, y):+.4f}")
    print(f"corr(contrast,  y [mask-averaged brain response])  = {corr(contrasts, y):+.4f}")

    print(f"\nMean luminance -- target: {luminances[:n_target].mean():.2f}, "
          f"general: {luminances[n_target:].mean():.2f}")
    print(f"Mean contrast  -- target: {contrasts[:n_target].mean():.2f}, "
          f"general: {contrasts[n_target:].mean():.2f}")

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    lum_corr = abs(corr(luminances, projection))
    con_corr = abs(corr(contrasts, projection))
    if max(lum_corr, con_corr) > 0.35:
        print(">> Substantial photometric confound detected. The direction significantly")
        print("   entangles luminance/contrast, not just facial identity. This is a DATA")
        print("   problem, not a direction-finding math problem -- normalize/match")
        print("   luminance & contrast between target and general images before encoding,")
        print("   rather than tuning PCA components or tolerance further.")
    elif max(lum_corr, con_corr) > 0.15:
        print(">> Mild confound present -- may be contributing but probably isn't the")
        print("   whole story. Worth normalizing photometrics as a precaution, but also")
        print("   look at other nuisance factors (framing/crop, resolution, background).")
    else:
        print(">> Luminance/contrast don't explain the direction. The non-selectivity is")
        print("   likely coming from something else -- worth checking framing/crop/")
        print("   resolution differences between your target and general image sets next.")


if __name__ == "__main__":
    main()
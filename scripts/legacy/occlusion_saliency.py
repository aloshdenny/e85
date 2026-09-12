"""
occlusion_saliency.py

Projects brain-region activation back onto the SOURCE PHOTO: which pixels of
this specific image are actually driving the OFA/FFA ("is this Mia") signal?

Method: occlusion sensitivity (Zeiler & Fergus style). Slide a patch across
the image, blank it out (fill with the image's own mean color) one position
at a time, re-run the FULL TribeModel pipeline on each occluded variant, and
measure how much the face-mask-averaged brain response drops relative to the
unoccluded baseline. A big drop means that patch was carrying the signal --
build a heatmap from all positions, upsample to full resolution, overlay on
the original photo.

Chosen over gradient-based saliency (Grad-CAM etc) deliberately: this
pipeline's model.predict() path goes through several frozen-pydantic-config
layers and a job-execution framework that reconstructs modules internally
(see the earlier debugging chain in validation.py's docstring) -- getting
gradients to flow cleanly through all of that reliably is much more fragile
than just running forward passes on perturbed inputs, which is exactly what
this pipeline is already built to do well.

BEFORE/AFTER COMPARISON: if --hf-checkpoint-dir is given, the whole
occlusion sweep runs TWICE on the SAME image -- once with original weights,
once with the abliterated checkpoint (via the same model_name redirect +
cache-busting-duplicate-events technique validated in validation.py) -- and
saves a side-by-side heatmap showing which regions used to drive the signal
and no longer do.

Usage:
  # Single heatmap, baseline weights only
  python occlusion_saliency.py --image ./val/mia1_face.jpg --mask ./abliterated_face/masks/face_mask.npy

  # Before/after surgery comparison on the same photo
  python occlusion_saliency.py --image ./val/mia1_face.jpg --mask ./abliterated_face/masks/face_mask.npy \
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
from infer_fairface_bulk import (
    get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline,
)
from validation import redirect_model_name
from tribev2.demo_utils import TribeModel
try:
    from abliteration import assert_redirectable_path
except ImportError:
    def assert_redirectable_path(path, *a, **k):
        return path


def build_occlusion_grid(img_bgr, patch_frac: float, stride_frac: float):
    """
    Returns (positions, patch_px, stride_px, n_rows, n_cols) where positions
    is a list of (row, col, y0, y1, x0, x1) in the image's OWN pixel space.
    Fractional sizing (relative to the shorter image dimension) so this
    behaves consistently across source images of different resolution.
    """
    h, w = img_bgr.shape[:2]
    short_side = min(h, w)
    patch_px = max(8, int(round(short_side * patch_frac)))
    stride_px = max(4, int(round(short_side * stride_frac)))

    positions = []
    row = 0
    y0 = 0
    while y0 < h:
        col = 0
        x0 = 0
        y1 = min(y0 + patch_px, h)
        while x0 < w:
            x1 = min(x0 + patch_px, w)
            positions.append((row, col, y0, y1, x0, x1))
            x0 += stride_px
            col += 1
        y0 += stride_px
        row += 1

    n_rows = row
    n_cols = max(p[1] for p in positions) + 1
    return positions, patch_px, stride_px, n_rows, n_cols


def build_filler(img_bgr, occluder: str, patch_px: int, blur_frac: float):
    """The image an occluded patch is drawn FROM.

    gray  flat mean colour -- the original choice. Measured behaviour: hiding
          almost any patch this way RAISED the OFA/FFA response (only 9% of
          patches lowered it for Mia, 5% for a general face), because a flat
          rectangle with hard edges is itself a large out-of-distribution event
          on top of an already out-of-distribution still-held-as-video input.
          Whatever signal loss the occluder was meant to reveal is swamped by
          the model reacting to the rectangle.

    blur  the same image, heavily blurred. Removes local detail -- the thing
          occlusion is supposed to remove -- while keeping local colour,
          luminance and coarse layout, so it introduces no new edge and no new
          colour. Blurred over the WHOLE image and then copied in, rather than
          blurring the crop in isolation, so the patch still draws on the pixels
          around it instead of smearing its own border inward.
    """
    if occluder == "blur":
        sigma = max(1.0, blur_frac * patch_px)
        k = int(sigma * 4) | 1      # odd kernel covering +/-2 sigma
        return cv2.GaussianBlur(img_bgr, (k, k), sigma)
    filler = np.empty_like(img_bgr)
    filler[:] = img_bgr.reshape(-1, 3).mean(axis=0)
    return filler


def feather_mask(h_patch, w_patch, feather_frac=0.25):
    """Raised-cosine ramp around the patch border. Even a blurred patch has a
    hard boundary where it meets untouched pixels, and that boundary is its own
    artificial edge; ramping the blend over the outer band removes it."""
    def ramp(n):
        f = max(1, int(round(n * feather_frac)))
        w = np.ones(n, dtype=np.float32)
        e = 0.5 * (1 - np.cos(np.linspace(0, np.pi, f + 2)[1:-1]))
        w[:f], w[-f:] = e, e[::-1]
        return w
    return np.outer(ramp(h_patch), ramp(w_patch))[..., None]


def make_occluded_image(img_bgr, y0, y1, x0, x1, filler, feather_frac=0.25):
    occluded = img_bgr.copy()
    a = feather_mask(y1 - y0, x1 - x0, feather_frac)
    src = img_bgr[y0:y1, x0:x1].astype(np.float32)
    dst = filler[y0:y1, x0:x1].astype(np.float32)
    occluded[y0:y1, x0:x1] = (src * (1 - a) + dst * a).astype(img_bgr.dtype)
    return occluded


def run_occlusion_sweep(model, img_bgr, masks, tmp_dir, duration, fps, batch_size,
                        patch_frac, stride_frac, timeline_suffix="",
                        occluder="gray", blur_frac=0.75, feather_frac=0.25):
    """
    `masks` is {roi_name: bool mask}. Returns (baselines, heatmaps, n_rows, n_cols),
    both dicts keyed by roi name, where heatmap[row, col] = baseline_y -
    occluded_y (positive = occluding this patch REDUCED the response, i.e. the
    patch was carrying the signal).

    Every ROI is read off the SAME forward passes -- predict() returns all 20484
    vertices anyway, so control ROIs are free. That matters here: a patch whose
    occlusion drops V1 and motor cortex as much as it drops OFA/FFA is not
    telling us about face identity, it is telling us the patch carried a lot of
    low-level image energy. Without the controls a single hot heatmap is
    uninterpretable.
    """
    positions, patch_px, stride_px, n_rows, n_cols = build_occlusion_grid(
        img_bgr, patch_frac, stride_frac)
    filler = build_filler(img_bgr, occluder, patch_px, blur_frac)
    print(f"  Occlusion grid: {n_rows}x{n_cols} = {len(positions)} positions "
          f"(patch={patch_px}px, stride={stride_px}px, occluder={occluder}, "
          f"feather={feather_frac})")

    all_images = [("baseline", img_bgr)]
    for (row, col, y0, y1, x0, x1) in positions:
        occ_img = make_occluded_image(img_bgr, y0, y1, x0, x1, filler, feather_frac)
        all_images.append((f"r{row}c{col}", occ_img))

    vec_by_key = {}
    for batch_start in range(0, len(all_images), batch_size):
        batch = all_images[batch_start:batch_start + batch_size]
        rows = []
        for key, img in batch:
            tl = f"{key}{timeline_suffix}"
            clip_path = tmp_dir / f"{tl}.mp4"
            write_static_clip(img, clip_path, duration=duration, fps=fps)
            rows.append((clip_path, tl))
        df = make_multi_row_df(rows, duration=duration)
        preds, segments = model.predict(events=df)
        grouped = group_preds_by_timeline(preds, segments)
        for (key, _), (clip_path, tl) in zip(batch, rows):
            vec = grouped.get(tl)
            if vec is not None:
                vec_by_key[key] = np.asarray(vec)
            clip_path.unlink(missing_ok=True)
        done = min(batch_start + batch_size, len(all_images))
        print(f"    ...{done}/{len(all_images)}")

    if "baseline" not in vec_by_key:
        raise RuntimeError("predict() returned nothing for the unoccluded image")

    baselines, heatmaps = {}, {}
    for roi, m in masks.items():
        b = float(vec_by_key["baseline"][m].mean())
        hm = np.zeros((n_rows, n_cols), dtype=np.float32)
        for (row, col, y0, y1, x0, x1) in positions:
            key = f"r{row}c{col}"
            if key in vec_by_key:
                hm[row, col] = b - float(vec_by_key[key][m].mean())
        baselines[roi], heatmaps[roi] = b, hm

    return baselines, heatmaps, n_rows, n_cols, positions


def render_overlay(img_bgr, heatmap, out_path, title=""):
    """Signed, symmetrically scaled overlay.

    The original version clipped to positive deltas and renormalised to the
    positive max. On the first real run that was actively misleading: the FACE
    heatmap ran [-0.00705, +0.00076], i.e. occluding almost anywhere RAISED the
    response, and clipping threw away the ten-times-larger effect while blowing
    the small positive residue up to full scale. So both signs are drawn, on one
    shared scale set by max|heatmap|:

      red   occluding here LOWERED the response -- the patch was driving it
      blue  occluding here RAISED the response -- the patch was suppressing it
    """
    h, w = img_bgr.shape[:2]
    peak = float(np.abs(heatmap).max())
    norm = heatmap / peak if peak > 1e-12 else heatmap          # [-1, 1]
    up = np.clip(cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC), -1, 1)

    # 0 -> blue, 128 -> neutral, 255 -> red
    colored = cv2.applyColorMap(((up + 1) * 127.5).astype(np.uint8), cv2.COLORMAP_JET)

    alpha = np.abs(up)[..., None] * 0.65   # only tint where the effect is real
    overlay = (img_bgr.astype(np.float32) * (1 - alpha) +
              colored.astype(np.float32) * alpha).astype(np.uint8)

    if title:
        cv2.putText(overlay, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), overlay)
    print(f"  Saved -> {out_path}")
    return overlay


def patch_statistics(img_bgr, positions, n_rows, n_cols):
    """Per-patch mean luminance and contrast, on the same grid as the heatmaps.
    Correlating a saliency map against these answers, in pixel space and for
    free, the question the brain-surface maps could not: is the 'salient'
    structure just local contrast? A face map that correlates with patch
    contrast as strongly as V1's does is measuring the photometric confound,
    not face identity."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_g = np.zeros((n_rows, n_cols), dtype=np.float32)
    std_g = np.zeros((n_rows, n_cols), dtype=np.float32)
    for (row, col, y0, y1, x0, x1) in positions:
        patch = gray[y0:y1, x0:x1]
        mean_g[row, col] = patch.mean()
        std_g[row, col] = patch.std()
    return mean_g, std_g


def _corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def summarize_sweep(label, baselines, heatmaps, patch_mean=None, patch_std=None):
    """Prints each ROI's unoccluded level and heatmap range, plus the number
    that actually decides interpretation: the spatial correlation between the
    FACE heatmap and each control's. Near +1 means the two ROIs are responding
    to the same pixels, so the face map is not carrying face-specific
    information -- it is tracking whatever low-level energy the occluder
    removed."""
    face = heatmaps.get("FACE")
    print(f"  [{label}]")
    for roi, hm in heatmaps.items():
        pos = float((hm > 0).mean())
        line = (f"    {roi:6s} baseline={baselines[roi]:+.5f}  "
                f"range [{hm.min():+.6f}, {hm.max():+.6f}]  mean={hm.mean():+.6f}  "
                f"{pos:.0%} of patches lowered response")
        print(line)
        sub = []
        if face is not None and roi != "FACE":
            sub.append(f"vs FACE map r={_corr(face, hm):+.3f}")
        if patch_std is not None:
            sub.append(f"vs patch contrast r={_corr(hm, patch_std):+.3f}")
            sub.append(f"vs patch luminance r={_corr(hm, patch_mean):+.3f}")
        if sub:
            print("           " + "   ".join(sub))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path,
                        help="Face ROI mask .npy (e.g. abliterated_face/masks/face_mask.npy)")
    parser.add_argument("--hf-checkpoint-dir", type=Path, default=None,
                        help="If given, also runs the sweep with the abliterated checkpoint "
                             "and saves a before/after comparison.")
    parser.add_argument("--cache-folder", default="./cache", type=Path)
    parser.add_argument("--out-dir", default="./saliency_out", type=Path)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-frac", type=float, default=0.15,
                        help="Occlusion patch size as a fraction of the image's "
                             "shorter dimension.")
    parser.add_argument("--occluder", default="gray", choices=["gray", "blur"],
                        help="What replaces an occluded patch. 'blur' removes local "
                             "detail without introducing a new edge or colour; 'gray' is "
                             "the original flat mean-colour fill.")
    parser.add_argument("--blur-frac", type=float, default=0.75,
                        help="Gaussian sigma for --occluder blur, as a fraction of the "
                             "patch size.")
    parser.add_argument("--feather-frac", type=float, default=0.25,
                        help="Fraction of the patch used for the raised-cosine blend ramp "
                             "at its border. 0 gives a hard-edged patch.")
    parser.add_argument("--no-controls", action="store_true",
                        help="Only map the face mask. By default control ROIs (V1, MOTOR) "
                             "are mapped from the same forward passes at no extra cost, "
                             "because a face heatmap that matches the V1 heatmap is "
                             "low-level image energy rather than face selectivity.")
    parser.add_argument("--stride-frac", type=float, default=0.08,
                        help="Occlusion stride as a fraction of the image's shorter "
                             "dimension. Smaller = smoother heatmap, more positions "
                             "(slower).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    masks = {"FACE": np.load(args.mask)}
    if not args.no_controls:
        from measure_identity_signal import build_masks as build_all_masks
        allm = build_all_masks()
        for roi in ["OFA", "FFA", "V1", "MOTOR"]:
            if roi in allm:
                masks[roi] = allm[roi]
    print("ROIs: " + ", ".join(f"{k}({int(v.sum())})" for k, v in masks.items()))

    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        sys.exit(f"[FATAL] Could not read {args.image}")
    print(f"Image: {args.image.name}, shape={img_bgr.shape}")

    tmp_root = get_tmp_root()
    tmp_dir = Path(tempfile.mkdtemp(prefix="occlusion_", dir=tmp_root))

    stem = f"{args.image.stem}_{args.occluder}"

    try:
        print("\nLoading TribeModel (baseline weights)...")
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)

        print("\nRunning occlusion sweep (baseline)...")
        base_b, hm_before, n_rows, n_cols, positions = run_occlusion_sweep(
            model, img_bgr, masks, tmp_dir, args.duration, args.fps, args.batch_size,
            args.patch_frac, args.stride_frac, occluder=args.occluder,
            blur_frac=args.blur_frac, feather_frac=args.feather_frac,
        )
        pmean, pstd = patch_statistics(img_bgr, positions, n_rows, n_cols)
        summarize_sweep("BEFORE", base_b, hm_before, pmean, pstd)
        np.savez(args.out_dir / f"{stem}_saliency_before.npz",
                 patch_mean=pmean, patch_std=pstd,
                 **{f"hm_{k}": v for k, v in hm_before.items()},
                 **{f"base_{k}": v for k, v in base_b.items()})

        overlay_before = render_overlay(
            img_bgr, hm_before["FACE"], args.out_dir / f"{stem}_saliency_before.png",
            title="BEFORE surgery",
        )
        for roi, hm in hm_before.items():
            if roi != "FACE":
                render_overlay(img_bgr, hm,
                               args.out_dir / f"{stem}_saliency_before_{roi}.png",
                               title=f"BEFORE / {roi}")

        if args.hf_checkpoint_dir is not None:
            # FRESH instance, redirected before it has ever called predict().
            # `model` above has already run the whole baseline sweep, and a
            # TribeModel's config tree freezes after its first predict(): the
            # redirect would appear to succeed while the extractor kept serving
            # the ORIGINAL weights, so the "after" heatmap would silently be a
            # second copy of the "before" one.
            redirect_path = assert_redirectable_path(args.hf_checkpoint_dir,
                                                     "hf checkpoint dir")
            del model
            model_after = TribeModel.from_pretrained("facebook/tribev2",
                                                     cache_folder=args.cache_folder)
            print(f"\nRedirecting a fresh model_name -> {redirect_path}")
            redirect_model_name(model_after, redirect_path)

            print("Running occlusion sweep (post-surgery, cache-busted timeline suffix)...")
            base_a, hm_after, _, _, _ = run_occlusion_sweep(
                model_after, img_bgr, masks, tmp_dir, args.duration, args.fps,
                args.batch_size, args.patch_frac, args.stride_frac,
                timeline_suffix="_after", occluder=args.occluder,
                blur_frac=args.blur_frac, feather_frac=args.feather_frac,
            )
            if np.array_equal(hm_after["FACE"], hm_before["FACE"]):
                raise RuntimeError(
                    "post-surgery heatmap is bit-identical to the baseline one -- "
                    "the redirect did not take effect, or predict() served cached "
                    "results. Do not read the comparison.")
            summarize_sweep("AFTER", base_a, hm_after, pmean, pstd)
            np.savez(args.out_dir / f"{stem}_saliency_after.npz",
                     **{f"hm_{k}": v for k, v in hm_after.items()},
                     **{f"base_{k}": v for k, v in base_a.items()})
            print(f"  Whole-image FACE delta from surgery: "
                  f"{base_a['FACE'] - base_b['FACE']:+.5f}")

            overlay_after = render_overlay(
                img_bgr, hm_after["FACE"], args.out_dir / f"{stem}_saliency_after.png",
                title="AFTER surgery",
            )

            side_by_side = np.concatenate([overlay_before, overlay_after], axis=1)
            comparison_path = args.out_dir / f"{stem}_saliency_comparison.png"
            cv2.imwrite(str(comparison_path), side_by_side)
            print(f"\nSide-by-side comparison -> {comparison_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
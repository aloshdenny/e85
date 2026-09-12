"""
part_identity_direction.py

Derives the ablation direction from the face PARTS the paired occlusion sweep
showed carry the target's identity, instead of from a whole-image target-vs-
general contrast.

Why this is not just another contrast:

  find_directions_contrastive uses normalize(mean(X_target) - mean(X_general)),
  and identity_direction.py already documents why that fails here -- every
  target photo comes from one narrow source, so the contrast is dominated by
  photographic provenance, which is perfectly confounded with the target.
  Sharper statistics cannot separate a confound that complete.

  This script never takes a between-image difference of raw activations. For
  each image it takes a WITHIN-image difference:

      delta_i^p = act(image_i) - act(image_i with part p blurred)

  Both terms are the same photograph under the same lighting, camera and
  compression, so provenance cancels inside each image before any target-vs-
  general comparison happens. Only the deltas are then contrasted:

      d_p = normalize( mean_target(delta^p) - mean_general(delta^p) )

  d_p reads as "the activation component part p contributes for THIS person,
  beyond what part p contributes for faces in general".

Which parts to use comes from the full-scale paired sweep: forehead and jawline
carry roughly twice the face-ROI weight for the target that they carry for
general faces, while eyes carry less than half. Those are the defaults.

Directions are captured at VJEPA2Layer OUTPUT, pooled over tokens -- the same
residual-stream basis abliterationv2's surgery operates in -- and saved as
"{method_name}_L{layer}", so eval_erasure.py consumes them directly with
--direction partid --direction-file <out>.

The run also reports, per layer, the cosine between each part direction and the
plain contrastive direction computed from the very same baseline activations.
A small cosine is the evidence that this is reaching a subspace the whole-image
contrast does not, rather than relabelling it.

Usage:
  python scripts/part_identity_direction.py --layers 25 30 35 \
      --n-target 150 --n-general 300 --cache-folder /home/research/e85_cache
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from abliteration import DEVICE, OUT_DIR, TribeModel, free, image_to_vjepa_input
from occlusion_saliency import build_filler
from part_occlusion_map import apply_masked_blur, collect_images, part_masks


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=[25, 30, 35])
    ap.add_argument("--parts", nargs="+", default=["forehead", "jawline"],
                    help="Parts whose within-image contribution defines the "
                         "direction. Defaults are the two the paired sweep "
                         "found most target-specific.")
    ap.add_argument("--n-target", type=int, default=150)
    ap.add_argument("--n-general", type=int, default=300)
    ap.add_argument("--blur-frac", type=float, default=0.10,
                    help="Blur sigma as a fraction of face width. Matches the "
                         "occlusion sweep so the manipulation is identical.")
    ap.add_argument("--predictor", type=Path,
                    default=Path("./models/shape_predictor_68_face_landmarks.dat"))
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path,
                    default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--method-name", default="partid",
                    help="Key prefix in the output npz; pass the same value to "
                         "eval_erasure.py --direction.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_path = args.out or (OUT_DIR / "part_identity_directions.npz")

    import dlib
    det = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(str(args.predictor))

    imgs = collect_images(args.target_zip, args.target_preds_npz,
                          args.general_preds_dir, args.general_zip,
                          args.n_target, args.n_general,
                          args.train_seed, args.train_size, args.seed)
    print(f"{len(imgs)} images ({sum(t for _, t in imgs)} target)")

    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    vjepa2 = model.data.video_feature.image.model.model
    blocks = vjepa2.encoder.layer
    vjepa2.eval().to(DEVICE)

    # One forward pass feeds every requested layer: hook them all at once
    # rather than re-running the encoder per layer.
    buf = {}

    def make_hook(L):
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            buf[L] = hidden.mean(dim=1).squeeze(0).detach().cpu().float().numpy()
        return hook_fn

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in args.layers]

    def activations(img):
        clip = image_to_vjepa_input(img).unsqueeze(0).to(DEVICE)
        buf.clear()
        with torch.no_grad():
            vjepa2(pixel_values_videos=clip)
        del clip
        return {L: buf[L].copy() for L in args.layers if L in buf}

    # Running sums keyed (layer, part, is_target) for the deltas, plus the raw
    # baseline activations so the naive contrast can be computed for comparison.
    d_sum = {(L, p, t): None for L in args.layers for p in args.parts for t in (0, 1)}
    d_n = {(L, p, t): 0 for L in args.layers for p in args.parts for t in (0, 1)}
    b_sum = {(L, t): None for L in args.layers for t in (0, 1)}
    b_n = {(L, t): 0 for L in args.layers for t in (0, 1)}

    def accum(store, counts, key, vec):
        if store[key] is None:
            store[key] = vec.astype(np.float64)
        else:
            store[key] += vec
        counts[key] += 1

    skipped = 0
    try:
        for n, (img, is_t) in enumerate(imgs):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = det(gray, 0 if min(gray.shape[:2]) >= 300 else 1)
            if not faces:
                skipped += 1
                continue
            pts = np.array([[p.x, p.y] for p in sp(gray, faces[0]).parts()],
                           dtype=np.float32)
            pm = part_masks(img.shape[:2], pts)
            missing = [p for p in args.parts if p not in pm]
            if missing:
                raise SystemExit(f"unknown part(s) {missing}; available: {sorted(pm)}")

            face_w = float(np.ptp(pts[:, 0])) or 1.0
            filler = build_filler(img, "blur", int(round(args.blur_frac * face_w)), 1.0)
            feather = max(3, int(0.06 * face_w))

            base = activations(img)
            t = int(bool(is_t))
            for L, v in base.items():
                accum(b_sum, b_n, (L, t), v)

            for p in args.parts:
                blurred = apply_masked_blur(img, filler, pm[p], feather)
                got = activations(blurred)
                for L in args.layers:
                    if L in got and L in base:
                        accum(d_sum, d_n, (L, p, t), base[L] - got[L])

            if (n + 1) % 25 == 0:
                print(f"  ...{n + 1}/{len(imgs)}", flush=True)
            free()
    finally:
        for h in handles:
            h.remove()

    print(f"\nskipped {skipped} images with no dlib detection\n")

    saved = {}
    print(f"{'layer':>5s} {'part':>10s} {'|dT|':>9s} {'|dG|':>9s} "
          f"{'cos(d,contrast)':>16s}")
    print("-" * 56)
    for L in args.layers:
        if b_n[(L, 1)] == 0 or b_n[(L, 0)] == 0:
            print(f"  L{L}: no usable images on one side, skipping")
            continue
        contrast = unit(b_sum[(L, 1)] / b_n[(L, 1)] - b_sum[(L, 0)] / b_n[(L, 0)])
        saved[f"contrast_L{L}"] = contrast.astype(np.float32)[None, :]

        combined = np.zeros_like(contrast, dtype=np.float64)
        for p in args.parts:
            if d_n[(L, p, 1)] == 0 or d_n[(L, p, 0)] == 0:
                continue
            dt = d_sum[(L, p, 1)] / d_n[(L, p, 1)]
            dg = d_sum[(L, p, 0)] / d_n[(L, p, 0)]
            diff = dt - dg
            combined += diff
            q = unit(diff)
            saved[f"{args.method_name}_{p}_L{L}"] = q.astype(np.float32)[None, :]
            print(f"{L:5d} {p:>10s} {np.linalg.norm(dt):9.4f} "
                  f"{np.linalg.norm(dg):9.4f} {cosine(q, contrast):16.3f}")

        q = unit(combined)
        saved[f"{args.method_name}_L{L}"] = q.astype(np.float32)[None, :]
        print(f"{L:5d} {'COMBINED':>10s} {'':>9s} {'':>9s} "
              f"{cosine(q, contrast):16.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **saved)
    print(f"\nsaved {len(saved)} directions -> {out_path}")
    print(f"evaluate with:  python scripts/eval_erasure.py --direction "
          f"{args.method_name} --direction-file {out_path} "
          f"--layers {' '.join(str(L) for L in args.layers)}")


if __name__ == "__main__":
    main()

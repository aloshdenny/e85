"""
photometric_null_test.py

Turns a regression argument into a causal demonstration.

likeness_gradient.py showed that 21 photometric and geometric attributes, fit
on general faces alone, predict 112% of the target's FACE(OFA+FFA) elevation
and 115% of OFA -- leaving nothing for identity. That is inference from a
linear model. This script intervenes instead: it rewrites the target's own
photographs so their photometry matches the general population, feeds them back
through TRIBE, and measures what is left of the gap.

  If the elevation is photographic, it collapses toward zero after matching.
  If any of it is about who is in the picture, a residual survives -- and that
  residual, not the raw +0.0226, is what an abliteration would need to remove.

Both arms run through the identical clip-and-predict pipeline in the same
process, so the comparison is paired and nothing is inherited from cache.

What is matched, and what deliberately is not:

  matched     luminance, contrast, sharpness, saturation -- global photometry,
              adjustable without inventing image content. Applied blur-first
              then exact affine, since blurring perturbs contrast.
  optional    face_area_frac, via reflect padding (--match-framing). Off by
              default: padding fabricates surroundings, and the decomposition
              printed first says whether framing is even a live contributor.
  untouched   pose, interocular distance, background uniformity. These cannot
              be changed without warping the face or synthesising a scene, so
              any residual gap is reported knowing they still differ.

Usage:
  python scripts/photometric_null_test.py --cache-folder /home/research/e85_cache
"""

import argparse
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR, TribeModel, decode_image_from_zip, free
from chunk_utils import ensure_fused_zip
from face_attributes import ATTR_NAMES
from infer_fairface_bulk import (
    get_tmp_root, group_preds_by_timeline, make_multi_row_df, write_static_clip,
)
from measure_identity_signal import build_masks

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR", "STS"]

# The globals this script can actually manipulate, and how they are measured.
GLOBAL_ATTRS = ["luminance", "contrast", "sharpness", "saturation", "colorfulness"]


def measure_globals(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b_, g_, r_ = [img[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r_ - g_, 0.5 * (r_ + g_) - b_
    return {
        "luminance": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "saturation": float(hsv[..., 1].mean()),
        "colorfulness": float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                             + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)),
    }


def blur_to_sharpness(img, target):
    """Gaussian blur, sigma found by bisection, to bring Laplacian variance down
    to `target`. Only ever reduces sharpness -- synthesising detail that was
    never captured would be a different manipulation entirely."""
    cur = measure_globals(img)["sharpness"]
    if target >= cur or target <= 0:
        return img
    lo, hi = 0.0, 8.0
    out = img
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        cand = cv2.GaussianBlur(img, (0, 0), mid) if mid > 1e-3 else img
        if measure_globals(cand)["sharpness"] > target:
            lo = mid
        else:
            hi = mid
        out = cand
    return out


def match_saturation(img, target):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    cur = hsv[..., 1].mean()
    if cur > 1e-6:
        hsv[..., 1] = np.clip(hsv[..., 1] * (target / cur), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def match_luma_contrast(img, target_lum, target_con):
    """Exact affine in intensity: scale about the mean to set contrast, then
    shift to set luminance. Applied identically to all channels so hue is
    preserved."""
    x = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur_lum, cur_con = gray.mean(), gray.std()
    alpha = (target_con / cur_con) if cur_con > 1e-6 else 1.0
    x = (x - cur_lum) * alpha + target_lum
    return np.clip(x, 0, 255).astype(np.uint8)


def pad_to_face_frac(img, cur_frac, target_frac):
    """Reflect-pad so the face occupies `target_frac` of the frame."""
    if target_frac <= 0 or target_frac >= cur_frac:
        return img
    scale = np.sqrt(cur_frac / target_frac)
    h, w = img.shape[:2]
    nh, nw = int(round(h * scale)), int(round(w * scale))
    py, px = (nh - h) // 2, (nw - w) // 2
    return cv2.copyMakeBorder(img, py, nh - h - py, px, nw - w - px,
                              cv2.BORDER_REFLECT_101)


def normalize(img, targets, cur_face_frac=None, target_face_frac=None):
    out = img
    if target_face_frac is not None and cur_face_frac:
        out = pad_to_face_frac(out, cur_face_frac, target_face_frac)
    out = blur_to_sharpness(out, targets["sharpness"])
    out = match_saturation(out, targets["saturation"])
    out = match_luma_contrast(out, targets["luminance"], targets["contrast"])
    return out


def predict_batch(model, images, names, tmp_root, fp16=True, workers=8):
    td = Path(tempfile.mkdtemp(prefix="photonull_", dir=tmp_root))
    try:
        rows = [(td / f"{nm}.mp4", nm) for nm in names]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda a: write_static_clip(a[0], a[1], duration=1.0, fps=2),
                        [(im, cp) for im, (cp, _) in zip(images, rows)]))
        df = make_multi_row_df(rows, duration=1.0)
        if fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                preds, segs = model.predict(events=df)
        else:
            preds, segs = model.predict(events=df)
        return group_preds_by_timeline(preds, segs)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def decompose(A, is_t, det, preds, masks):
    """Split the predicted target-general gap into per-attribute contributions,
    so the normalisation targets whatever actually drives it."""
    g, t = det & ~is_t, det & is_t
    mu, sd = A[g].mean(0), np.maximum(A[g].std(0), 1e-9)
    Cg = np.column_stack([np.ones(int(g.sum())), (A[g] - mu) / sd])
    y = preds[:, masks["FACE(OFA+FFA)"]].mean(1)
    coef, *_ = np.linalg.lstsq(Cg, y[g], rcond=None)
    zt = ((A[t] - mu) / sd).mean(0)
    contrib = coef[1:] * zt          # general z-mean is 0 by construction
    order = np.argsort(-np.abs(contrib))
    print("Which attributes drive the predicted FACE gap "
          f"(total {contrib.sum():+.5f})")
    print(f"{'attribute':>20s} {'z(target)':>10s} {'contribution':>13s}")
    print("-" * 46)
    for i in order[:10]:
        print(f"{ATTR_NAMES[i]:>20s} {zt[i]:+10.2f} {contrib[i]:+13.5f}")
    return contrib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--match-framing", action="store_true",
                    help="Also reflect-pad to match face_area_frac. Off by "
                         "default because padding invents image content.")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--save-examples", type=int, default=4)
    args = ap.parse_args()

    d = np.load(OUT_DIR / "attributes.npz", allow_pickle=True)
    A = d["attrs"].astype(np.float64)
    P = d["preds"].astype(np.float64)
    is_t = d["is_target"].astype(bool)
    det = d["detected"].astype(bool)
    masks = build_masks()
    rois = [r for r in ROIS if r in masks]

    decompose(A, is_t, det, P, masks)

    gi = {n: i for i, n in enumerate(ATTR_NAMES)}
    g = det & ~is_t
    targets = {k: float(A[g, gi[k]].mean()) for k in GLOBAL_ATTRS}
    tgt_face_frac = float(A[g, gi["face_area_frac"]].mean())
    print(f"\ngeneral-population targets: "
          + ", ".join(f"{k}={v:.1f}" for k, v in targets.items()))
    print(f"target photos currently: "
          + ", ".join(f"{k}={A[det & is_t, gi[k]].mean():.1f}" for k in GLOBAL_ATTRS))
    if args.match_framing:
        print(f"framing: matching face_area_frac to {tgt_face_frac:.3f} "
              f"(target photos {A[det & is_t, gi['face_area_frac']].mean():.3f})")

    z = ensure_fused_zip(args.target_zip)
    import zipfile
    with zipfile.ZipFile(z) as zf:
        # Archives zipped on macOS carry a __MACOSX/._name resource fork beside
        # every real entry; those are not images and decoding them throws.
        names = [n for n in zf.namelist()
                 if not n.endswith("/")
                 and not n.startswith("__MACOSX/")
                 and not Path(n).name.startswith("._")
                 and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp",
                                                ".bmp"}]
        imgs, bad = [], 0
        for n in names:
            try:
                im = decode_image_from_zip(zf, n)
            except Exception:
                im = None
            if im is None:
                bad += 1
                continue
            imgs.append((n, im))
    if bad:
        print(f"  skipped {bad} undecodable entries")
    print(f"\n{len(imgs)} target images loaded")

    ff_by_name = {}
    if args.match_framing:
        allnames = [str(x) for x in d["names"]]
        for i, nm in enumerate(allnames):
            if is_t[i] and det[i]:
                ff_by_name[Path(nm).name] = float(A[i, gi["face_area_frac"]])

    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    tmp_root = get_tmp_root()

    before = {r: [] for r in rois}
    after = {r: [] for r in rois}
    pre_globals, post_globals = [], []
    saved = 0
    ex_dir = OUT_DIR / "photometric_examples"

    for s in range(0, len(imgs), args.batch):
        chunk = imgs[s:s + args.batch]
        orig = [im for _, im in chunk]
        norm = []
        for nm, im in chunk:
            cff = ff_by_name.get(Path(nm).name) if args.match_framing else None
            norm.append(normalize(im, targets, cff,
                                  tgt_face_frac if args.match_framing else None))
        pre_globals += [measure_globals(im) for im in orig]
        post_globals += [measure_globals(im) for im in norm]

        if saved < args.save_examples:
            ex_dir.mkdir(parents=True, exist_ok=True)
            for k in range(min(args.save_examples - saved, len(chunk))):
                cv2.imwrite(str(ex_dir / f"ex{saved + k}_before.jpg"), orig[k])
                cv2.imwrite(str(ex_dir / f"ex{saved + k}_after.jpg"), norm[k])
            saved += min(args.save_examples - saved, len(chunk))

        tags_o = [f"o{s + i}" for i in range(len(chunk))]
        tags_n = [f"n{s + i}" for i in range(len(chunk))]
        grouped = predict_batch(model, orig + norm, tags_o + tags_n, tmp_root,
                                fp16=not args.no_fp16)
        for tag, store in ((tags_o, before), (tags_n, after)):
            for tg in tag:
                if tg not in grouped:
                    continue
                v = np.asarray(grouped[tg])
                v = v.mean(0) if v.ndim > 1 else v
                for r in rois:
                    store[r].append(float(v[masks[r]].mean()))
        print(f"  ...{min(s + args.batch, len(imgs))}/{len(imgs)}", flush=True)
        free()

    print("\nDid the normalisation actually land?")
    print(f"{'attribute':>14s} {'target photos':>14s} {'normalised':>12s} "
          f"{'general goal':>13s}")
    print("-" * 58)
    for k in GLOBAL_ATTRS:
        pre = np.mean([x[k] for x in pre_globals])
        post = np.mean([x[k] for x in post_globals])
        print(f"{k:>14s} {pre:14.1f} {post:12.1f} {targets[k]:13.1f}")

    gen_resp = {r: P[g][:, masks[r]].mean(1).mean() for r in rois}
    print("\nFACE-cortex response, paired through one pipeline")
    print(f"{'ROI':>14s} {'target':>10s} {'normalised':>11s} {'general':>10s} "
          f"{'gap before':>11s} {'gap after':>10s} {'removed':>9s}")
    print("-" * 82)
    for r in rois:
        b = float(np.mean(before[r])) if before[r] else float("nan")
        a = float(np.mean(after[r])) if after[r] else float("nan")
        gg = gen_resp[r]
        gap_b, gap_a = b - gg, a - gg
        rem = (100 * (gap_b - gap_a) / gap_b) if abs(gap_b) > 1e-9 else 0.0
        print(f"{r:>14s} {b:+10.5f} {a:+11.5f} {gg:+10.5f} {gap_b:+11.5f} "
              f"{gap_a:+10.5f} {rem:8.0f}%")

    fb = float(np.mean(before["FACE(OFA+FFA)"])) - gen_resp["FACE(OFA+FFA)"]
    fa = float(np.mean(after["FACE(OFA+FFA)"])) - gen_resp["FACE(OFA+FFA)"]
    print("\n" + "=" * 82)
    if abs(fb) > 1e-9 and abs(fa) < 0.25 * abs(fb):
        print("RESULT: the elevation was photographic. Matching global photometry")
        print(f"  alone collapsed the FACE gap from {fb:+.5f} to {fa:+.5f}. There is")
        print("  no identity-specific response left to abliterate.")
    elif abs(fb) > 1e-9 and abs(fa) < 0.6 * abs(fb):
        print("RESULT: mostly photographic, with a residual.")
        print(f"  FACE gap {fb:+.5f} -> {fa:+.5f}. The remainder is the only")
        print("  honest abliteration target, and it is far smaller than assumed.")
        print("  Pose, framing and background were left unmatched and may hold it.")
    else:
        print("RESULT: the gap survives photometric matching.")
        print(f"  FACE gap {fb:+.5f} -> {fa:+.5f}. Something beyond global")
        print("  photometry drives it; framing and pose are the next suspects.")
    print("=" * 82)
    if saved:
        print(f"\nbefore/after examples written to {ex_dir}")


if __name__ == "__main__":
    main()

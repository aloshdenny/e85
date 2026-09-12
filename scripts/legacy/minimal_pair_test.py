"""
minimal_pair_test.py

Minimal-pair testing: hold the photographic container fixed and vary only who
is in it.

Every previous end-to-end number in this project compared raw decoded photos --
target photos from one source against FairFace/FFHQ photos from another. Under
that comparison the model's predicted face-region response separates the target
at AUC 0.810, but provenance alone separates FairFace from FFHQ at AUC 0.74-1.00
in EVERY ROI including auditory and motor cortex, and 21 trivial scalars explain
58% of predicted face-cortex variance with luminance carrying beta +0.48. So
0.810 has never been shown to be about the person.

Three conditions, each stripping more of the container away:

  raw       as decoded. Reproduces the historical number.
  matched   every face re-cropped so the detected face box covers a fixed
            fraction of a fixed-size frame, then photometrically normalised to
            a common mean and standard deviation. This equalises crop
            tightness, resolution, luminance and contrast -- the measured
            confounds -- leaving background and pose.
  composite 2x2 grids of four matched tiles, in swap pairs: A = [target, G1,
            G2, G3] and B = [G4, G1, G2, G3], identical but for one quadrant.
            Paired, so the three shared tiles cancel exactly and the only
            surviving difference is who is in the fourth.

The composite condition is the one that needs its own sanity check, printed
below: a four-face collage is further out of distribution than a portrait, so
if the composite's face-region response does not even resemble a single face's,
its deltas mean nothing regardless of significance.

Usage:
  python scripts/minimal_pair_test.py --n-target 30 --n-general 30 --n-pairs 30
"""

import os, sys, argparse, random, zipfile, tempfile, shutil
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from abliteration import decode_image_from_zip, normalize_photometrics, OUT_DIR, free, TribeModel
from chunk_utils import discover_npz, load_npz, npz_image_names, ensure_fused_zip
from infer_fairface_bulk import (
    get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline,
)
from measure_identity_signal import build_masks, auc

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR"]
TILE = 128
FRAME = 256


def standardize(img, box, out_size, face_frac=1.4, max_frac_err=0.05):
    """Re-crop so the detected face box covers `face_frac` of the frame, then
    normalise luminance and contrast. Kills crop tightness, resolution,
    brightness and contrast in one step; background and pose survive.

    Returns (image, achieved_face_frac) or (None, achieved) if the image cannot
    meet the spec from real pixels.

    face_frac defaults ABOVE 1.0 because every image in this project is already
    a face crop: the detected box is 0.91x the short side for target photos and
    1.17x for general ones -- a 29% crop-tightness gap, itself one of the
    confounds. You cannot zoom out of a crop, only in, so the common target has
    to be tighter than the tightest source. 1.4 keeps nearly everything and, by
    construction, leaves every kept image at EXACTLY the same face/frame ratio.
    The cost is that it crops to the inner face: hair, hairline and background
    fall outside, which removes the forehead band where the one target-specific
    part effect lives. That is the trade -- this condition tests inner-face
    identity with the container held fixed.

    NOTHING IS SYNTHESISED. The first version padded with BORDER_REPLICATE when
    the required box ran off the edge, which smeared edge pixels into long
    streaks -- and it did so mostly on the general set, whose faces sit closer
    to the frame edge, manufacturing precisely the target-vs-general difference
    a minimal pair exists to remove. An image that cannot supply the crop is
    dropped instead; with 99.9% detection there are plenty of others.
    """
    h, w = img.shape[:2]
    if box is None:
        return None, 0.0
    x0, y0, x1, y1 = [float(v) for v in box]
    face = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    side = min(face / face_frac, float(min(h, w)))     # never exceed the image
    achieved = face / side
    if achieved > face_frac * (1 + max_frac_err):
        return None, achieved                          # face too big for the frame

    a = int(round(min(max(cx - side / 2, 0), w - side)))
    b = int(round(min(max(cy - side / 2, 0), h - side)))
    crop = img[b:b + int(side), a:a + int(side)]
    if crop.size == 0:
        return None, achieved
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return normalize_photometrics(crop), achieved


def load_set(zip_path, names, n, rng, mtcnn):
    out = []
    with zipfile.ZipFile(ensure_fused_zip(zip_path), "r") as zf:
        for nm in rng.sample(list(names), min(n * 3, len(names))):
            if len(out) >= n:
                break
            try:
                img = decode_image_from_zip(zf, nm)
            except Exception:
                continue
            bx, _ = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if bx is None or not len(bx):
                continue
            out.append((nm, img, bx[0]))
    return out


def predict_images(model, imgs, tmp_root, tag, fp16=True):
    td = Path(tempfile.mkdtemp(prefix=f"mp_{tag}_", dir=tmp_root))
    try:
        rows = []
        for i, im in enumerate(imgs):
            tl = f"{tag}_{i}"
            p = td / f"{tl}.mp4"
            write_static_clip(im, p, duration=1.0, fps=2)
            rows.append((p, tl))
        df = make_multi_row_df(rows, duration=1.0)
        if fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                preds, segs = model.predict(events=df)
        else:
            preds, segs = model.predict(events=df)
        g = group_preds_by_timeline(preds, segs)
        return np.stack([np.asarray(g[tl]) for _, tl in rows if tl in g])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def report(name, Pt, Pg, masks):
    print(f"\n  {name}")
    for roi in ROIS:
        m = masks[roi]
        t, g = Pt[:, m].mean(1), Pg[:, m].mean(1)
        print(f"    {roi.split('(')[0]:6s} AUC={auc(t, g):.3f}  "
              f"target={t.mean():+.5f}  general={g.mean():+.5f}")
    return auc(Pt[:, masks["FACE(OFA+FFA)"]].mean(1), Pg[:, masks["FACE(OFA+FFA)"]].mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=30)
    ap.add_argument("--n-general", type=int, default=30)
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path, default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    from facenet_pytorch import MTCNN
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=dev)
    masks = build_masks()
    rng = random.Random(args.seed)

    tnames = npz_image_names(load_npz(args.target_preds_npz))
    T = load_set(args.target_zip, tnames, args.n_target, rng, mtcnn)

    gnames = []
    for f in rng.sample(discover_npz(args.general_preds_dir), 400):
        try:
            gnames.extend(npz_image_names(load_npz(f)))
        except Exception:
            pass
    G = load_set(args.general_zip, gnames, args.n_general + args.n_pairs + 3, rng, mtcnn)
    print(f"loaded {len(T)} target, {len(G)} general (all with a detected face)")

    tmp_root = get_tmp_root()
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)

    # ---- condition 1: raw ---------------------------------------------------
    Pt = predict_images(model, [i for _, i, _ in T], tmp_root, "rawt")
    Pg = predict_images(model, [i for _, i, _ in G[: args.n_general]], tmp_root, "rawg")
    auc_raw = report("RAW (as decoded -- reproduces the historical comparison)", Pt, Pg, masks)
    free()

    # ---- condition 2: matched ----------------------------------------------
    Ts, Gs, fr_t, fr_g, keep_t, keep_g = [], [], [], [], [], []
    for k, (_, i, b) in enumerate(T):
        im, fr = standardize(i, b, FRAME)
        if im is not None:
            Ts.append(im); fr_t.append(fr); keep_t.append(k)
    for k, (_, i, b) in enumerate(G):
        im, fr = standardize(i, b, FRAME)
        if im is not None:
            Gs.append(im); fr_g.append(fr); keep_g.append(k)
    print(f"\n  matched: kept {len(Ts)}/{len(T)} target, {len(Gs)}/{len(G)} general")
    print(f"  achieved face fraction: target {np.mean(fr_t):.3f}+/-{np.std(fr_t):.3f}, "
          f"general {np.mean(fr_g):.3f}+/-{np.std(fr_g):.3f}  "
          f"(equal means crop tightness is genuinely equalised)")
    Pt2 = predict_images(model, Ts, tmp_root, "mt")
    Pg2 = predict_images(model, Gs[: args.n_general], tmp_root, "mg")
    auc_match = report("MATCHED (same crop fraction, size, luminance, contrast)",
                       Pt2, Pg2, masks)
    free()

    # ---- condition 3: composite swap pairs ---------------------------------
    tiles_t = [cv2.resize(x, (TILE, TILE)) for x in Ts]
    tiles_g = [cv2.resize(x, (TILE, TILE)) for x in Gs]

    def grid(four):
        top = np.concatenate(four[:2], axis=1)
        bot = np.concatenate(four[2:], axis=1)
        return np.concatenate([top, bot], axis=0)

    A_imgs, B_imgs = [], []
    for k in range(args.n_pairs):
        base = [tiles_g[(k + 1) % len(tiles_g)],
                tiles_g[(k + 2) % len(tiles_g)],
                tiles_g[(k + 3) % len(tiles_g)]]
        swap_in_a = tiles_t[k % len(tiles_t)]
        swap_in_b = tiles_g[(k + 4) % len(tiles_g)]
        pos = k % 4                      # rotate which quadrant is swapped
        a = base[:pos] + [swap_in_a] + base[pos:]
        b = base[:pos] + [swap_in_b] + base[pos:]
        A_imgs.append(grid(a))
        B_imgs.append(grid(b))

    Pa = predict_images(model, A_imgs, tmp_root, "ca")
    Pb = predict_images(model, B_imgs, tmp_root, "cb")
    free()

    m = masks["FACE(OFA+FFA)"]
    print("\n  COMPOSITE sanity check (is a 4-face collage even face-like?)")
    print(f"    single matched face  FACE response = "
          f"{np.concatenate([Pt2[:, m].mean(1), Pg2[:, m].mean(1)]).mean():+.5f}")
    print(f"    4-face composite     FACE response = "
          f"{np.concatenate([Pa[:, m].mean(1), Pb[:, m].mean(1)]).mean():+.5f}")

    print("\n  COMPOSITE swap pairs (paired; only one quadrant differs)")
    n = min(len(Pa), len(Pb))
    for roi in ROIS:
        mm = masks[roi]
        d = Pa[:n, mm].mean(1) - Pb[:n, mm].mean(1)
        t = d.mean() / max(d.std(ddof=1) / np.sqrt(n), 1e-12)
        print(f"    {roi.split('(')[0]:6s} delta={d.mean():+.6f}  t={t:+6.2f}"
              f"{'*' if abs(t) > 2 else ' '}  (n={n})")

    print(f"\n  FACE AUC  raw {auc_raw:.3f}  ->  matched {auc_match:.3f}")
    print("  A collapse toward 0.50 means the historical separation was the "
          "container, not the person.")


if __name__ == "__main__":
    main()

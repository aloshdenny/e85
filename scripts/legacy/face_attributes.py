"""
face_attributes.py

Builds a named, interpretable attribute table for exactly the images whose
vjepa2 activations are already cached, so attributes, activations and predicted
brain maps can all be indexed by the same row.

WHAT THIS IS FOR. The measured problem is that the ablation direction
mean(X_target) - mean(X_general) sits almost entirely inside general face
variance (top-20-PC leak 0.83-1.00 at every depth), so removing it removes
generic face processing. Naming that variance lets us do two things a blind
covariance whitening cannot:

  * say WHICH shared properties the "Mia direction" is actually loading on --
    crop tightness, contrast, skin tone, an eyewear proxy -- which is the
    confound audit this project has needed at every round;
  * project that named subspace OUT of the Mia contrast, leaving the part that
    is Mia rather than "people photographed like Mia".

Note the direction of use. Ablating attribute directions themselves would be
maximally NON-surgical: suppressing "glasses" or "young female" suppresses
everyone who matches. Identity is the residual after the shared attributes are
removed, not the attributes.

Attributes come from MTCNN's box + 5 landmarks and from plain image statistics,
so nothing needs downloading beyond the detector and nothing is a black box:
every column below is something you can point at in the photo. Age/gender/race
are NOT derivable this way -- FairFace's own labels no longer map onto this
dataset (the images were renamed to sequential indices when FairFace and FFHQ
were merged) -- so those would need a separate classifier pass.

A facenet (VGGFace2) embedding is stored alongside. It is not an attribute: it
is an INDEPENDENT face-identity space, used later to check whether whatever we
end up ablating actually aligns with identity rather than with photographic
style.

Usage:
  python scripts/face_attributes.py --general-sample-size 2000 --seed 0
"""

import os, sys, argparse, random, zipfile
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from chunk_utils import (
    discover_npz, load_npz, preds_as_image_vectors, npz_image_names, ensure_fused_zip,
)
from abliteration import decode_image_from_zip, build_face_mask, OUT_DIR

ATTR_NAMES = [
    # framing / geometry -- the "crop tightness" family of confounds
    "face_area_frac", "face_cx", "face_cy", "interocular_frac", "roll_deg",
    "yaw_proxy", "mouth_width_frac", "eye_mouth_frac",
    # global photometrics -- the confound photometric normalisation targets
    "luminance", "contrast", "sharpness", "colorfulness", "saturation",
    # regional appearance
    "skin_L", "skin_a", "skin_b",
    "eye_edge_density",      # glasses proxy: edge energy in the eye band
    "jaw_edge_density",      # jawline definition proxy
    "forehead_luma",
    "hair_darkness",
    "bg_uniformity",         # studio backdrop vs natural scene
]


def _safe(v, default=0.0):
    v = float(v)
    return v if np.isfinite(v) else default


def compute_attributes(img_bgr, box, lm):
    """box = [x0,y0,x1,y1] or None; lm = (5,2) landmarks or None."""
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    a = dict.fromkeys(ATTR_NAMES, 0.0)

    a["luminance"] = _safe(gray.mean())
    a["contrast"] = _safe(gray.std())
    a["sharpness"] = _safe(cv2.Laplacian(gray, cv2.CV_32F).var())
    a["saturation"] = _safe(hsv[..., 1].mean())
    b_, g_, r_ = [img_bgr[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r_ - g_, 0.5 * (r_ + g_) - b_
    a["colorfulness"] = _safe(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                              + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    border = np.concatenate([gray[:4].ravel(), gray[-4:].ravel(),
                             gray[:, :4].ravel(), gray[:, -4:].ravel()])
    a["bg_uniformity"] = _safe(-border.std())   # higher = flatter backdrop

    if box is not None:
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        a["face_area_frac"] = _safe((bw * bh) / (w * h))
        a["face_cx"] = _safe(((x0 + x1) / 2) / w)
        a["face_cy"] = _safe(((y0 + y1) / 2) / h)

    if lm is not None:
        le, re, nose, ml, mr = [np.asarray(p, dtype=np.float32) for p in lm]
        eye_mid = (le + re) / 2
        iod = float(np.linalg.norm(re - le)) or 1.0
        a["interocular_frac"] = _safe(iod / short)
        a["roll_deg"] = _safe(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
        a["yaw_proxy"] = _safe((nose[0] - eye_mid[0]) / iod)
        a["mouth_width_frac"] = _safe(np.linalg.norm(mr - ml) / iod)
        mouth_mid = (ml + mr) / 2
        a["eye_mouth_frac"] = _safe(abs(mouth_mid[1] - eye_mid[1]) / iod)

        def band(y_lo, y_hi, x_lo, x_hi):
            ys = slice(max(0, int(y_lo)), min(h, max(int(y_hi), int(y_lo) + 1)))
            xs = slice(max(0, int(x_lo)), min(w, max(int(x_hi), int(x_lo) + 1)))
            return ys, xs

        # eye band: where spectacle frames live
        ys, xs = band(eye_mid[1] - 0.45 * iod, eye_mid[1] + 0.45 * iod,
                      le[0] - 0.5 * iod, re[0] + 0.5 * iod)
        eye_patch = gray[ys, xs]
        if eye_patch.size:
            edges = cv2.Canny(eye_patch.astype(np.uint8), 60, 160)
            a["eye_edge_density"] = _safe(edges.mean() / 255.0)

        # jaw band: below the mouth
        ys, xs = band(mouth_mid[1] + 0.15 * iod, mouth_mid[1] + 0.95 * iod,
                      ml[0] - 0.6 * iod, mr[0] + 0.6 * iod)
        jaw = gray[ys, xs]
        if jaw.size:
            a["jaw_edge_density"] = _safe(cv2.Canny(jaw.astype(np.uint8), 60, 160).mean() / 255.0)

        # forehead band: above the eyes
        ys, xs = band(eye_mid[1] - 1.15 * iod, eye_mid[1] - 0.55 * iod,
                      le[0] - 0.2 * iod, re[0] + 0.2 * iod)
        fh = gray[ys, xs]
        if fh.size:
            a["forehead_luma"] = _safe(fh.mean())

        # cheeks: skin tone, away from eyes/mouth/hair
        ys, xs = band(eye_mid[1] + 0.35 * iod, eye_mid[1] + 0.85 * iod,
                      le[0] - 0.25 * iod, le[0] + 0.15 * iod)
        cheek = lab[ys, xs]
        if cheek.size:
            a["skin_L"] = _safe(cheek[..., 0].mean())
            a["skin_a"] = _safe(cheek[..., 1].mean())
            a["skin_b"] = _safe(cheek[..., 2].mean())

        # hair: dark fraction in the top corners, outside the face box
        top = gray[: max(1, int(0.28 * h))]
        a["hair_darkness"] = _safe((top < 80).mean())

    return np.array([a[k] for k in ATTR_NAMES], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path, default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--general-sample-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--activation-cache", type=Path,
                    default=OUT_DIR / "raw_activations_norm",
                    help="Checked against, so a row-order mismatch fails here rather than "
                         "silently pairing the wrong attributes with the wrong activations.")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "attributes.npz")
    args = ap.parse_args()

    from facenet_pytorch import MTCNN, InceptionResnetV1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=dev)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(dev)

    mask = build_face_mask(include_secondary=False)

    # ---- replay the EXACT sampling that built the activation cache ----------
    # abliteration.py loads target images in npz order, then general images from
    # random.Random(seed).sample(discover_npz(dir), n). Reproduced here rather
    # than re-derived, because the activation rows are in precisely that order.
    records = []   # (name, img_bgr, preds_vec, is_target)

    tzip = ensure_fused_zip(args.target_zip)
    tdata = load_npz(args.target_preds_npz)
    tpreds = preds_as_image_vectors(tdata["preds"])
    tnames = npz_image_names(tdata)
    with zipfile.ZipFile(tzip, "r") as zf:
        for i, name in enumerate(tnames):
            try:
                records.append((name, decode_image_from_zip(zf, name), tpreds[i], True))
            except Exception as e:
                print(f"  [WARN] target/{name}: {e}")
    n_target = len(records)
    print(f"target images: {n_target}")

    gzip_path = ensure_fused_zip(args.general_zip)
    npz_files = discover_npz(args.general_preds_dir)
    rng = random.Random(args.seed)
    chosen = rng.sample(npz_files, min(args.general_sample_size, len(npz_files)))
    with zipfile.ZipFile(gzip_path, "r") as zf:
        for npz_path in chosen:
            try:
                d = load_npz(npz_path)
                preds = preds_as_image_vectors(d["preds"])
                names = npz_image_names(d)
                for i, name in enumerate(names):
                    records.append((name, decode_image_from_zip(zf, name), preds[i], False))
            except Exception as e:
                print(f"  [WARN] {npz_path.name}: {e}")
    print(f"total rows: {len(records)} ({n_target} target)")

    cache_x = args.activation_cache / "raw_X_L0.npy"
    if cache_x.exists():
        n_cached = np.load(cache_x, mmap_mode="r").shape[0]
        if n_cached != len(records):
            raise SystemExit(
                f"[FATAL] {len(records)} images replayed but the activation cache has "
                f"{n_cached} rows. The sampling did not reproduce -- attributes would be "
                f"paired with the wrong activations. Check --general-sample-size/--seed.")
        print(f"row order verified against {cache_x.name} ({n_cached} rows)")

    # ---- attributes + identity embeddings ----------------------------------
    # Detection is per-image on purpose. MTCNN.detect() accepts a list, but only
    # when every image has identical dimensions -- these zips hold mixed sizes,
    # so the batched call threw for any ragged batch and the except-branch
    # recorded "no face" for all of it. That silently produced 0% detection on
    # the target set, whose images are the most size-varied. Per-image costs a
    # few tens of seconds over the whole set and cannot fail that way.
    attrs, embs, detected, faces_buf = [], [], [], []
    B = 64

    def flush_embeddings():
        if not faces_buf:
            return
        t = torch.from_numpy(np.stack(faces_buf)).permute(0, 3, 1, 2).float().to(dev)
        t = (t - 127.5) / 128.0
        with torch.no_grad():
            embs.append(embedder(t).cpu().numpy())
        faces_buf.clear()

    for idx, (name, img, _, _) in enumerate(records):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        b = l = None
        try:
            bx, pr, lm = mtcnn.detect(rgb, landmarks=True)
            if bx is not None and len(bx):
                b, l = bx[0], (lm[0] if lm is not None and len(lm) else None)
        except Exception as e:
            if idx < 5:
                print(f"  [WARN] detect failed on {name}: {e}")
        detected.append(b is not None)
        attrs.append(compute_attributes(img, b, l))

        if b is not None:
            x0, y0, x1, y1 = [int(max(0, v)) for v in b]
            crop = img[y0:y1, x0:x1]
        else:
            crop = img
        if crop.size == 0:
            crop = img
        faces_buf.append(cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2RGB))

        if len(faces_buf) == B:
            flush_embeddings()
        if idx % 512 == 0:
            print(f"  ...{idx}/{len(records)}")
    flush_embeddings()

    A = np.stack(attrs)
    E = np.concatenate(embs, 0)
    P = np.stack([r[2] for r in records]).astype(np.float32)
    is_t = np.array([r[3] for r in records])
    det = np.array(detected)
    print(f"\nface detected in {det.mean():.1%} of images "
          f"(target {det[is_t].mean():.1%}, general {det[~is_t].mean():.1%})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, attrs=A, attr_names=np.array(ATTR_NAMES), facenet=E, preds=P,
        is_target=is_t, detected=det, names=np.array([r[0] for r in records]),
        mask=mask,
    )
    print(f"Saved -> {args.out}  attrs={A.shape} facenet={E.shape} preds={P.shape}")


if __name__ == "__main__":
    main()

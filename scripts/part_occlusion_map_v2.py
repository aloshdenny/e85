"""
part_occlusion_map_v2.py  (generated from scripts/legacy/part_occlusion_map.py)

Changes vs the original, both forced by what is reachable now:
  * ROIS gains ATL_TP / TP / ATL, so the anterior-temporal stage of the
    pathway is measured alongside OFA and FFA instead of being invisible.
  * Landmarks come from insightface buffalo_l's 1k3d68 (same iBUG-68 layout
    the IDX groups index into) because dlib's shape_predictor .dat lives on
    the 4090, which is offline. part_masks() itself is untouched.

Original header follows.

part_occlusion_map.py

Which cortical ROIs respond to which NAMED part of the face -- eyes, eyebrows,
nose, lips, jawline, forehead, cheeks -- measured by blurring one part at a
time and reading the change in predicted activity.

Two things make this different from occlusion_saliency.py's sliding grid.

  Named regions. dlib's 68 landmarks give real part outlines, so a result reads
  "blurring the eyebrows moved FFA by X" instead of "grid cell r7c6 moved it".
  It is also ~13 forward passes per image rather than ~195.

  An area-matched control, which the grid version could not provide and without
  which none of this is interpretable. A bigger blurred region perturbs the
  image more whatever it covers, so part effects must be compared against
  RANDOM blobs of the same area in the same face. Reported effects are the
  residual after regressing the control effects on area -- how much a part
  matters BEYOND its size.

Occlusion is blur, not flat grey: measured earlier, flat grey raised the face
response almost everywhere (only 9% of patches lowered it) because the
rectangle is itself a large out-of-distribution event. Blur removes local
detail without adding an edge or a colour.

The question this is meant to settle: predicted OFA/FFA correlates +0.54/+0.66
with plain image luminance and <=0.13 with any pose or geometry feature, which
suggests the face pathway here is close to a photometric readout. If lips and
eyebrows move OFA/FFA no more than an area-matched random blob does, that is
confirmed, and there is no part-specific face structure to target.

Usage:
  python scripts/part_occlusion_map.py --n-target 20 --n-general 20
"""

import os, sys, time, argparse, random, zipfile, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "legacy"))
from abliteration import decode_image_from_zip, OUT_DIR, free, TribeModel
from chunk_utils import discover_npz, load_npz, npz_image_names, ensure_fused_zip
from infer_fairface_bulk import (
    get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline,
)
from measure_identity_signal import build_masks
from occlusion_saliency import build_filler

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "ATL_TP", "TP", "ATL",
        "V1", "AUD", "MOTOR", "STS"]

# dlib 68-landmark groups
IDX = {
    "jaw": list(range(0, 17)),
    "eyebrows": list(range(17, 27)),
    "nose": list(range(27, 36)),
    "eyes": list(range(36, 48)),
    "lips": list(range(48, 68)),
}


def part_masks(shape, pts, split_forehead=False):
    """Binary masks per named part. Filled convex hulls for compact parts; the
    jaw is a dilated polyline, since the hull of its landmarks would swallow the
    entire lower face. Forehead and cheeks are derived, having no landmarks."""
    h, w = shape
    face_w = float(np.ptp(pts[:, 0])) or 1.0
    face_h = float(np.ptp(pts[:, 1])) or 1.0
    out = {}

    for name in ["eyes", "eyebrows", "nose", "lips"]:
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(pts[IDX[name]].astype(np.int32)), 255)
        m = cv2.dilate(m, np.ones((max(3, int(0.03 * face_w)),) * 2, np.uint8))
        out[name] = m

    m = np.zeros((h, w), np.uint8)
    cv2.polylines(m, [pts[IDX["jaw"]].astype(np.int32)], False, 255,
                  thickness=max(3, int(0.09 * face_w)))
    out["jawline"] = m

    # forehead: band above the eyebrow line, same width as the brows.
    #
    # split=True cuts it into three stacked sub-bands, which is the whole point
    # of the follow-up: the single band is where the one surviving target-
    # specific effect lives (FFA +0.00206 vs +0.00076, t = +5.87), but in her
    # photos it also contains her fringe and hairline, and hair_darkness already
    # correlates -0.46/-0.55 with OFA/FFA. Lower band is skin for essentially
    # every face, upper band is where a hairline falls. If the effect is in
    # _high it is hair; if it is in _low it is forehead skin.
    #
    # The cut is GEOMETRIC on purpose. Splitting on pixel darkness would define
    # the hair region using the very quantity under suspicion, and any result
    # would be circular.
    brow = pts[IDX["eyebrows"]]
    top = brow[:, 1].min()
    x0, x1 = int(brow[:, 0].min()), int(brow[:, 0].max())

    def band_rect(lo_frac, hi_frac):
        m = np.zeros((h, w), np.uint8)
        yb = int(top - lo_frac * face_h)
        ya = int(top - hi_frac * face_h)
        cv2.rectangle(m, (max(0, x0), max(0, ya)), (min(w - 1, x1), max(0, yb)), 255, -1)
        return m

    if split_forehead:
        out["forehead_low"] = band_rect(0.05, 0.17)
        out["forehead_mid"] = band_rect(0.17, 0.29)
        out["forehead_high"] = band_rect(0.29, 0.42)
    else:
        out["forehead"] = band_rect(0.05, 0.42)

    # cheeks: between the outer eye corners and the jaw, lateral to the nose
    m = np.zeros((h, w), np.uint8)
    for eye_out, jaw_i in [(36, 3), (45, 13)]:
        p_eye, p_jaw, p_nose = pts[eye_out], pts[jaw_i], pts[33]
        poly = np.array([p_eye, p_jaw, p_nose, [p_nose[0], p_eye[1]]], np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(poly), 255)
    out["cheeks"] = m
    return out


def random_control_mask(shape, pts, area, rng):
    """A blob of the given pixel area, placed at random inside the face box."""
    h, w = shape
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    r = max(2, int(np.sqrt(max(area, 1.0) / np.pi)))
    cx = int(rng.uniform(x0 + r * 0.5, max(x0 + r * 0.5 + 1, x1 - r * 0.5)))
    cy = int(rng.uniform(y0 + r * 0.5, max(y0 + r * 0.5 + 1, y1 - r * 0.5)))
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))), r, 255, -1)
    return m


def apply_masked_blur(img, filler, mask, feather_px):
    k = int(max(3, feather_px)) | 1
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), 0)[..., None]
    return (img.astype(np.float32) * (1 - a) + filler.astype(np.float32) * a).astype(img.dtype)


def collect_images(target_zip, target_preds_npz, general_preds_dir, general_zip,
                   n_target, n_general, train_seed, train_size, seed,
                   general_image_zip=None):
    rng = random.Random(seed)
    out = []
    z = ensure_fused_zip(target_zip)
    names = []
    try:
        names = list(npz_image_names(load_npz(target_preds_npz)))
    except Exception:
        pass
    if not names:
        with zipfile.ZipFile(z) as zf:
            names = [
                m for m in zf.namelist()
                if Path(m).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                and not Path(m).name.startswith("._")
                and "__MACOSX" not in m
            ]
    # Prefer original studio stills; fall back to whatever names the npz has.
    studio = [nm for nm in names if "in_the_wild" not in nm]
    pool_t = studio if len(studio) >= n_target else names
    with zipfile.ZipFile(z) as zf:
        for nm in rng.sample(pool_t, min(n_target, len(pool_t))):
            out.append((decode_image_from_zip(zf, nm), True))

    if general_image_zip is not None and Path(general_image_zip).exists():
        with zipfile.ZipFile(ensure_fused_zip(general_image_zip)) as zf:
            gnames = [
                m for m in zf.namelist()
                if m.startswith("occlusion/")
                and Path(m).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            if not gnames:
                gnames = [
                    m for m in zf.namelist()
                    if Path(m).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    and not Path(m).name.startswith("._")
                ]
            for nm in rng.sample(gnames, min(n_general, len(gnames))):
                out.append((decode_image_from_zip(zf, nm), False))
        return out

    allz = discover_npz(general_preds_dir)
    used = set(random.Random(train_seed).sample(allz, min(train_size, len(allz))))
    pool = [f for f in allz if f not in used]
    gz = ensure_fused_zip(general_zip)
    n = 0
    with zipfile.ZipFile(gz) as zf:
        for f in rng.sample(pool, min(n_general * 3, len(pool))):
            if n >= n_general:
                break
            try:
                for nm in npz_image_names(load_npz(f)):
                    if n >= n_general:
                        break
                    out.append((decode_image_from_zip(zf, nm), False))
                    n += 1
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=20)
    ap.add_argument("--n-general", type=int, default=20)
    ap.add_argument("--n-controls", type=int, default=5)
    ap.add_argument("--blur-frac", type=float, default=0.10,
                    help="Gaussian sigma as a fraction of face width.")
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path, default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--predictor", type=Path,
                    default=Path("./models/shape_predictor_68_face_landmarks.dat"))
    ap.add_argument("--split-forehead", action="store_true",
                    help="Cut the forehead band into low/mid/high sub-bands to "
                         "separate skin forehead from hairline.")
    ap.add_argument("--only-parts", nargs="*", default=None,
                    help="Restrict to these parts, so a follow-up costs a few "
                         "passes per image instead of thirteen.")
    ap.add_argument("--write-workers", type=int, default=13,
                    help="Threads encoding the per-variant clips.")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--readout-lowrank", type=Path, default=None,
                    help="If given, patch predictor.weights with this saved "
                         "low-rank residual (nod_finetune_readout.py output) "
                         "before running the occlusion sweep.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write the row array. Default: "
                         "abliterated/part_occlusion.npy, or "
                         "part_occlusion_suppressed.npy if --readout-lowrank.")
    ap.add_argument("--general-image-zip", type=Path, default=None,
                    help="If set, load general-arm images from this zip "
                         "(members under occlusion/) instead of FairFace preds.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=2000)
    args = ap.parse_args()

    from insightface.app import FaceAnalysis
    _app = FaceAnalysis(name="buffalo_l",
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _app.prepare(ctx_id=0, det_size=(640, 640))
    masks_roi = build_masks()

    if args.out is not None:
        out_path = args.out
    elif args.readout_lowrank is not None:
        out_path = OUT_DIR / "part_occlusion_suppressed.npy"
    else:
        out_path = OUT_DIR / "part_occlusion.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imgs = collect_images(args.target_zip, args.target_preds_npz, args.general_preds_dir,
                          args.general_zip, args.n_target, args.n_general,
                          args.train_seed, args.train_size, args.seed,
                          general_image_zip=args.general_image_zip)
    print(f"{len(imgs)} images ({sum(t for _, t in imgs)} target)")

    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)

    if args.readout_lowrank is not None:
        from tribev2.model import FmriEncoderModel
        import gc
        instances = [o for o in gc.get_objects() if isinstance(o, FmriEncoderModel)]
        if not instances:
            raise RuntimeError("no live FmriEncoderModel found to patch")
        fe = instances[-1]
        d = np.load(args.readout_lowrank)
        U = torch.tensor(d["U"], dtype=fe.predictor.weights.dtype, device=fe.predictor.weights.device)
        V = torch.tensor(d["V"], dtype=fe.predictor.weights.dtype, device=fe.predictor.weights.device)
        with torch.no_grad():
            fe.predictor.weights[0] += U @ V
        print(f"  patched predictor.weights[0] in place with rank-{int(d['rank'])} "
              f"residual from {args.readout_lowrank}")
    tmp_root = get_tmp_root()
    rng = np.random.default_rng(args.seed)

    t_start = time.time()
    rows = []   # (is_target, part, area_frac, {roi: delta})
    skipped = 0
    for n, (img, is_t) in enumerate(imgs):
        # FairFace crops are tight enough that the detector finds NOTHING in
        # them (measured: 0/25 raw, 21/25 once padded) -- the face fills the
        # frame and there is no margin to anchor on. So detect on a padded
        # COPY and shift the landmarks back into original coordinates: the
        # image handed to TRIBE below is still the untouched original, which
        # matters because padding would otherwise change the stimulus and make
        # this arm incomparable to the target arms.
        pad_y, pad_x = int(0.35 * img.shape[0]), int(0.35 * img.shape[1])
        padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                    cv2.BORDER_REPLICATE)
        faces = _app.get(padded)
        if not faces:
            skipped += 1
            continue
        # biggest detection, so a background face never wins over the subject
        f0 = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        pts = np.array(f0.landmark_3d_68, dtype=np.float32)[:, :2]
        pts -= np.array([pad_x, pad_y], dtype=np.float32)
        # a face whose landmarks fall outside the real frame is a padding
        # artifact, not a face
        if (pts[:, 0].max() < 0 or pts[:, 1].max() < 0
                or pts[:, 0].min() > img.shape[1] or pts[:, 1].min() > img.shape[0]):
            skipped += 1
            continue

        pm = part_masks(img.shape[:2], pts, split_forehead=args.split_forehead)
        if args.only_parts:
            pm = {k: v for k, v in pm.items() if k in args.only_parts}
        face_w = float(np.ptp(pts[:, 0])) or 1.0
        # Sigma is tied to FACE width, not to the part, and deliberately modest.
        # An earlier draft used sigma ~0.3*face_w, which averaged so widely that
        # every part came out flat grey -- reintroducing precisely the hard-fill
        # artifact that made the first occlusion run uninterpretable. This is big
        # enough to destroy part-level detail (eyes and lips carry detail at a
        # few px) while keeping local colour and luminance.
        filler = build_filler(img, "blur", int(round(args.blur_frac * face_w)), 1.0)
        feather = max(3, int(0.06 * face_w))

        variants = [("baseline", img, 0.0)]
        for name, m in pm.items():
            variants.append((f"part:{name}", apply_masked_blur(img, filler, m, feather),
                             float(m.mean() / 255.0)))
        areas = [float(m.sum() / 255.0) for m in pm.values()]
        for c in range(args.n_controls):
            a = float(rng.uniform(min(areas), max(areas)))
            m = random_control_mask(img.shape[:2], pts, a, rng)
            variants.append((f"ctrl:{c}", apply_masked_blur(img, filler, m, feather),
                             float(m.mean() / 255.0)))

        td = Path(tempfile.mkdtemp(prefix=f"part_{n}_", dir=tmp_root))
        try:
            # Clip writing is the bottleneck once fp16 lands. Measured mid-run:
            # GPU fell to 64% mean util at 10 GB of 24 GB while 24 CPU cores sat
            # 92% idle, because the 13 mp4s per image were encoded one after
            # another between GPU calls. cv2's encoder releases the GIL, so a
            # thread pool actually parallelises here.
            clip_rows = [(td / f"i{n}_{k.replace(':', '_')}.mp4",
                          f"i{n}_{k.replace(':', '_')}") for k, _, _ in variants]
            with ThreadPoolExecutor(max_workers=args.write_workers) as ex:
                list(ex.map(lambda a: write_static_clip(a[0], a[1], duration=1.0, fps=2),
                            [(im, cp) for (_, im, _), (cp, _) in zip(variants, clip_rows)]))
            df = make_multi_row_df(clip_rows, duration=1.0)
            # fp16 autocast: benchmarked at 2.99x with r = 1.0000 against the
            # fp32 reference on the same clips, so it is free throughput. (bf16
            # raises "unsupported ScalarType" -- the caching layer hands results
            # to numpy. Cutting num_frames does nothing; the setting never
            # reaches the extractor.)
            if args.fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    preds, segs = model.predict(events=df)
            else:
                preds, segs = model.predict(events=df)
            grouped = group_preds_by_timeline(preds, segs)
        finally:
            shutil.rmtree(td, ignore_errors=True)

        base_tl = f"i{n}_baseline"
        if base_tl not in grouped:
            skipped += 1
            continue
        base = np.asarray(grouped[base_tl])
        for key, _, afrac in variants[1:]:
            tl = f"i{n}_{key.replace(':', '_')}"
            if tl not in grouped:
                continue
            v = np.asarray(grouped[tl])
            d = {r: float(base[masks_roi[r]].mean() - v[masks_roi[r]].mean()) for r in ROIS}
            rows.append((is_t, key, afrac, d))
        if n % 5 == 0:
            el = time.time() - t_start
            rate = el / max(n + 1, 1)
            print(f"  ...{n + 1}/{len(imgs)}  {rate:.1f}s/img  "
                  f"eta {(len(imgs) - n - 1) * rate / 3600:.1f}h", flush=True)
        # Partial saves: this run is hours long, and losing it to a late failure
        # would be worse than the cost of writing the array out periodically.
        if args.save_every and n and n % args.save_every == 0:
            np.save(out_path.with_name(out_path.stem + "_partial.npy"),
                    np.array(rows, dtype=object), allow_pickle=True)
        free()

    print(f"\nskipped {skipped} images with no face detection\n")

    def area_adjusted(sel):
        """Effect of each part beyond what its AREA alone buys, using the random
        controls from the same image pool to fit delta ~ area per ROI."""
        ctrl = [(a, d) for t, k, a, d in rows if k.startswith("ctrl") and sel(t)]
        out = {}
        for roi in ROIS:
            if len(ctrl) >= 3:
                A = np.array([[a, 1.0] for a, _ in ctrl])
                y = np.array([d[roi] for _, d in ctrl])
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            else:
                coef = np.zeros(2)
            measured = sorted({k.split(":", 1)[1] for _, k, _, _ in rows
                               if k.startswith("part:")})
            for part in measured:
                vals = [(a, d[roi]) for t, k, a, d in rows
                        if k == f"part:{part}" and sel(t)]
                if not vals:
                    continue
                resid = [v - (coef[0] * a + coef[1]) for a, v in vals]
                out.setdefault(part, {})[roi] = (float(np.mean(resid)),
                                                 float(np.std(resid) / max(np.sqrt(len(resid)), 1)),
                                                 float(np.mean([a for a, _ in vals])))
        return out

    for label, sel in [("TARGET (Mia)", lambda t: t), ("GENERAL", lambda t: not t)]:
        res = area_adjusted(sel)
        print("=" * 96)
        print(f"{label} -- effect of blurring each part, AREA-ADJUSTED "
              f"(mean +/- sem; + = blurring LOWERED the response)")
        print("=" * 96)
        print(f"{'part':10s} {'area%':>6s}" + "".join(f"{r.split('(')[0]:>13s}" for r in ROIS))
        print("-" * 96)
        for part, per_roi in res.items():
            a = per_roi[ROIS[0]][2] * 100
            line = f"{part:10s} {a:6.1f}"
            for r in ROIS:
                m, s, _ = per_roi[r]
                star = "*" if abs(m) > 2 * s and s > 0 else " "
                line += f"{m:+11.5f}{star} "
            print(line)
        print()

    np.save(out_path, np.array(rows, dtype=object), allow_pickle=True)
    print(f"Saved -> {out_path}   (* = |mean| > 2 sem)")


if __name__ == "__main__":
    main()

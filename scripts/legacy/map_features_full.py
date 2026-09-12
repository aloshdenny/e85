"""
map_features_full.py

The feature -> cortex map over EVERY image in the general set, streaming.

This is the analysis the full 101,698 predictions genuinely buy something for,
because it needs no new forward passes -- the .npz predictions already exist, so
the only cost is detecting landmarks and reading files. Occlusion is the opposite
case: it re-runs the encoder 13 times per image, which is 14.5 GPU-days at this
scale for a 6.9% change in the t-statistic (see precision_forecast.py), because
the 175 target photos, not the general set, are what bounds that comparison.

Streaming rather than loading: 101,698 x 20,484 float32 predictions is ~8.3 GB.
A vertex-wise Pearson correlation only needs running sums -- sum(a), sum(a^2),
sum(p), sum(p^2), sum(a*p) -- so memory stays flat regardless of how many images
are processed, and the run can be resumed or cut short without losing the map.

Outputs a 21-attribute x 20,484-vertex correlation map, summarised per ROI, plus
the per-image attribute table and facenet embeddings for later use.

Usage:
  python scripts/map_features_full.py --limit 0        # 0 = all
"""

import sys, argparse, zipfile, time
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from chunk_utils import (
    discover_npz, load_npz, preds_as_image_vectors, npz_image_names, ensure_fused_zip,
)
from abliteration import decode_image_from_zip, OUT_DIR
from face_attributes import compute_attributes, ATTR_NAMES
from measure_identity_signal import build_masks

ROIS = ["OFA", "FFA", "FACE(OFA+FFA)", "V1", "AUD", "MOTOR", "STS"]
NV = 20484


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path, default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--limit", type=int, default=0, help="0 = every image")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "feature_cortex_map_full.npz")
    ap.add_argument("--save-every", type=int, default=10000)
    args = ap.parse_args()

    from facenet_pytorch import MTCNN, InceptionResnetV1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=dev)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(dev)
    masks = build_masks()

    files = discover_npz(args.general_preds_dir)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} npz files")

    na = len(ATTR_NAMES)
    # running sums for a vertex-wise correlation, held at fixed size
    n = 0
    s_a = np.zeros(na); s_aa = np.zeros(na)
    s_p = np.zeros(NV); s_pp = np.zeros(NV)
    s_ap = np.zeros((na, NV))

    attrs_all, emb_all, names_all, det_all = [], [], [], []
    faces_buf, buf_meta = [], []
    t0 = time.time()

    def flush():
        if not faces_buf:
            return
        t = torch.from_numpy(np.stack(faces_buf)).permute(0, 3, 1, 2).float().to(dev)
        t = (t - 127.5) / 128.0
        with torch.no_grad():
            emb_all.append(embedder(t).cpu().numpy().astype(np.float32))
        faces_buf.clear()

    zp = ensure_fused_zip(args.general_zip)
    with zipfile.ZipFile(zp, "r") as zf:
        for i, f in enumerate(files):
            try:
                d = load_npz(f)
                preds = preds_as_image_vectors(d["preds"]).astype(np.float64)
                nms = npz_image_names(d)
            except Exception:
                continue
            for j, nm in enumerate(nms):
                try:
                    img = decode_image_from_zip(zf, nm)
                except Exception:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                b = l = None
                try:
                    bx, _, lm = mtcnn.detect(rgb, landmarks=True)
                    if bx is not None and len(bx):
                        b = bx[0]
                        l = lm[0] if lm is not None and len(lm) else None
                except Exception:
                    pass

                a = compute_attributes(img, b, l).astype(np.float64)
                p = preds[j]

                # only detected faces contribute to the map: the geometry
                # columns are undefined without landmarks
                if b is not None:
                    n += 1
                    s_a += a; s_aa += a * a
                    s_p += p; s_pp += p * p
                    s_ap += np.outer(a, p)

                attrs_all.append(a.astype(np.float32))
                names_all.append(nm)
                det_all.append(b is not None)

                if b is not None:
                    x0, y0, x1, y1 = [int(max(0, v)) for v in b]
                    crop = img[y0:y1, x0:x1]
                else:
                    crop = img
                if crop.size == 0:
                    crop = img
                faces_buf.append(cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2RGB))
                if len(faces_buf) == args.batch:
                    flush()

            if i % 2000 == 0 and i:
                el = time.time() - t0
                print(f"  ...{i}/{len(files)} files, {n} detected, "
                      f"{el/i:.3f}s/file, eta {(len(files)-i)*el/i/3600:.1f}h", flush=True)
    flush()

    if n < 10:
        raise SystemExit("too few detections to build a map")

    mu_a, mu_p = s_a / n, s_p / n
    sd_a = np.sqrt(np.maximum(s_aa / n - mu_a ** 2, 0))
    sd_p = np.sqrt(np.maximum(s_pp / n - mu_p ** 2, 0))
    cov = s_ap / n - np.outer(mu_a, mu_p)
    corr = cov / np.maximum(np.outer(sd_a, sd_p), 1e-12)

    print(f"\n{n:,} images with a detected face\n")
    print(f"{'attribute':20s}" + "".join(f"{r.split('(')[0]:>12s}" for r in ROIS))
    print("-" * (20 + 12 * len(ROIS)))
    for k, nm in enumerate(ATTR_NAMES):
        print(f"{nm:20s}" + "".join(f"{corr[k][masks[r]].mean():+12.3f}" for r in ROIS))

    E = np.concatenate(emb_all, 0) if emb_all else np.zeros((0, 512), np.float32)
    np.savez_compressed(
        args.out, corr=corr.astype(np.float32), attr_names=np.array(ATTR_NAMES),
        n=n, attrs=np.stack(attrs_all), detected=np.array(det_all),
        names=np.array(names_all), facenet=E)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

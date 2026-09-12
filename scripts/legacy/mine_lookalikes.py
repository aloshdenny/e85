"""Mine Facenet/InsightFace lookalikes of Mia from FairFace+FFHQ.

Writes target/lookalikes.zip with:
  lookalike/   top-N cosine to the mia.zip gallery centroid
  random/      seed-fixed general faces (collateral control)
  occlusion/   another seed-fixed set for the part-occlusion general arm

Usage:
  python scripts/mine_lookalikes.py --pool 1500 --top 25
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

sys.path.append(str(Path(__file__).parent))
from chunk_utils import ensure_fused_zip, resolve_zip_member


def decode_image_from_zip(zf, name):
    member = resolve_zip_member(zf, name)
    data = zf.read(member)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not decode {name}")
    return img

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def zip_image_names(zf):
    return [
        n for n in zf.namelist()
        if Path(n).suffix.lower() in IMAGE_EXTS
        and not Path(n).name.startswith("._")
        and "__MACOSX" not in n
    ]


def gallery_centroid(app, mia_zip: Path, n_gallery: int, seed: int):
    embs = []
    with zipfile.ZipFile(ensure_fused_zip(mia_zip)) as zf:
        names = zip_image_names(zf)
        rng = np.random.default_rng(seed)
        if len(names) > n_gallery:
            names = list(rng.choice(names, n_gallery, replace=False))
        for n in names:
            try:
                im = decode_image_from_zip(zf, n)
            except Exception:
                continue
            emb, _ = detect_embed(app, im)
            if emb is None:
                continue
            embs.append(emb)
    if len(embs) < 8:
        raise SystemExit(f"gallery too small: {len(embs)}")
    print(f"gallery centroid from {len(embs)} mia.zip faces")
    return unit(np.mean(embs, axis=0))


def detect_embed(app, im):
    faces = app.get(im)
    if not faces:
        return None, None
    faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
    return unit(faces[0].normed_embedding.astype(np.float64)), im


def encode_jpg(bgr, quality=92):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mia-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--general-zip", type=Path,
                    default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--n-random", type=int, default=25)
    ap.add_argument("--n-occlusion", type=int, default=20)
    ap.add_argument("--gallery", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("./target/lookalikes.zip"))
    ap.add_argument("--tsv", type=Path, default=Path("./target_preds/lookalikes.tsv"))
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    centroid = gallery_centroid(app, args.mia_zip, args.gallery, args.seed)
    # FairFace crops are 224px; buffalo_l at det_size=640 misses them.
    app.prepare(ctx_id=-1, det_size=(224, 224))

    gz = ensure_fused_zip(args.general_zip)
    rng = np.random.default_rng(args.seed)
    scored = []
    with zipfile.ZipFile(gz) as zf:
        names = zip_image_names(zf)
        if len(names) > args.pool:
            names = list(rng.choice(names, args.pool, replace=False))
        print(f"scoring {len(names)} general faces...")
        n_miss = 0
        for i, n in enumerate(names):
            try:
                im = decode_image_from_zip(zf, n)
            except Exception:
                n_miss += 1
                continue
            emb, _ = detect_embed(app, im)
            if emb is None:
                n_miss += 1
                continue
            sim = float(emb @ centroid)
            scored.append((sim, n, im))
            if (i + 1) % 100 == 0:
                print(f"  ...{i + 1}/{len(names)}  kept {len(scored)}  miss {n_miss}",
                      flush=True)

    if len(scored) < args.top + args.n_random + args.n_occlusion:
        raise SystemExit(f"only {len(scored)} detections, need more pool")

    scored.sort(key=lambda t: t[0], reverse=True)
    print(f"\ntop lookalike sim={scored[0][0]:+.3f}  "
          f"median={scored[len(scored)//2][0]:+.3f}  "
          f"min={scored[-1][0]:+.3f}")

    top = scored[: args.top]
    used = {n for _, n, _ in top}
    rest = [(s, n, im) for s, n, im in scored if n not in used]
    rng2 = np.random.default_rng(args.seed + 1)
    rest_idx = rng2.permutation(len(rest))
    rand = [rest[i] for i in rest_idx[: args.n_random]]
    used |= {n for _, n, _ in rand}
    rest2 = [(s, n, im) for s, n, im in rest if n not in used]
    occ = [rest2[i] for i in rng2.permutation(len(rest2))[: args.n_occlusion]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf, \
            args.tsv.open("w") as tsv:
        tsv.write("split\trank\tsim\tsource\n")
        for i, (sim, n, im) in enumerate(top):
            dest = f"lookalike/top_{i:02d}.jpg"
            zf.writestr(dest, encode_jpg(im))
            tsv.write(f"lookalike\t{i}\t{sim:.6f}\t{n}\n")
        for i, (sim, n, im) in enumerate(rand):
            dest = f"random/rnd_{i:02d}.jpg"
            zf.writestr(dest, encode_jpg(im))
            tsv.write(f"random\t{i}\t{sim:.6f}\t{n}\n")
        for i, (sim, n, im) in enumerate(occ):
            dest = f"occlusion/g_{i:02d}.jpg"
            zf.writestr(dest, encode_jpg(im))
            tsv.write(f"occlusion\t{i}\t{sim:.6f}\t{n}\n")

    print(f"wrote {args.out}  ({args.top} lookalikes, {args.n_random} random, "
          f"{args.n_occlusion} occlusion general)")
    print(f"wrote {args.tsv}")


if __name__ == "__main__":
    main()

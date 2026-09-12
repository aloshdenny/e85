"""
token_pooling_probe.py

The one untested lever: every direction this project has searched came from
collect_layer_activations' `hidden.mean(dim=1)` -- the mean over ALL ~2048
tokens (8 temporal x 16 x 16 spatial at CLIP_FRAMES=16, patch 16, tubelet 2).
That average pools face tokens together with background, hair and border, so if
identity lives in a subset of positions it is diluted before any direction
search ever sees it.

This extracts three poolings from the SAME forward pass, at several layers:

  mean_all   what has always been used, reproduced for comparison
  face       only tokens whose 16x16 spatial cell overlaps the detected face box
  nonface    only tokens outside it -- the control that decides the whole thing

nonface is the part that makes this interpretable. If identity decodes just as
well from background tokens as from face tokens, then whatever is decodable is
a property of the photograph, not the face, and no amount of re-pooling will
help. That is the same question the provenance decoy asked of the predictions,
asked now of the encoder.

Scored by ridge to facenet's VGGFace2 embedding, cross-validated on general
images, reported as held-out cosine and top-1 retrieval within the fold. The
reference to beat is mean-pooling's 28.1% retrieval at L35 (chance 0.26%).

Usage:
  python scripts/token_pooling_probe.py --layers 15 25 30 35 39
"""

import os, sys, time, argparse, random, zipfile
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
import abliteration as abl
from abliteration import (
    build_face_mask, decode_image_from_zip, image_to_vjepa_input, OUT_DIR, free, TribeModel,
)
from chunk_utils import discover_npz, load_npz, npz_image_names, ensure_fused_zip
from probe_identity_content import ridge_cv

GRID = 16          # 256px / 16px patches
CELLS = GRID * GRID


def face_cell_mask(box, img_shape):
    """Which of the 16x16 spatial cells overlap the face box, after the same
    plain resize to 256x256 that image_to_vjepa_input performs (no aspect
    preservation, so the mapping is a straight linear rescale per axis)."""
    m = np.zeros((GRID, GRID), bool)
    if box is None:
        m[:] = True
        return m
    h, w = img_shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    c0 = int(np.floor(max(x0, 0) / w * GRID))
    c1 = int(np.ceil(min(x1, w) / w * GRID))
    r0 = int(np.floor(max(y0, 0) / h * GRID))
    r1 = int(np.ceil(min(y1, h) / h * GRID))
    m[max(r0, 0):max(r1, r0 + 1), max(c0, 0):max(c1, c0 + 1)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 25, 30, 35, 39])
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path, default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--general-sample-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--attrs", type=Path, default=OUT_DIR / "attributes.npz")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "token_pooled.npz")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from facenet_pytorch import MTCNN
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=dev)

    # Same image order as the activation cache and attributes.npz: target npz
    # order first, then random.Random(seed).sample over the general npz files.
    mask = build_face_mask(include_secondary=False)
    records = []
    with zipfile.ZipFile(ensure_fused_zip(args.target_zip)) as zf:
        for nm in npz_image_names(load_npz(args.target_preds_npz)):
            try:
                records.append((decode_image_from_zip(zf, nm), True))
            except Exception:
                pass
    n_target = len(records)
    rng = random.Random(args.seed)
    files = discover_npz(args.general_preds_dir)
    chosen = rng.sample(files, min(args.general_sample_size, len(files)))
    with zipfile.ZipFile(ensure_fused_zip(args.general_zip)) as zf:
        for f in chosen:
            try:
                for nm in npz_image_names(load_npz(f)):
                    records.append((decode_image_from_zip(zf, nm), False))
            except Exception:
                pass
    print(f"{len(records)} images ({n_target} target)")

    d = np.load(args.attrs, allow_pickle=True)
    F = d["facenet"].astype(np.float64)
    if len(F) != len(records):
        raise SystemExit(f"attributes.npz has {len(F)} rows, replayed {len(records)}")

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    vj = model.data.video_feature.image.model.model
    blocks = vj.encoder.layer
    vj.eval().to(abl.DEVICE)

    buf = {}

    def mk(idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            buf[idx] = h.detach()[0]          # (tokens, hidden)
        return hook

    handles = [blocks[l].register_forward_hook(mk(l)) for l in args.layers]

    pools = ["mean_all", "face", "nonface"]
    store = {(l, p): [] for l in args.layers for p in pools}
    frac_face, detected = [], []
    t0 = time.time()

    try:
        for i, (img, _) in enumerate(records):
            bx, _ = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            box = bx[0] if bx is not None and len(bx) else None
            detected.append(box is not None)
            cm = face_cell_mask(box, img.shape)
            frac_face.append(float(cm.mean()))
            flat = cm.reshape(-1)                      # (256,) spatial cells

            clip = image_to_vjepa_input(img).unsqueeze(0).to(abl.DEVICE)
            buf.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                vj(pixel_values_videos=clip)

            for l in args.layers:
                H = buf[l].float()                     # (T*256, hidden)
                n_tok = H.shape[0]
                # token index = t*256 + h*16 + w, so the spatial cell repeats
                # every 256 tokens across the 8 temporal positions
                cell = torch.arange(n_tok, device=H.device) % CELLS
                sel = torch.from_numpy(flat).to(H.device)[cell]
                store[(l, "mean_all")].append(H.mean(0).cpu().numpy())
                store[(l, "face")].append(
                    (H[sel].mean(0) if sel.any() else H.mean(0)).cpu().numpy())
                store[(l, "nonface")].append(
                    (H[~sel].mean(0) if (~sel).any() else H.mean(0)).cpu().numpy())
            del clip
            if i % 250 == 0:
                el = time.time() - t0
                print(f"  ...{i}/{len(records)}  {el/max(i,1):.3f}s/img  "
                      f"eta {(len(records)-i)*el/max(i,1)/60:.1f}m", flush=True)
                free()
    finally:
        for h in handles:
            h.remove()

    det = np.array(detected)
    print(f"\nface detected {det.mean():.1%}; face cells cover "
          f"{np.mean(frac_face):.1%} of the grid on average\n")

    arrs = {f"{p}_L{l}": np.stack(store[(l, p)]).astype(np.float32)
            for l in args.layers for p in pools}
    np.savez_compressed(args.out, n_target=n_target, detected=det,
                        frac_face=np.array(frac_face), **arrs)

    gen = np.zeros(len(records), bool)
    gen[n_target:] = True
    keep = gen & det
    Fg = F[keep]

    print(f"identity decoding from {keep.sum()} general images "
          f"(chance retrieval {args.folds/keep.sum():.2%})")
    print(f"{'layer':>5s} {'pooling':>10s} {'cos':>8s} {'retrieval':>10s}")
    print("-" * 38)
    for l in args.layers:
        for p in pools:
            X = arrs[f"{p}_L{l}"][keep].astype(np.float64)
            c, r = ridge_cv(X, Fg, args.folds)
            print(f"{l:5d} {p:>10s} {c:8.3f} {r:10.1%}")
    print("\nface >> nonface means the face tokens carry identity the average "
          "was diluting.\nface ~= nonface means it is a property of the photo, "
          "and re-pooling cannot help.")


if __name__ == "__main__":
    main()

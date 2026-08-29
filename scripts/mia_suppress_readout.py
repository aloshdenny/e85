"""
mia_suppress_readout.py

Fine-tune a low-rank residual on TRIBE's frozen readout to suppress face-cortex
response to Mia, without touching the 1B-parameter encoder.

Training set: the 175 original studio Mia crops in target/mia.zip (NOT the 26
in-the-wild podcast/interview crops).

Holdout: those 26 in-the-wild crops -- the test of whether the residual learned
*her* rather than studio photography.

Anchor set: a random sample of general face images (FairFace+FFHQ zip, or NOD
COCO fallback) -- predictions on these must stay at the frozen baseline so we
do not globally damp face cortex.

Objective (different from nod_finetune_readout.py, which fit real fMRI):
  pull FACE(OFA+FFA) on Mia toward the general-population baseline mean,
  while anchoring V1/AUD/MOTOR (and general faces) to the unmodified readout.

Same machinery as NOD: bottleneck X is captured by identity-patching
predictor.forward, residual is U @ V added to frozen W0.

Usage:
  python scripts/mia_suppress_readout.py --rank 16 --epochs 400
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))
from abliteration import TribeModel, free
from chunk_utils import ensure_fused_zip, resolve_zip_member
from infer_fairface_bulk import (
    get_tmp_root, group_preds_by_timeline, make_multi_row_df, write_static_clip,
)
from infer_target_face import decode_image_from_zip
from measure_identity_signal import build_masks
from tribev2.model import FmriEncoderModel


def find_fmri_encoder(cache_folder: Path):
    instances = []
    orig_init = FmriEncoderModel.__init__

    def patched(self, *a, **kw):
        orig_init(self, *a, **kw)
        instances.append(self)

    FmriEncoderModel.__init__ = patched
    try:
        m = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)
    finally:
        FmriEncoderModel.__init__ = orig_init
    if not instances:
        raise RuntimeError("FmriEncoderModel not instantiated during load")
    return m, instances[0]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR"]
NOD_COCO = Path.home() / "nod" / "stimuli" / "coco"


def load_mia_from_zip(zip_path: Path):
    """Return (studio, wild) as lists of (name, bgr)."""
    studio, wild = [], []
    with zipfile.ZipFile(ensure_fused_zip(zip_path)) as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                continue
            if Path(name).name.startswith("._") or name.startswith("__MACOSX/"):
                continue
            try:
                img = decode_image_from_zip(zf, name)
            except Exception:
                continue
            if "in_the_wild" in name:
                wild.append((name, img))
            else:
                studio.append((name, img))
    return studio, wild


def load_general_images(general_zip: Path | None, n: int, seed: int):
    """Sample n general face images from FairFace zip or NOD COCO fallback."""
    rng = np.random.default_rng(seed)
    out = []

    if general_zip and general_zip.exists():
        with zipfile.ZipFile(ensure_fused_zip(general_zip)) as zf:
            names = [
                m for m in zf.namelist()
                if Path(m).suffix.lower() in IMAGE_EXTS
                and not Path(m).name.startswith("._")
                and "__MACOSX" not in m
            ]
            if len(names) > n:
                names = list(rng.choice(names, n, replace=False))
            for name in names:
                try:
                    out.append((f"general/{Path(name).name}", decode_image_from_zip(zf, name)))
                except Exception:
                    pass
        if out:
            print(f"  general: {len(out)} from {general_zip.name}")
            return out

    if NOD_COCO.exists():
        paths = sorted(NOD_COCO.glob("*.jpg"))
        if len(paths) > n:
            paths = list(rng.choice(paths, n, replace=False))
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                out.append((f"nod/{p.name}", img))
        print(f"  general: {len(out)} from NOD COCO fallback")
    return out


def capture_bottlenecks(model, fe, items, batch_size, tmp_root):
    """items: [(name, bgr), ...] -> X (N,2048), names."""
    predictor = fe.predictor
    real_forward = predictor.forward

    def identity_forward(x, subject_id=None):
        return x

    names_out = []
    chunks = []
    td = Path(tempfile.mkdtemp(prefix="mia_sup_", dir=tmp_root))
    try:
        predictor.forward = identity_forward
        for s in range(0, len(items), batch_size):
            batch = items[s:s + batch_size]
            rows = []
            for i, (name, img) in enumerate(batch):
                tl = f"b{s + i}"
                clip = td / f"{tl}.mp4"
                write_static_clip(img, clip, duration=1.0, fps=2)
                rows.append((clip, tl, name))
            df = make_multi_row_df([(c, tl) for c, tl, _ in rows], duration=1.0)
            with torch.autocast("cuda", dtype=torch.float16):
                preds, segs = model.predict(events=df)
            grouped = group_preds_by_timeline(preds, segs)
            for _, tl, name in rows:
                if tl in grouped:
                    names_out.append(name)
                    chunks.append(np.asarray(grouped[tl], dtype=np.float32))
            for c, _, _ in rows:
                c.unlink(missing_ok=True)
            print(f"  bottleneck ...{min(s + batch_size, len(items))}/{len(items)}", flush=True)
    finally:
        predictor.forward = real_forward
        shutil.rmtree(td, ignore_errors=True)

    if not chunks:
        raise RuntimeError("no bottlenecks captured")
    return np.stack(chunks), names_out


def roi_means(preds, masks, rois=ROIS):
    return {r: preds[:, masks[r]].mean(1) for r in rois}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mia-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--general-zip", type=Path,
                    default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--n-general", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lam-suppress", type=float, default=5.0,
                    help="Weight on pulling Mia FACE toward general baseline.")
    ap.add_argument("--lam-anchor", type=float, default=3.0,
                    help="Weight on keeping non-face / general predictions at baseline.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-folder", type=Path,
                    default=Path("/home/research/.cache/huggingface"))
    ap.add_argument("--out", type=Path, default=Path("./abliterated/mia_suppress_readout.npz"))
    args = ap.parse_args()

    studio, wild = load_mia_from_zip(args.mia_zip)
    print(f"Mia: {len(studio)} studio (train), {len(wild)} in-the-wild (holdout)")
    if len(studio) < 10 or len(wild) < 5:
        raise SystemExit("need studio + wild splits in mia.zip")

    general = load_general_images(args.general_zip, args.n_general, args.seed)
    if len(general) < 20:
        raise SystemExit("need >=20 general anchor images")

    print("Loading TribeModel...")
    model, fe = find_fmri_encoder(args.cache_folder)
    device = fe.device
    W0 = fe.predictor.weights[0].detach().clone()
    b0 = fe.predictor.bias[0].detach().clone()
    W0_np, b0_np = W0.cpu().numpy(), b0.cpu().numpy()
    print(f"frozen readout W {tuple(W0.shape)}")

    tmp_root = get_tmp_root()
    all_items = studio + general
    print(f"Capturing bottlenecks for {len(all_items)} train images...")
    X_all, names_all = capture_bottlenecks(model, fe, all_items, args.batch, tmp_root)

    studio_names = {n for n, _ in studio}
    gen_names = {n for n, _ in general}
    is_studio = np.array([n in studio_names for n in names_all])
    is_gen = np.array([n in gen_names for n in names_all])
    X_st = X_all[is_studio]
    X_gen = X_all[is_gen]
    print(f"aligned: {X_st.shape[0]} studio, {X_gen.shape[0]} general")

    # Holdout bottlenecks (wild only -- never seen in training)
    print(f"Capturing bottlenecks for {len(wild)} holdout images...")
    X_ho, ho_names = capture_bottlenecks(model, fe, wild, args.batch, tmp_root)
    free()

    masks = build_masks()
    face_m = masks["FACE(OFA+FFA)"]
    anchor_w = np.ones(20484, dtype=np.float32)
    anchor_w[face_m] = 0.0   # face on Mia is free to move; elsewhere pinned

    base_st = X_st @ W0_np + b0_np
    base_gen = X_gen @ W0_np + b0_np
    base_ho = X_ho @ W0_np + b0_np

    target_face = float(base_gen[:, face_m].mean())
    print(f"general baseline FACE mean (anchor target): {target_face:+.5f}")
    print(f"studio baseline FACE mean: {base_st[:, face_m].mean():+.5f}")
    print(f"holdout  baseline FACE mean: {base_ho[:, face_m].mean():+.5f}")

    W0_t = W0.to(device)
    b0_t = b0.to(device)
    anchor_w_t = torch.tensor(anchor_w, device=device)

    def fit_residual(Xs, Xg, rank, epochs, lr, wd, lam_sup, lam_anchor, seed):
        torch.manual_seed(seed)
        U = nn.Parameter(torch.randn(2048, rank, device=device) * 0.01)
        V = nn.Parameter(torch.zeros(rank, 20484, device=device))
        opt = torch.optim.AdamW([U, V], lr=lr, weight_decay=wd)

        Xs_t = torch.tensor(Xs, device=device, dtype=torch.float32)
        Xg_t = torch.tensor(Xg, device=device, dtype=torch.float32)
        base_s_t = Xs_t @ W0_t + b0_t
        base_g_t = Xg_t @ W0_t + b0_t
        target_t = torch.tensor(target_face, device=device)

        best_loss, best_UV = 1e18, None
        for ep in range(epochs):
            opt.zero_grad()
            pred_s = Xs_t @ (W0_t + U @ V) + b0_t
            pred_g = Xg_t @ (W0_t + U @ V) + b0_t

            face_s = pred_s[:, face_m].mean(1)
            suppress = ((face_s - target_t) ** 2).mean()

            anchor_s = (((pred_s - base_s_t) ** 2) * anchor_w_t).mean()
            anchor_g = ((pred_g - base_g_t) ** 2).mean()

            loss = lam_sup * suppress + lam_anchor * (anchor_s + anchor_g)
            loss.backward()
            opt.step()

            lv = float(loss.item())
            if lv < best_loss:
                best_loss, best_UV = lv, (U.detach().clone(), V.detach().clone())
            if ep % 50 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    fs = pred_s[:, face_m].mean().item()
                print(f"  ep {ep:3d}  loss={lv:.5f}  studio FACE={fs:+.5f}", flush=True)
        return best_UV

    print(f"\nTraining rank-{args.rank} residual "
          f"(lam_sup={args.lam_suppress}, lam_anchor={args.lam_anchor})...")
    U, V = fit_residual(X_st, X_gen, args.rank, args.epochs, args.lr, args.wd,
                        args.lam_suppress, args.lam_anchor, args.seed)

    def apply(X):
        with torch.no_grad():
            Xt = torch.tensor(X, device=device, dtype=torch.float32)
            return (Xt @ (W0_t + U @ V) + b0_t).cpu().numpy()

    def report(label, base, ft):
        print(f"\n{label}")
        print(f"{'ROI':>14s} {'baseline':>10s} {'finetuned':>10s} {'delta':>10s}")
        print("-" * 48)
        for r in ROIS:
            bm, fm = base[:, masks[r]].mean(), ft[:, masks[r]].mean()
            print(f"{r:>14s} {bm:+10.5f} {fm:+10.5f} {fm - bm:+10.5f}")

    ft_st = apply(X_st)
    ft_ho = apply(X_ho)
    ft_gen = apply(X_gen)

    report("STUDIO (trained on)", base_st, ft_st)
    report("IN-THE-WILD HOLDOUT (never trained)", base_ho, ft_ho)
    report(f"GENERAL ANCHOR (n={X_gen.shape[0]})", base_gen, ft_gen)

    ho_drop = (base_ho[:, face_m].mean() - ft_ho[:, face_m].mean())
    ho_frac = ho_drop / (base_ho[:, face_m].mean() - target_face + 1e-9)
    print(f"\nHoldout FACE moved {ho_drop:+.5f} toward general "
          f"({100 * ho_frac:.0f}% of gap closed)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, U=U.cpu().numpy(), V=V.cpu().numpy(), rank=args.rank,
             target_face=target_face,
             studio_names=np.array([n for n, _ in studio]),
             holdout_names=np.array(ho_names))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

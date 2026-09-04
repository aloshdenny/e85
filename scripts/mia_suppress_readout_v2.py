"""
mia_suppress_readout_v2.py

v1 closed Mia's FACE gap but also damped random FairFace faces (anchor was
NOD COCO scenes). v2:

  * FairFace+FFHQ portraits as the general set (real faces)
  * hard FACE anchor on those portraits so the residual cannot globally
    turn down OFA/FFA
  * "lookalike" = cosine in the 2048-d bottleneck to the studio-Mia centroid,
    not InsightFace. Top neighbors are HELD OUT of training.

Train: 175 studio Mia (suppress FACE -> FairFace mean) + random FairFace
       (predictions pinned to frozen baseline, including FACE).
Holdout: 26 in-the-wild Mia + top-K TRIBE-neighbors + a random FairFace slice.

Writes a NEW file (mia_suppress_readout_v2.npz). v1 is untouched.

Usage:
  python scripts/mia_suppress_readout_v2.py --n-general 500 --rank 16
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))
from abliteration import free
from infer_fairface_bulk import get_tmp_root
from measure_identity_signal import build_masks
from mia_suppress_readout import (
    IMAGE_EXTS, ROIS, capture_bottlenecks, find_fmri_encoder,
    load_general_images, load_mia_from_zip,
)
from chunk_utils import ensure_fused_zip
from infer_target_face import decode_image_from_zip


def load_all_from_zip(zip_path: Path):
    items = []
    with zipfile.ZipFile(ensure_fused_zip(zip_path)) as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                continue
            if Path(name).name.startswith("._") or name.startswith("__MACOSX/"):
                continue
            try:
                items.append((name, decode_image_from_zip(zf, name)))
            except Exception:
                continue
    return items


def unit_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def report(label, base, ft, masks):
    print(f"\n{label}  n={base.shape[0]}")
    print(f"{'ROI':>14s} {'baseline':>10s} {'finetuned':>10s} {'delta':>10s}")
    print("-" * 48)
    for r in ROIS:
        bm, fm = base[:, masks[r]].mean(), ft[:, masks[r]].mean()
        print(f"{r:>14s} {bm:+10.5f} {fm:+10.5f} {fm - bm:+10.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mia-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--target-zip", type=Path, default=None,
                    help="If set, suppress this person instead of Mia "
                         "(all zip members; holdout via --holdout-frac).")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--split-mode", choices=["random", "in_the_wild", "face_top"],
                    default="random",
                    help="Holdout split: random frac, in_the_wild paths, or top-FACE images.")
    ap.add_argument("--person", type=str, default=None)
    ap.add_argument("--general-zip", type=Path,
                    default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--n-general", type=int, default=500)
    ap.add_argument("--n-neighbor-holdout", type=int, default=25)
    ap.add_argument("--n-random-holdout", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lam-suppress", type=float, default=5.0)
    ap.add_argument("--lam-anchor", type=float, default=3.0,
                    help="Full-vector / non-face anchor.")
    ap.add_argument("--lam-anchor-face", type=float, default=20.0,
                    help="Hard pin on general-face FACE(OFA+FFA).")
    ap.add_argument("--min-drop", type=float, default=0.015,
                    help="If the target is NOT above FairFace, still pull their "
                         "FACE down by this amount so the recipe is well-posed "
                         "for any person, not only people who already spike.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-folder", type=Path,
                    default=Path("/home/research/.cache/huggingface"))
    ap.add_argument("--bottleneck-cache", type=Path,
                    default=Path("/home/research/e85_scratch/v2_bottlenecks.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("./abliterated/mia_suppress_readout_v2.npz"))
    ap.add_argument("--save-preds", type=Path, default=None,
                    help="If set, write frozen-readout target_preds npz (preds+filenames).")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    person = args.person or "Mia"
    zip_path = args.target_zip or args.mia_zip
    face_top_items = None

    if args.split_mode == "in_the_wild":
        studio, wild = load_mia_from_zip(zip_path)
        print(f"{person}: {len(studio)} train, {len(wild)} in-the-wild holdout "
              f"(split=in_the_wild) from {zip_path}")
    elif args.split_mode == "face_top":
        face_top_items = load_all_from_zip(zip_path)
        studio, wild = [], []
        print(f"{person}: {len(face_top_items)} total; face_top split pending "
              f"from {zip_path}")
    elif args.target_zip is not None:
        items = load_all_from_zip(args.target_zip)
        rng.shuffle(items)
        n_ho = max(5, int(round(len(items) * args.holdout_frac)))
        if n_ho >= len(items) - 8:
            n_ho = max(5, len(items) // 5)
        wild, studio = items[:n_ho], items[n_ho:]
        print(f"{person}: {len(studio)} train, {len(wild)} holdout "
              f"(split=random) from {args.target_zip}")
    else:
        studio, wild = load_mia_from_zip(args.mia_zip)
        print(f"{person}: {len(studio)} studio (train), {len(wild)} in-the-wild (holdout)")

    reuse_gen = args.bottleneck_cache.exists()
    general = []
    if not reuse_gen:
        general = load_general_images(args.general_zip, args.n_general, args.seed)
        print(f"general portraits loaded: {len(general)}")
    if face_top_items is None and (len(studio) < 8 or len(wild) < 5):
        raise SystemExit("need enough train + holdout images of the target")

    cache = args.bottleneck_cache
    recapture_target = (
        args.target_zip is not None
        or args.split_mode in ("in_the_wild", "face_top")
    )

    def split_face_top(X_all, all_names, W0_np, b0_np):
        masks_e = build_masks()
        face_me = masks_e["FACE(OFA+FFA)"]
        scores = (X_all @ W0_np + b0_np)[:, face_me].mean(axis=1)
        n_ho = max(5, int(round(len(all_names) * args.holdout_frac)))
        if n_ho >= len(all_names) - 8:
            n_ho = max(5, len(all_names) // 5)
        ho_idx = np.argsort(-scores)[:n_ho]
        st_idx = np.array([i for i in range(len(all_names)) if i not in set(ho_idx)])
        X_st_o = X_all[st_idx]
        X_ho_o = X_all[ho_idx]
        st_n = [all_names[i] for i in st_idx]
        ho_n = [all_names[i] for i in ho_idx]
        print(f"face_top split: {len(st_idx)} train FACE={scores[st_idx].mean():+.5f}  "
              f"{len(ho_idx)} holdout FACE={scores[ho_idx].mean():+.5f}  "
              f"(holdout range {scores[ho_idx].min():+.5f}..{scores[ho_idx].max():+.5f})")
        return X_st_o, st_n, X_ho_o, ho_n

    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        X_gen, gen_names = d["X_gen"], [str(x) for x in d["gen_names"]]
        W0_np, b0_np = d["W0"], d["b0"]
        print(f"reusing FairFace cache {cache}  gen {X_gen.shape}")
        if recapture_target:
            print("Loading TribeModel for target bottlenecks...")
            model, fe = find_fmri_encoder(args.cache_folder)
            device = fe.device
            W0_t = fe.predictor.weights[0].detach().clone()
            b0_t = fe.predictor.bias[0].detach().clone()
            W0_np, b0_np = W0_t.cpu().numpy(), b0_t.cpu().numpy()
            tmp_root = get_tmp_root()
            if face_top_items is not None:
                print(f"Capturing bottlenecks: {len(face_top_items)} all (face_top)...")
                X_all, all_names = capture_bottlenecks(
                    model, fe, face_top_items, args.batch, tmp_root)
                X_st, st_names, X_ho, ho_names = split_face_top(
                    X_all, all_names, W0_np, b0_np)
            else:
                print(f"Capturing bottlenecks: {len(studio)} train...")
                X_st, st_names = capture_bottlenecks(model, fe, studio, args.batch, tmp_root)
                print(f"Capturing bottlenecks: {len(wild)} holdout...")
                X_ho, ho_names = capture_bottlenecks(model, fe, wild, args.batch, tmp_root)
            free()
        else:
            X_st, st_names = d["X_st"], [str(x) for x in d["st_names"]]
            X_ho, ho_names = d["X_ho"], [str(x) for x in d["ho_names"]]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            W0_t = torch.tensor(W0_np, device=device)
            b0_t = torch.tensor(b0_np, device=device)
            print(f"  also reused {person} bottlenecks  studio {X_st.shape} hold {X_ho.shape}")
    else:
        if not general:
            general = load_general_images(args.general_zip, args.n_general, args.seed)
        if len(general) < 80:
            raise SystemExit("need >=80 general faces (or a FairFace bottleneck cache)")
        print("Loading TribeModel...")
        model, fe = find_fmri_encoder(args.cache_folder)
        device = fe.device
        W0 = fe.predictor.weights[0].detach().clone()
        b0 = fe.predictor.bias[0].detach().clone()
        W0_np, b0_np = W0.cpu().numpy(), b0.cpu().numpy()
        W0_t, b0_t = W0.to(device), b0.to(device)
        print(f"frozen readout W {tuple(W0.shape)}")
        tmp_root = get_tmp_root()
        if face_top_items is not None:
            print(f"Capturing bottlenecks: {len(face_top_items)} all (face_top)...")
            X_all, all_names = capture_bottlenecks(
                model, fe, face_top_items, args.batch, tmp_root)
            X_st, st_names, X_ho, ho_names = split_face_top(
                X_all, all_names, W0_np, b0_np)
        else:
            print(f"Capturing bottlenecks: {len(studio)} train...")
            X_st, st_names = capture_bottlenecks(model, fe, studio, args.batch, tmp_root)
            print(f"Capturing bottlenecks: {len(wild)} holdout...")
            X_ho, ho_names = capture_bottlenecks(model, fe, wild, args.batch, tmp_root)
        print(f"Capturing bottlenecks: {len(general)} general...")
        X_gen, gen_names = capture_bottlenecks(model, fe, general, args.batch, tmp_root)
        free()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X_st=X_st, st_names=np.array(st_names),
                 X_ho=X_ho, ho_names=np.array(ho_names),
                 X_gen=X_gen, gen_names=np.array(gen_names),
                 W0=W0_np, b0=b0_np)
        print(f"cached bottlenecks -> {cache}")

    if len(X_st) < 8 or len(X_ho) < 5:
        raise SystemExit(f"need enough train + holdout after split ({len(X_st)}/{len(X_ho)})")

    masks = build_masks()
    face_m = masks["FACE(OFA+FFA)"]
    nonface_w = np.ones(20484, dtype=np.float32)
    nonface_w[face_m] = 0.0

    # TRIBE-space neighbors of studio Mia
    c = unit_rows(X_st).mean(0)
    c = c / max(float(np.linalg.norm(c)), 1e-9)
    sim = unit_rows(X_gen) @ c
    order = np.argsort(-sim)
    print(f"\nTRIBE bottleneck similarity to {person} centroid (general faces)")
    print(f"  max={sim[order[0]]:+.3f}  p90={np.quantile(sim, 0.9):+.3f}  "
          f"median={np.median(sim):+.3f}  min={sim.min():+.3f}")

    n_nb = min(args.n_neighbor_holdout, len(order) // 4)
    n_rh = min(args.n_random_holdout, len(order) - n_nb - 40)
    nb_idx = order[:n_nb]
    rest = order[n_nb:]
    rh_idx = rng.choice(rest, n_rh, replace=False)
    train_pool = np.array([i for i in rest if i not in set(rh_idx)])
    print(f"  splits: neighbor-holdout {len(nb_idx)}  "
          f"random-holdout {len(rh_idx)}  train-anchor {len(train_pool)}")

    X_nb, X_rh, X_tr = X_gen[nb_idx], X_gen[rh_idx], X_gen[train_pool]
    base_st = X_st @ W0_np + b0_np
    base_ho = X_ho @ W0_np + b0_np
    base_nb = X_nb @ W0_np + b0_np
    base_rh = X_rh @ W0_np + b0_np
    base_tr = X_tr @ W0_np + b0_np

    gen_face = float(base_tr[:, face_m].mean())
    self_face = float(base_st[:, face_m].mean())
    if self_face > gen_face + 0.003:
        target_face = gen_face
        print(f"\nFACE means (frozen readout)  -> PULL TO FAIRFACE MEAN "
              f"(target is elevated)")
    else:
        target_face = self_face - args.min_drop
        print(f"\nFACE means (frozen readout)  -> PULL THIS IDENTITY DOWN "
              f"by {args.min_drop:.3f} (not elevated vs FairFace; still a "
              f"selective-control test)")
    print(f"  suppress-to               {target_face:+.5f}")
    print(f"  train-anchor FairFace     {gen_face:+.5f}")
    print(f"  train {person:16s}    {self_face:+.5f}")
    print(f"  holdout {person:14s}    {base_ho[:, face_m].mean():+.5f}")
    print(f"  TRIBE neighbors           {base_nb[:, face_m].mean():+.5f}  "
          f"(sim {sim[nb_idx].mean():+.3f}..{sim[nb_idx].max():+.3f})")
    print(f"  random FairFace hold      {base_rh[:, face_m].mean():+.5f}")

    nonface_t = torch.tensor(nonface_w, device=device)
    target_t = torch.tensor(target_face, device=device)

    def fit():
        torch.manual_seed(args.seed)
        U = nn.Parameter(torch.randn(2048, args.rank, device=device) * 0.01)
        V = nn.Parameter(torch.zeros(args.rank, 20484, device=device))
        opt = torch.optim.AdamW([U, V], lr=args.lr, weight_decay=args.wd)
        Xs = torch.tensor(X_st, device=device)
        Xg = torch.tensor(X_tr, device=device)
        bs = torch.tensor(base_st, device=device)
        bg = torch.tensor(base_tr, device=device)
        best, best_UV = 1e18, None
        for ep in range(args.epochs):
            opt.zero_grad()
            W = W0_t + U @ V
            ps = Xs @ W + b0_t
            pg = Xg @ W + b0_t
            suppress = ((ps[:, face_m].mean(1) - target_t) ** 2).mean()
            # hard pin: general FACE per image stays at frozen baseline
            pin_face = ((pg[:, face_m].mean(1) - bg[:, face_m].mean(1)) ** 2).mean()
            pin_gen = ((pg - bg) ** 2).mean()
            pin_mia_body = (((ps - bs) ** 2) * nonface_t).mean()
            loss = (args.lam_suppress * suppress
                    + args.lam_anchor_face * pin_face
                    + args.lam_anchor * (pin_gen + pin_mia_body))
            loss.backward()
            opt.step()
            lv = float(loss.item())
            if lv < best:
                best, best_UV = lv, (U.detach().clone(), V.detach().clone())
            if ep % 50 == 0 or ep == args.epochs - 1:
                with torch.no_grad():
                    fs = ps[:, face_m].mean().item()
                    fg = pg[:, face_m].mean().item()
                print(f"  ep {ep:3d}  loss={lv:.5f}  "
                      f"studio FACE={fs:+.5f}  anchor FACE={fg:+.5f}", flush=True)
        return best_UV

    print(f"\nTraining rank-{args.rank}  lam_sup={args.lam_suppress}  "
          f"lam_anchor={args.lam_anchor}  lam_anchor_face={args.lam_anchor_face}")
    U, V = fit()

    def apply(X):
        with torch.no_grad():
            Xt = torch.tensor(X, device=device, dtype=torch.float32)
            return (Xt @ (W0_t + U @ V) + b0_t).cpu().numpy()

    ft_st, ft_ho = apply(X_st), apply(X_ho)
    ft_nb, ft_rh, ft_tr = apply(X_nb), apply(X_rh), apply(X_tr)
    report(f"{person.upper()} TRAIN (suppress)", base_st, ft_st, masks)
    report(f"{person.upper()} HOLDOUT", base_ho, ft_ho, masks)
    report("TRIBE-NEIGHBOR HOLDOUT (never trained)", base_nb, ft_nb, masks)
    report("RANDOM FAIRFACE HOLDOUT (never trained)", base_rh, ft_rh, masks)
    report("FAIRFACE TRAIN ANCHOR", base_tr, ft_tr, masks)

    def face(a):
        return float(a[:, face_m].mean())

    wild_drop = face(base_ho) - face(ft_ho)
    rand_drop = face(base_rh) - face(ft_rh)
    nb_drop = face(base_nb) - face(ft_nb)
    intended = (face(base_ho) - target_face)
    print("\n" + "=" * 64)
    print("V2 VERDICT")
    print(f"  {person} holdout FACE drop {wild_drop:+.5f}  "
          f"({100 * wild_drop / (intended + 1e-9):.0f}% of intended move)")
    print(f"  random FairFace drop      {rand_drop:+.5f}  (must stay ~0)")
    print(f"  TRIBE-neighbor drop       {nb_drop:+.5f}")
    selective = abs(rand_drop) < 0.4 * abs(wild_drop) if abs(wild_drop) > 1e-4 else abs(rand_drop) < 0.002
    closed = wild_drop > 0.5 * max(intended, 1e-4)
    if closed and selective:
        print(f"  OK: identity-selective. Holdout {person} moved, random faces did not.")
    elif closed and not selective:
        print("  FAIL: target moved but random FairFace moved too. Global FACE damp.")
    elif selective and not closed:
        print(f"  FAIL: selective but holdout {person} did not move enough.")
    else:
        print(f"  FAIL: neither selective nor effective on holdout {person}.")
    print("=" * 64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, U=U.cpu().numpy(), V=V.cpu().numpy(), rank=args.rank,
             target_face=target_face,
             split_mode=args.split_mode,
             lam_suppress=args.lam_suppress, lam_anchor=args.lam_anchor,
             lam_anchor_face=args.lam_anchor_face,
             neighbor_names=np.array([gen_names[i] for i in nb_idx]),
             neighbor_sim=sim[nb_idx],
             wild_drop=wild_drop, rand_drop=rand_drop, nb_drop=nb_drop)
    print(f"\nSaved -> {args.out}")
    if args.save_preds is not None:
        from chunk_utils import save_npz
        # restore original zip order is not required; train then holdout is fine
        P = np.concatenate([base_st, base_ho]).astype(np.float32)
        names = np.array(list(st_names) + list(ho_names))
        save_npz(args.save_preds, preds=P, filenames=names)
        print(f"Saved baseline preds -> {args.save_preds}  {P.shape}")


if __name__ == "__main__":
    main()

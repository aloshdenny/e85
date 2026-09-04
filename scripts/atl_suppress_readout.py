"""
atl_suppress_readout.py

Tests a specific architectural claim about where to target for clean,
identity-selective face suppression, instead of assuming OFA/FFA is
automatically the right answer.

THE CLAIM (paraphrased from the user's own neuroanatomy notes): the anterior
temporal lobe is not a face-specific structure. It's a convergence zone where
shape, color, prior memory, and other modality streams merge into a single
object/identity representation ("circular-ish, red & shiny" -> apple). Damage
there (or, here, suppression there) should NOT give clean, selective
person-identity suppression the way OFA/FFA did -- either it does very little
(consistent with abliterationv2.py's old finding: ATL/TP were near-zero for
Mia in the original Destrieux-contrast diagnostic), or it "leaks" into
general object/face processing because you're not hitting a face-specific
node, you're hitting the place where everything ties together.

THE TEST. Same rank-16 low-rank residual recipe already validated for OFA/FFA
(mia_suppress_readout_v2.py, face_top split, lam_suppress=15 -- OK on both
Mia and Sins), but with the suppression objective and the hard anchor pin
SWAPPED from FACE(OFA+FFA) to ATL_TP (Destrieux Pole_temporal + G_temporal_inf
+ G_oc-temp_med-Parahip, bilateral -- same labels as abliterationv2.py's
SECONDARY_FACE_ROIS). FACE(OFA+FFA) is deliberately NOT hard-pinned here --
it's folded into the generic, weak (lam_anchor=3) "everything else" anchor,
exactly the protection ATL/TP got in the original OFA/FFA recipe. This is
the symmetric flip of that experiment, and FACE(OFA+FFA) drift on the target's
own holdout is printed explicitly as the collateral-damage readout: if
suppressing ATL drags OFA/FFA down too, that is the "everything merges here"
prediction showing up empirically, not just architecturally.

Usage:
  python scripts/atl_suppress_readout.py --target-zip target/mia.zip --person Mia \
      --split-mode face_top --lam-suppress 15.0
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
    IMAGE_EXTS, capture_bottlenecks, find_fmri_encoder,
    load_general_images, load_mia_from_zip,
)
from chunk_utils import ensure_fused_zip
from infer_target_face import decode_image_from_zip

REPORT_ROIS = ["ATL_TP", "TP", "ATL", "FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR"]


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
    for r in REPORT_ROIS:
        bm, fm = base[:, masks[r]].mean(), ft[:, masks[r]].mean()
        print(f"{r:>14s} {bm:+10.5f} {fm:+10.5f} {fm - bm:+10.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mia-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--target-zip", type=Path, default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--split-mode", choices=["random", "in_the_wild", "face_top"],
                    default="face_top",
                    help="face_top here holds out the images with the highest ATL_TP "
                         "(not FACE) under the frozen readout.")
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
    ap.add_argument("--lam-suppress", type=float, default=15.0)
    ap.add_argument("--lam-anchor", type=float, default=3.0,
                    help="Generic anchor, covers FACE(OFA+FFA) this time.")
    ap.add_argument("--lam-anchor-target", type=float, default=20.0,
                    help="Hard pin on general-population ATL_TP.")
    ap.add_argument("--min-drop", type=float, default=0.015)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-folder", type=Path,
                    default=Path("/home/research/.cache/huggingface"))
    ap.add_argument("--bottleneck-cache", type=Path,
                    default=Path("/home/research/e85_scratch/v2_bottlenecks.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("./abliterated/atl_suppress_readout.npz"))
    ap.add_argument("--target-roi", choices=["atl", "face", "both"], default="atl",
                    help="atl = ATL_TP only (original experiment). face = FACE(OFA+FFA) "
                         "only (reproduces mia_suppress_readout_v2.py's target under this "
                         "script's diagnostics). both = union of ATL_TP and FACE(OFA+FFA), "
                         "suppressed together as one combined objective -- the untested "
                         "third condition.")
    ap.add_argument("--mask-to-target", action="store_true",
                    help="Architecturally restrict the residual to the ATL_TP output "
                         "columns -- zero elsewhere by construction (both in training "
                         "and at inference), not just by a soft anchor penalty. Tests "
                         "whether a surgically column-restricted edit matches or beats "
                         "the free (loss-only-targeted) residual.")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    person = args.person or "Mia"
    zip_path = args.target_zip or args.mia_zip
    face_top_items = None

    if args.split_mode == "in_the_wild":
        studio, wild = load_mia_from_zip(zip_path)
        print(f"{person}: {len(studio)} train, {len(wild)} in-the-wild holdout from {zip_path}")
    elif args.split_mode == "face_top":
        face_top_items = load_all_from_zip(zip_path)
        studio, wild = [], []
        print(f"{person}: {len(face_top_items)} total; ATL_TP-top split pending from {zip_path}")
    else:
        items = load_all_from_zip(zip_path)
        rng.shuffle(items)
        n_ho = max(5, int(round(len(items) * args.holdout_frac)))
        if n_ho >= len(items) - 8:
            n_ho = max(5, len(items) // 5)
        wild, studio = items[:n_ho], items[n_ho:]
        print(f"{person}: {len(studio)} train, {len(wild)} holdout (random) from {zip_path}")

    reuse_gen = args.bottleneck_cache.exists()
    general = []
    if not reuse_gen:
        general = load_general_images(args.general_zip, args.n_general, args.seed)
        print(f"general portraits loaded: {len(general)}")
    if face_top_items is None and (len(studio) < 8 or len(wild) < 5):
        raise SystemExit("need enough train + holdout images of the target")

    cache = args.bottleneck_cache
    recapture_target = True   # always recapture: this is a different target ROI/person set each run

    def split_atl_top(X_all, all_names, W0_np, b0_np):
        # Must match whichever ROI --target-roi actually optimises, or the
        # "top" holdout wouldn't be the images that matter for THIS run's
        # objective. Computed here (not from the later target_m) because the
        # split has to happen before bottleneck capture finishes.
        masks_e = build_masks()
        if args.target_roi == "atl":
            target_me = masks_e["ATL_TP"]
        elif args.target_roi == "face":
            target_me = masks_e["FACE(OFA+FFA)"]
        else:
            target_me = masks_e["ATL_TP"] | masks_e["FACE(OFA+FFA)"]
        scores = (X_all @ W0_np + b0_np)[:, target_me].mean(axis=1)
        n_ho = max(5, int(round(len(all_names) * args.holdout_frac)))
        if n_ho >= len(all_names) - 8:
            n_ho = max(5, len(all_names) // 5)
        ho_idx = np.argsort(-scores)[:n_ho]
        st_idx = np.array([i for i in range(len(all_names)) if i not in set(ho_idx)])
        print(f"{args.target_roi}-top split: {len(st_idx)} train score={scores[st_idx].mean():+.5f}  "
              f"{len(ho_idx)} holdout score={scores[ho_idx].mean():+.5f}  "
              f"(holdout range {scores[ho_idx].min():+.5f}..{scores[ho_idx].max():+.5f})")
        return (X_all[st_idx], [all_names[i] for i in st_idx],
                X_all[ho_idx], [all_names[i] for i in ho_idx])

    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        X_gen, gen_names = d["X_gen"], [str(x) for x in d["gen_names"]]
        W0_np, b0_np = d["W0"], d["b0"]
        print(f"reusing FairFace cache {cache}  gen {X_gen.shape}")
        print("Loading TribeModel for target bottlenecks...")
        model, fe = find_fmri_encoder(args.cache_folder)
        device = fe.device
        W0_t = fe.predictor.weights[0].detach().clone()
        b0_t = fe.predictor.bias[0].detach().clone()
        W0_np, b0_np = W0_t.cpu().numpy(), b0_t.cpu().numpy()
        tmp_root = get_tmp_root()
        if face_top_items is not None:
            print(f"Capturing bottlenecks: {len(face_top_items)} all...")
            X_all, all_names = capture_bottlenecks(model, fe, face_top_items, args.batch, tmp_root)
            X_st, st_names, X_ho, ho_names = split_atl_top(X_all, all_names, W0_np, b0_np)
        else:
            print(f"Capturing bottlenecks: {len(studio)} train...")
            X_st, st_names = capture_bottlenecks(model, fe, studio, args.batch, tmp_root)
            print(f"Capturing bottlenecks: {len(wild)} holdout...")
            X_ho, ho_names = capture_bottlenecks(model, fe, wild, args.batch, tmp_root)
        free()
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
            print(f"Capturing bottlenecks: {len(face_top_items)} all...")
            X_all, all_names = capture_bottlenecks(model, fe, face_top_items, args.batch, tmp_root)
            X_st, st_names, X_ho, ho_names = split_atl_top(X_all, all_names, W0_np, b0_np)
        else:
            print(f"Capturing bottlenecks: {len(studio)} train...")
            X_st, st_names = capture_bottlenecks(model, fe, studio, args.batch, tmp_root)
            print(f"Capturing bottlenecks: {len(wild)} holdout...")
            X_ho, ho_names = capture_bottlenecks(model, fe, wild, args.batch, tmp_root)
        print(f"Capturing bottlenecks: {len(general)} general...")
        X_gen, gen_names = capture_bottlenecks(model, fe, general, args.batch, tmp_root)
        free()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X_st=X_st, st_names=np.array(st_names), X_ho=X_ho,
                 ho_names=np.array(ho_names), X_gen=X_gen, gen_names=np.array(gen_names),
                 W0=W0_np, b0=b0_np)
        print(f"cached bottlenecks -> {cache}")

    if len(X_st) < 8 or len(X_ho) < 5:
        raise SystemExit(f"need enough train + holdout after split ({len(X_st)}/{len(X_ho)})")

    masks = build_masks()
    atl_only_m = masks["ATL_TP"]
    face_only_m = masks["FACE(OFA+FFA)"]
    if args.target_roi == "atl":
        target_m, face_m = atl_only_m, face_only_m          # watch FACE for collateral
    elif args.target_roi == "face":
        target_m, face_m = face_only_m, atl_only_m          # watch ATL for collateral
    else:  # both
        target_m = atl_only_m | face_only_m
        face_m = np.zeros(20484, dtype=bool)                # no separate "other face-pathway
                                                              # region" left to watch; V1/AUD/
                                                              # MOTOR (already reported) cover
                                                              # the collateral question instead
    generic_w = np.ones(20484, dtype=np.float32)
    generic_w[target_m] = 0.0             # target gets its own dedicated loss term below

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

    gen_target = float(base_tr[:, target_m].mean())
    self_target = float(base_st[:, target_m].mean())
    if self_target > gen_target + 0.003:
        target_val = gen_target
        print("\nATL_TP means (frozen readout)  -> PULL TO FAIRFACE MEAN (target is elevated)")
    else:
        target_val = self_target - args.min_drop
        print(f"\nATL_TP means (frozen readout)  -> PULL THIS IDENTITY DOWN by {args.min_drop:.3f} "
              f"(not elevated vs FairFace)")
    print(f"  suppress-to               {target_val:+.5f}")
    print(f"  train-anchor FairFace     {gen_target:+.5f}")
    print(f"  train {person:16s}    {self_target:+.5f}")
    print(f"  holdout {person:14s}    {base_ho[:, target_m].mean():+.5f}")
    print(f"  holdout {person} FACE(OFA+FFA) (unpinned, watch only) "
          f"{base_ho[:, face_m].mean():+.5f}")

    generic_t = torch.tensor(generic_w, device=device)
    target_t = torch.tensor(target_val, device=device)

    # Column mask for --mask-to-target: 1.0 on ATL_TP output columns, 0.0
    # everywhere else. Applied to (U @ V) every forward pass -- training AND
    # apply() -- so non-target columns are EXACTLY W0's original value, not
    # merely close to it. This is the architectural version of the soft
    # anchor: guaranteed by construction rather than learned.
    col_mask = torch.zeros(20484, device=device)
    col_mask[target_m] = 1.0

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
            residual = U @ V
            if args.mask_to_target:
                residual = residual * col_mask[None, :]
            W = W0_t + residual
            ps = Xs @ W + b0_t
            pg = Xg @ W + b0_t
            suppress = ((ps[:, target_m].mean(1) - target_t) ** 2).mean()
            pin_target = ((pg[:, target_m].mean(1) - bg[:, target_m].mean(1)) ** 2).mean()
            pin_gen = ((pg - bg) ** 2).mean()
            pin_body = (((ps - bs) ** 2) * generic_t).mean()
            loss = (args.lam_suppress * suppress
                    + args.lam_anchor_target * pin_target
                    + args.lam_anchor * (pin_gen + pin_body))
            loss.backward()
            opt.step()
            lv = float(loss.item())
            if lv < best:
                best, best_UV = lv, (U.detach().clone(), V.detach().clone())
            if ep % 50 == 0 or ep == args.epochs - 1:
                with torch.no_grad():
                    fs = ps[:, target_m].mean().item()
                    ff = ps[:, face_m].mean().item()
                print(f"  ep {ep:3d}  loss={lv:.5f}  studio ATL_TP={fs:+.5f}  "
                      f"studio FACE(unpinned)={ff:+.5f}", flush=True)
        return best_UV

    print(f"\nTraining rank-{args.rank}  lam_sup={args.lam_suppress}  "
          f"lam_anchor={args.lam_anchor}  lam_anchor_target={args.lam_anchor_target}")
    U, V = fit()

    def apply(X):
        with torch.no_grad():
            Xt = torch.tensor(X, device=device, dtype=torch.float32)
            residual = U @ V
            if args.mask_to_target:
                residual = residual * col_mask[None, :]
            return (Xt @ (W0_t + residual) + b0_t).cpu().numpy()

    ft_st, ft_ho = apply(X_st), apply(X_ho)
    ft_nb, ft_rh, ft_tr = apply(X_nb), apply(X_rh), apply(X_tr)
    report(f"{person.upper()} TRAIN (suppress)", base_st, ft_st, masks)
    report(f"{person.upper()} HOLDOUT", base_ho, ft_ho, masks)
    report("TRIBE-NEIGHBOR HOLDOUT (never trained)", base_nb, ft_nb, masks)
    report("RANDOM FAIRFACE HOLDOUT (never trained)", base_rh, ft_rh, masks)
    report("FAIRFACE TRAIN ANCHOR", base_tr, ft_tr, masks)

    def roi_val(a, m):
        return float(a[:, m].mean())

    wild_drop = roi_val(base_ho, target_m) - roi_val(ft_ho, target_m)
    rand_drop = roi_val(base_rh, target_m) - roi_val(ft_rh, target_m)
    nb_drop = roi_val(base_nb, target_m) - roi_val(ft_nb, target_m)
    face_collateral = roi_val(base_ho, face_m) - roi_val(ft_ho, face_m)
    face_rand_collateral = roi_val(base_rh, face_m) - roi_val(ft_rh, face_m)
    intended = roi_val(base_ho, target_m) - target_val

    # ---- PATTERN-level check, not just the mean -------------------------
    # measure_identity_signal.py's own opening finding: averaging over an
    # ROI is exactly the operation that can hide identity-carrying structure
    # -- a shared low-rank direction could reshuffle the 776-vertex
    # FACE(OFA+FFA) pattern per image while leaving its MEAN exactly at
    # baseline (which is what every prior run measured as "0.00000
    # collateral"). This checks the actual per-image vectors, not their
    # average, for the target's own holdout and for the random FairFace
    # holdout.
    def pattern_stats(base, ft, mask, label):
        Bp, Fp = base[:, mask], ft[:, mask]
        Bn = Bp / np.maximum(np.linalg.norm(Bp, axis=1, keepdims=True), 1e-9)
        Fn = Fp / np.maximum(np.linalg.norm(Fp, axis=1, keepdims=True), 1e-9)
        cos = (Bn * Fn).sum(1)
        l2 = np.linalg.norm(Fp - Bp, axis=1)
        rel_l2 = l2 / np.maximum(np.linalg.norm(Bp, axis=1), 1e-9)
        print(f"  {label:34s} cos={cos.mean():.5f} (min {cos.min():.5f})  "
              f"rel_L2={rel_l2.mean():.4f} (max {rel_l2.max():.4f})")
        return float(cos.mean()), float(rel_l2.mean())

    print("\nPATTERN-level FACE(OFA+FFA) check (776-d vector per image, not the mean)")
    print("  cos=1.0 / rel_L2=0.0 means the pattern is untouched; a mean-collateral of")
    print("  0.00000 does NOT by itself guarantee this.")
    face_cos_ho, face_rl2_ho = pattern_stats(base_ho, ft_ho, face_m,
                                             f"{person} holdout FACE pattern")
    face_cos_rh, face_rl2_rh = pattern_stats(base_rh, ft_rh, face_m,
                                             "random FairFace FACE pattern")
    atl_cos_ho, atl_rl2_ho = pattern_stats(base_ho, ft_ho, target_m,
                                           f"{person} holdout ATL_TP pattern (target, for reference)")

    print("\n" + "=" * 70)
    print("ATL VERDICT")
    print(f"  {person} holdout ATL_TP drop      {wild_drop:+.5f}  "
          f"({100 * wild_drop / (intended + 1e-9):.0f}% of intended move)")
    print(f"  random FairFace ATL_TP drop     {rand_drop:+.5f}  (must stay ~0)")
    print(f"  TRIBE-neighbor ATL_TP drop      {nb_drop:+.5f}")
    print(f"  {person} holdout FACE collateral  {face_collateral:+.5f}  "
          f"(unpinned -- the claim predicts this moves too)")
    print(f"  random FairFace FACE collateral {face_rand_collateral:+.5f}  "
          f"(if this moves, ATL suppression leaked into general face processing)")
    selective = abs(rand_drop) < 0.4 * abs(wild_drop) if abs(wild_drop) > 1e-4 else abs(rand_drop) < 0.002
    closed = wild_drop > 0.5 * max(intended, 1e-4)
    if closed and selective:
        print(f"  ATL_TP itself: OK, identity-selective by the same bar OFA/FFA passed.")
    elif closed and not selective:
        print("  ATL_TP itself: FAIL -- target moved but random FairFace moved too. Global damp.")
    elif selective and not closed:
        print(f"  ATL_TP itself: FAIL -- selective but holdout {person} did not move enough.")
    else:
        print(f"  ATL_TP itself: FAIL -- neither selective nor effective.")
    if abs(face_collateral) > 0.3 * abs(wild_drop) and abs(wild_drop) > 1e-4:
        print(f"  COLLATERAL: FACE(OFA+FFA) moved substantially with ATL_TP "
              f"({abs(face_collateral)/max(abs(wild_drop),1e-9):.0%} of the ATL move) -- "
              f"consistent with ATL not being a clean, separable node.")
    else:
        print(f"  COLLATERAL: FACE(OFA+FFA) stayed roughly put "
              f"({abs(face_collateral)/max(abs(wild_drop),1e-9):.0%} of the ATL move).")
    if args.mask_to_target:
        print(f"  mask_to_target=True: FACE(OFA+FFA) mean collateral is architecturally "
              f"forced to 0 -- the number above should read essentially 0.00000 by "
              f"construction. What ISN'T forced to zero is the PATTERN check below.")
    print(f"\n  PATTERN check (this is the number the mean can hide):")
    print(f"    {person} holdout FACE  cos={face_cos_ho:.5f}  rel_L2={face_rl2_ho:.4f}")
    print(f"    random FairFace FACE   cos={face_cos_rh:.5f}  rel_L2={face_rl2_rh:.4f}")
    if face_cos_ho < 0.999 or face_rl2_ho > 0.02:
        print(f"    NOT untouched: the FACE pattern moved even though the mean read "
              f"~{face_collateral:+.5f}. Averaging was hiding real structure change.")
    else:
        print(f"    genuinely untouched, not just mean-zero: pattern survives at "
              f"cos>=0.999.")
    print("=" * 70)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, U=U.cpu().numpy(), V=V.cpu().numpy(), rank=args.rank,
             target_val=target_val, split_mode=args.split_mode,
             mask_to_target=args.mask_to_target,
             lam_suppress=args.lam_suppress, lam_anchor=args.lam_anchor,
             lam_anchor_target=args.lam_anchor_target,
             wild_drop=wild_drop, rand_drop=rand_drop, nb_drop=nb_drop,
             face_collateral=face_collateral, face_rand_collateral=face_rand_collateral,
             face_cos_ho=face_cos_ho, face_rl2_ho=face_rl2_ho,
             face_cos_rh=face_cos_rh, face_rl2_rh=face_rl2_rh,
             atl_cos_ho=atl_cos_ho, atl_rl2_ho=atl_rl2_ho)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

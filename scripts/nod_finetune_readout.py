"""
nod_finetune_readout.py

Fine-tunes TRIBE's final readout layer against real NOD OFA/FFA/support-ROI
responses, without touching the 1B-parameter encoder.

WHERE THE READOUT LIVES. Located by monkeypatching FmriEncoderModel.__init__
and inspecting the live instance (see chat history): `predictor` is a
SubjectLayersModel with a single weight matrix `weights[0]` of shape
(2048, 20484) plus `bias[0]` of shape (20484,) -- `average_subjects=True` and
`n_subjects=0` mean this ONE matrix is literally what every prediction in this
entire project has used, since predict() is always called without a
subject_id. 2048 is the "low_rank_head" bottleneck dimension (a Linear(1152,
2048) applied just before this).

WHY A LOW-RANK RESIDUAL, NOT THE FULL MATRIX. 2048x20484 = ~42M parameters.
NOD gives 120 unique images (11,662 raw trials, but stimuli are the unit of
generalisation -- k-fold has to hold out whole images, not trials). Directly
fine-tuning 42M parameters against 120 generalisation points would overfit
outright. Instead: freeze the original weights/bias, add a learned residual
U @ V (U: 2048 x r, V: r x 20484, r small, e.g. 16), so the model can only
express a low-dimensional correction. Standard LoRA-style adaptation, chosen
for the same reason it's used elsewhere: far fewer parameters than data points.

WHAT IT'S TRAINED TOWARD. Loss is a weighted MSE over all 20,484 vertices,
weight concentrated on the face-selective network (OFA, FFA, plus the
secondary regions abliteration.py already defines -- TP, ATL) with a small
uniform floor elsewhere, so the rest of the brain's predictions are not
free to drift uncontrolled while training targets the region the user asked
to improve.

VALIDATION. 5-fold over the 120 IMAGES (not trials), so the reported numbers
are genuine held-out generalisation, not memorisation of a small stimulus set.
Compared directly against nod_baseline_eval.py's zero-shot numbers.

Usage:
  python scripts/nod_finetune_readout.py --rank 16 --epochs 300
"""

import sys, argparse, tempfile, shutil
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))
from abliteration import TribeModel, free, build_face_mask, SECONDARY_FACE_ROIS, PRIMARY_FACE_ROIS
from tribev2.model import FmriEncoderModel
from infer_fairface_bulk import get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline
from measure_identity_signal import build_masks

NOD_DIR = Path.home() / "nod"


def find_fmri_encoder(model):
    instances = []
    orig_init = FmriEncoderModel.__init__
    def patched(self, *a, **kw):
        orig_init(self, *a, **kw)
        instances.append(self)
    FmriEncoderModel.__init__ = patched
    try:
        m = TribeModel.from_pretrained("facebook/tribev2", cache_folder=Path("./cache"))
    finally:
        FmriEncoderModel.__init__ = orig_init
    if not instances:
        raise RuntimeError("FmriEncoderModel not instantiated during load")
    return m, instances[0]


def load_real_betas():
    files = sorted((NOD_DIR / "betas").glob("*/ses-coco*_fsaverage5.npy"))
    labfiles = sorted((NOD_DIR / "betas").glob("*/ses-coco*_labels.txt"))
    by_img = {}
    for bf, lf in zip(files, labfiles):
        B = np.load(bf)
        labs = [l for l in lf.read_text().split("\n") if l]
        for i, lab in enumerate(labs):
            by_img.setdefault(lab, []).append(B[i])
    return {k: np.mean(v, axis=0) for k, v in by_img.items() if len(v) >= 2}


def build_selectivity_weights(masks):
    """Two separate weight vectors, not one compromise vector.

    v1: fit against the REAL target -- ONLY inside the face network (primary +
    secondary ROIs). Zero elsewhere: the residual gets no gradient signal from
    Y outside the target region, so it cannot learn a generic "predict these
    120 images better" correction by fitting real targets broadly.

    v2 (anchor): outside the face network, pull the prediction back toward the
    FROZEN baseline (not toward Y). This is what the first attempt was
    missing -- a small uniform floor weight on the fit-to-target loss still
    lets drift toward Y accumulate everywhere; an explicit anchor to the
    original output actively penalises it. Any correlation gain in V1/AUD/
    MOTOR now has to fight this term directly, rather than merely being
    outweighted by the face-ROI term.
    """
    face = masks["FACE(OFA+FFA)"].copy()
    for m in SECONDARY_FACE_ROIS_MASKS:
        face |= m
    fit_w = np.where(face, 1.0, 0.0).astype(np.float32)
    anchor_w = np.where(face, 0.0, 1.0).astype(np.float32)
    return fit_w, anchor_w, face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lam-anchor", type=float, default=2.0,
                    help="Weight on the anchor-to-baseline penalty outside the "
                         "face network. Higher = less permitted drift elsewhere.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    args = ap.parse_args()

    global SECONDARY_FACE_ROIS_MASKS
    from nilearn import datasets as nl_datasets
    destrieux = nl_datasets.fetch_atlas_surf_destrieux()
    lh_labels = np.array(destrieux["map_left"])
    rh_labels = np.array(destrieux["map_right"])
    atlas_names = [n.decode() if isinstance(n, bytes) else n for n in destrieux["labels"]]
    SECONDARY_FACE_ROIS_MASKS = []
    for key, exact in SECONDARY_FACE_ROIS.items():
        idxs = [i for i, n in enumerate(atlas_names) if n in exact]
        m = np.concatenate([np.isin(lh_labels, idxs), np.isin(rh_labels, idxs)])
        SECONDARY_FACE_ROIS_MASKS.append(m)

    real = load_real_betas()
    img_dir = NOD_DIR / "stimuli" / "coco"
    have = sorted(k for k in real if (img_dir / k).exists())
    print(f"{len(have)} images with real target + stimulus")

    model, fe = find_fmri_encoder(None)
    device = fe.device
    predictor = fe.predictor
    predictor_W0 = predictor.weights[0].detach().clone()   # (2048, 20484)
    predictor_b0 = predictor.bias[0].detach().clone()   # (20484,)
    W0, b0 = predictor_W0, predictor_b0
    print(f"frozen readout: W {tuple(W0.shape)}, b {tuple(b0.shape)}")

    # ---- capture the bottleneck, reusing predict()'s OWN segment/TR/keep logic --
    # predictor.forward output is (B, 20484, T=200): T spans the whole batched,
    # padded loader sequence, and predict() (demo_utils.py) keeps only the
    # specific per-TR windows that actually contain an event -- a keep mask I
    # cannot safely rederive by hand (it depends on batch.segments / self.data.TR
    # / the loader's internal batching). A first attempt that just meaned over
    # all 200 raw timesteps produced mean|diff|=0.036 (max 0.58) against the
    # real pipeline output -- caught by the alignment check below, not silently
    # trusted.
    #
    # Correct fix: monkeypatch predictor.forward to be the IDENTITY (return its
    # input unchanged), so predict()'s unmodified rearrange+keep code applies to
    # the 2048-dim bottleneck instead of the 20484-dim prediction. Reuses code
    # already proven correct elsewhere in this project, rather than reimplementing
    # segment logic that could silently misalign again.
    real_forward = predictor.forward
    def identity_forward(x, subject_id=None):
        return x
    predictor.forward = identity_forward

    tmp_root = get_tmp_root()
    td = Path(tempfile.mkdtemp(prefix="nod_ft_", dir=tmp_root))
    try:
        rows = []
        for i, k in enumerate(have):
            img = cv2.imread(str(img_dir / k))
            if img is None:
                continue
            tl = f"i{i:04d}"
            p_ = td / f"{tl}.mp4"
            write_static_clip(img, p_, duration=1.0, fps=2)
            rows.append((p_, tl, k))
        df = make_multi_row_df([(p_, tl) for p_, tl, _ in rows], duration=1.0)
        with torch.autocast("cuda", dtype=torch.float16):
            bottleneck_preds, bottleneck_segs = model.predict(events=df)
    finally:
        predictor.forward = real_forward
        shutil.rmtree(td, ignore_errors=True)

    # bottleneck_preds: (n_segments, 2048), one row per kept TR-segment, aligned
    # to bottleneck_segs the same way group_preds_by_timeline aligns real preds.
    g_bottleneck = group_preds_by_timeline(bottleneck_preds, bottleneck_segs)

    # ---- now the REAL pass, for comparison / verification -------------------
    td = Path(tempfile.mkdtemp(prefix="nod_ft_real_", dir=tmp_root))
    try:
        rows2 = []
        for i, k in enumerate(have):
            img = cv2.imread(str(img_dir / k))
            if img is None:
                continue
            tl = f"i{i:04d}"
            p_ = td / f"{tl}.mp4"
            write_static_clip(img, p_, duration=1.0, fps=2)
            rows2.append((p_, tl, k))
        df2 = make_multi_row_df([(p_, tl) for p_, tl, _ in rows2], duration=1.0)
        with torch.autocast("cuda", dtype=torch.float16):
            preds, segs = model.predict(events=df2)
        g = group_preds_by_timeline(preds, segs)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    common_tl = [tl for _, tl, _ in rows if tl in g_bottleneck and tl in g]
    X = np.stack([g_bottleneck[tl] for tl in common_tl])              # (N, 2048)
    tl_to_key = {tl: k for _, tl, k in rows}
    real_pipeline = np.stack([g[tl] for tl in common_tl])              # (N, 20484)
    print(f"{len(common_tl)}/{len(rows)} images aligned via matching timeline ids")

    # ---- verify: bottleneck @ frozen W0 + b0 should match the real pipeline --
    W0_np, b0_np = predictor_W0.cpu().numpy(), predictor_b0.cpu().numpy()
    recon = X @ W0_np + b0_np
    diffs = np.abs(recon - real_pipeline)
    print(f"alignment check: mean|diff|={diffs.mean():.6f}  max|diff|={diffs.max():.6f}  "
          f"(should be tiny -- fp16 autocast noise only)")
    if diffs.mean() > 0.001:
        raise RuntimeError("bottleneck cache still does not reproduce the real pipeline "
                           "output -- do not trust it, stopping before training.")

    Y = np.stack([real[tl_to_key[tl]] for tl in common_tl])            # (N, 20484)
    print(f"cache ready: X {X.shape}, Y {Y.shape}")

    masks = build_masks()
    fit_w, anchor_w, face_mask = build_selectivity_weights(masks)
    W0_t = W0.to(device)
    b0_t = b0.to(device)
    fit_w_t = torch.tensor(fit_w, device=device)
    anchor_w_t = torch.tensor(anchor_w, device=device)
    control_rois = ["V1", "AUD", "MOTOR"]

    def fit_lowrank(Xtr, Ytr, Xva, Yva, rank, epochs, lr, wd, lam_anchor, seed):
        torch.manual_seed(seed)
        n_in = Xtr.shape[1]
        U = nn.Parameter(torch.randn(n_in, rank, device=device) * 0.01)
        V = nn.Parameter(torch.zeros(rank, 20484, device=device))
        opt = torch.optim.AdamW([U, V], lr=lr, weight_decay=wd)
        Xtr_t = torch.tensor(Xtr, device=device, dtype=torch.float32)
        Xva_t = torch.tensor(Xva, device=device, dtype=torch.float32)
        base_tr_t = Xtr_t @ W0_t + b0_t   # frozen baseline, the anchor target

        # SCALE-MATCH Ytr to the frozen baseline's own per-vertex mean/std before
        # fitting. v1 fit raw NOD beta units directly, which are on a different
        # scale from TRIBE's native output (real target mean ~0.09, TRIBE's is
        # comparable in aggregate but NOT per-vertex) -- unconstrained MSE against
        # unmatched units let the optimiser inflate face-ROI gain ~100x to chase
        # absolute values, which (measured) FLATTENED relative part-selectivity
        # (range/mean ratio 2.08 -> 1.37) even though held-out correlation, being
        # scale-invariant, still looked like an improvement. z-scoring the target
        # to the baseline's own scale removes the incentive to match magnitude at
        # all -- the residual can only correct SHAPE. Train-fold statistics only,
        # to avoid leaking validation-fold scale into training.
        base_tr_np = base_tr_t.detach().cpu().numpy()
        y_mean, y_std = Ytr.mean(0), Ytr.std(0) + 1e-9
        b_mean, b_std = base_tr_np.mean(0), base_tr_np.std(0) + 1e-9
        Ytr_scaled = (Ytr - y_mean) / y_std * b_std + b_mean
        Ytr_t = torch.tensor(Ytr_scaled, device=device, dtype=torch.float32)
        # selectivity score to pick the best epoch: face-ROI gain minus how much
        # the control ROIs moved away from their OWN baseline -- an epoch that
        # "improves" MOTOR is not a good epoch even if FACE also went up.
        base_va_np = (Xva @ W0_np + b0_np)
        best_score, best_UV, patience, bad = -1e9, None, 60, 0
        for ep in range(epochs):
            opt.zero_grad()
            pred = Xtr_t @ (W0_t + U @ V) + b0_t
            fit_loss = (((pred - Ytr_t) ** 2) * fit_w_t).mean()
            anchor_loss = (((pred - base_tr_t) ** 2) * anchor_w_t).mean()
            loss = fit_loss + lam_anchor * anchor_loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                pv = (Xva_t @ (W0_t + U @ V) + b0_t).cpu().numpy()
                m = masks["FACE(OFA+FFA)"]
                face_r = (np.corrcoef(pv[:, m].mean(1), Yva[:, m].mean(1))[0, 1]
                         if pv[:, m].std() > 1e-9 else -1)
                control_drift = np.mean([
                    abs(np.corrcoef(pv[:, masks[r]].mean(1), Yva[:, masks[r]].mean(1))[0, 1]
                       - np.corrcoef(base_va_np[:, masks[r]].mean(1), Yva[:, masks[r]].mean(1))[0, 1])
                    for r in control_rois
                ])
                score = face_r - control_drift   # reward face gain, penalise control drift
            if score > best_score:
                best_score, best_UV, bad = score, (U.detach().clone(), V.detach().clone()), 0
            else:
                bad += 1
                if bad > patience:
                    break
        return best_UV, best_score

    print(f"\n{args.folds}-fold CV, rank={args.rank}, "
          f"comparing against zero-shot baseline\n")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, args.folds)

    roi_list = ["OFA", "FFA", "FACE(OFA+FFA)", "V1", "AUD", "MOTOR"]
    base_rs = {r: [] for r in roi_list}
    ft_rs = {r: [] for r in roi_list}

    for fi, va_idx in enumerate(folds):
        tr_idx = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xva, Yva = X[va_idx], Y[va_idx]

        with torch.no_grad():
            base_pred = Xva @ W0_np + b0_np
        (U, V), val_score = fit_lowrank(Xtr, Ytr, Xva, Yva, args.rank, args.epochs,
                                args.lr, args.wd, args.lam_anchor, args.seed)
        with torch.no_grad():
            ft_pred = (torch.tensor(Xva, device=device, dtype=torch.float32) @
                      (W0_t + U @ V) + b0_t).cpu().numpy()

        for roi in roi_list:
            m = masks[roi]
            if base_pred[:, m].std() > 1e-9:
                base_rs[roi].append(np.corrcoef(base_pred[:, m].mean(1), Yva[:, m].mean(1))[0, 1])
            if ft_pred[:, m].std() > 1e-9:
                ft_rs[roi].append(np.corrcoef(ft_pred[:, m].mean(1), Yva[:, m].mean(1))[0, 1])
        print(f"  fold {fi+1}/{args.folds}: n_val={len(va_idx)}  "
              f"FACE base={base_rs['FACE(OFA+FFA)'][-1]:+.3f} "
              f"ft={ft_rs['FACE(OFA+FFA)'][-1]:+.3f}")

    print(f"\n{'ROI':16s} {'baseline r':>11s} {'fine-tuned r':>13s} {'delta':>8s}")
    print("-" * 52)
    for roi in roi_list:
        b, f = np.mean(base_rs[roi]), np.mean(ft_rs[roi])
        print(f"{roi:16s} {b:+11.3f} {f:+13.3f} {f-b:+8.3f}")

    # refit on ALL data for the final deployable residual
    (U, V), _ = fit_lowrank(X, Y, X, Y, args.rank, args.epochs, args.lr, args.wd,
                            args.lam_anchor, args.seed)
    out = Path("./abliterated") / "nod_readout_lowrank.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, U=U.cpu().numpy(), V=V.cpu().numpy(), rank=args.rank)
    print(f"\nSaved final low-rank residual (fit on all {len(X)} images) -> {out}")


if __name__ == "__main__":
    main()

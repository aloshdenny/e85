"""
likeness_gradient.py

Decides whether TRIBE's elevated face-cortex response to the target is about
her LIKENESS or merely about her photo set's provenance.

The goal this serves is not identity erasure. It is that the model should not
spike for the target or for people who look like her. Baseline: target images
predict FACE(OFA+FFA) at +0.09316 against +0.06803 for general faces. Before
trying to pull that down, two things have to be true, and neither has been
tested:

  1. GRADED BY LIKENESS. Among GENERAL faces -- none of them the target -- does
     the predicted face response rise with facenet similarity to her? If yes,
     "her likeness" is a real, continuous direction in face space and there is
     something principled to suppress. If the relationship is flat, the +0.09
     is a property of her 175 photos, not of her appearance, and suppressing it
     would hit every photo that shares their provenance.

  2. NOT JUST PHOTOMETRY. Her photos are professionally shot; general faces are
     snapshots. Similarity to her may be standing in for "well lit, tightly
     cropped, front facing". So the correlation is recomputed after regressing
     out the 21 photometric and geometric attributes, and it is checked against
     control ROIs. A likeness effect should survive the controls and should NOT
     appear in V1/AUD/MOTOR; a provenance effect will fail both ways.

Everything needed is already cached in abliterated/attributes.npz -- facenet
embeddings and full cortical predictions for the same 2,175 images -- so this
is CPU-only and takes seconds. No model is loaded.

Usage:
  python scripts/likeness_gradient.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR
from measure_identity_signal import build_masks

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR", "STS"]


def unit_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else 0.0


def spearman(x, y):
    from scipy.stats import rankdata
    return pearson(rankdata(x), rankdata(y))


def residualize(y, C):
    """Remove the part of y linearly predictable from columns of C."""
    coef, *_ = np.linalg.lstsq(C, y, rcond=None)
    return y - C @ coef


def fisher_ci(r, n, z=1.96):
    if abs(r) >= 1 or n < 4:
        return (r, r)
    a = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(a - z * se)), float(np.tanh(a + z * se)))


def main():
    d = np.load(OUT_DIR / "attributes.npz", allow_pickle=True)
    F = d["facenet"].astype(np.float64)
    P = d["preds"].astype(np.float64)
    A = d["attrs"].astype(np.float64)
    is_t = d["is_target"].astype(bool)
    det = d["detected"].astype(bool)
    attr_names = list(d["attr_names"])

    masks = build_masks()
    rois = [r for r in ROIS if r in masks]

    keep = det & np.isfinite(F).all(1) & np.isfinite(A).all(1)
    print(f"{keep.sum()} usable images of {len(keep)} "
          f"({(keep & is_t).sum()} target, {(keep & ~is_t).sum()} general)\n")

    Fn = unit_rows(F)
    centroid = Fn[keep & is_t].mean(0)
    centroid /= max(np.linalg.norm(centroid), 1e-9)
    sim_all = Fn @ centroid

    resp = {r: P[:, masks[r]].mean(1) for r in rois}

    g = keep & ~is_t
    t = keep & is_t
    sim_g = sim_all[g]
    n_g = int(g.sum())

    print("Target vs general, and how far the general population reaches "
          "toward her")
    print(f"  facenet similarity to her centroid: target {sim_all[t].mean():+.3f}, "
          f"general {sim_g.mean():+.3f} (max general {sim_g.max():+.3f})")
    fr = resp["FACE(OFA+FFA)"]
    print(f"  FACE response: target {fr[t].mean():+.5f}, "
          f"general {fr[g].mean():+.5f}\n")

    # 1. Does response track likeness among general faces only?
    C = np.column_stack([np.ones(n_g), (A[g] - A[g].mean(0)) /
                         np.maximum(A[g].std(0), 1e-9)])
    sim_r = residualize(sim_g.copy(), C)

    print("Correlation with facenet similarity to target, GENERAL FACES ONLY")
    print(f"  n={n_g}  'partial' = after regressing out all "
          f"{len(attr_names)} photometric/geometric attributes")
    print(f"{'ROI':>14s} {'pearson':>9s} {'spearman':>9s} {'partial':>9s} "
          f"{'95% CI (partial)':>22s}")
    print("-" * 68)
    partials = {}
    for r in rois:
        y = resp[r][g]
        rp = pearson(sim_g, y)
        rs = spearman(sim_g, y)
        pr = pearson(sim_r, residualize(y.copy(), C))
        lo, hi = fisher_ci(pr, n_g)
        partials[r] = (pr, lo, hi)
        star = "*" if lo * hi > 0 else " "
        print(f"{r:>14s} {rp:+9.3f} {rs:+9.3f} {pr:+9.3f}{star} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>21s}")

    # 2. Decile profile -- a monotone rise is the thing worth suppressing.
    print("\nFACE response by similarity decile, general faces only")
    order = np.argsort(sim_g)
    fg = resp["FACE(OFA+FFA)"][g]
    print(f"{'decile':>7s} {'sim range':>18s} {'n':>5s} {'FACE resp':>11s}")
    print("-" * 45)
    for i in range(10):
        idx = order[int(i * n_g / 10):int((i + 1) * n_g / 10)]
        print(f"{i + 1:7d} {f'{sim_g[idx].min():+.3f}..{sim_g[idx].max():+.3f}':>18s} "
              f"{len(idx):5d} {fg[idx].mean():+11.5f}")

    top = order[int(0.9 * n_g):]
    bot = order[:int(0.1 * n_g)]
    gap = fg[top].mean() - fg[bot].mean()
    tgt_gap = fr[t].mean() - fg.mean()
    print(f"\n  top decile - bottom decile = {gap:+.5f}")
    print(f"  target - general mean       = {tgt_gap:+.5f}")
    if abs(tgt_gap) > 1e-12:
        print(f"  the likeness gradient spans {100 * gap / tgt_gap:.0f}% of the "
              f"target's own elevation")

    # 3. If likeness does not explain her elevation, does photometry? Fit
    #    response from the 21 attributes on GENERAL faces only, then ask what
    #    that model predicts for her photos. Whatever it predicts is the part
    #    of her elevation attributable to how the pictures were taken rather
    #    than to who is in them.
    print("\nIs the target's own elevation explained by photometry alone?")
    print("  (attribute model fit on general faces only, applied to hers)")
    mu, sd = A[g].mean(0), np.maximum(A[g].std(0), 1e-9)
    Cg = np.column_stack([np.ones(int(g.sum())), (A[g] - mu) / sd])
    Ct = np.column_stack([np.ones(int(t.sum())), (A[t] - mu) / sd])
    print(f"{'ROI':>14s} {'actual':>10s} {'predicted':>10s} {'residual':>10s} "
          f"{'explained':>10s}")
    print("-" * 58)
    for r in rois:
        coef, *_ = np.linalg.lstsq(Cg, resp[r][g], rcond=None)
        gap = resp[r][t].mean() - resp[r][g].mean()
        pred_gap = float((Ct @ coef).mean() - (Cg @ coef).mean())
        resid_gap = gap - pred_gap
        frac = (100 * pred_gap / gap) if abs(gap) > 1e-12 else 0.0
        print(f"{r:>14s} {gap:+10.5f} {pred_gap:+10.5f} {resid_gap:+10.5f} "
              f"{frac:9.0f}%")
    print("  'actual' is her mean minus the general mean; 'predicted' is what "
          "the\n  photometric model expects from her pictures' properties alone.")

    # 4. Verdict
    face_pr, face_lo, face_hi = partials["FACE(OFA+FFA)"]
    ctrl = [abs(partials[r][0]) for r in ("V1", "AUD", "MOTOR") if r in partials]
    max_ctrl = max(ctrl) if ctrl else 0.0
    print("\n" + "=" * 68)
    sig = face_lo * face_hi > 0
    if sig and face_pr > 0 and abs(face_pr) > 2 * max_ctrl:
        print("VERDICT: graded likeness effect, selective to face cortex.")
        print("  Response rises with similarity to her among faces that are not")
        print("  her, it survives the photometric controls, and it is not")
        print("  mirrored in control ROIs. Suppression by similarity is well")
        print("  posed -- fit the readout residual against this gradient.")
    elif sig and face_pr > 0:
        print("VERDICT: gradient present but NOT face-selective.")
        print(f"  FACE partial r={face_pr:+.3f} against max control "
              f"|r|={max_ctrl:.3f}. Suppressing this would drag control ROIs")
        print("  along with it. Needs an anchor penalty on the controls.")
    else:
        print("VERDICT: no graded likeness effect.")
        print("  The elevation does not extend to faces that merely resemble")
        print("  her, so it is a property of the 175 photos rather than of her")
        print("  appearance. Suppressing it would target provenance.")
    print("=" * 68)


if __name__ == "__main__":
    main()

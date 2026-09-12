"""
attribute_matched_control.py

Settles the disagreement between two earlier answers about why TRIBE's face
cortex reads high for the target.

  likeness_gradient.py  regressed FACE response on 21 photometric/geometric
      attributes, fit on general faces, and found the model predicts 112% of
      her elevation -- i.e. photography explains all of it.

  photometric_null_test.py  actually rewrote her photographs to match general
      luminance, contrast, sharpness and saturation, and recovered only 24% of
      the gap.

Both cannot be right. The likely culprit is extrapolation: her attributes sit
at z = +3.3 (sharpness) and +2.2 (eye edge density) against the general
distribution, and a linear model fit near z = 0 is being asked to predict far
outside its support. If so the regression is untrustworthy and the intervention
is the honest number.

This script tests that directly, without transforming any pixels. For each
target photo it finds the general faces nearest to it in standardised attribute
space and compares her response against those specific controls rather than
against the general average. Matched controls answer "compared with ordinary
photographs that look like hers photographically, is she still high?".

It also reports common support, which is the part that decides whether any of
this is answerable. If no general photograph comes close to hers, then every
method here -- regression, normalisation, matching -- is extrapolating, and the
correct conclusion is that the comparison cannot be made with this photo set.

Usage:
  python scripts/attribute_matched_control.py --k 25
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR
from face_attributes import ATTR_NAMES
from measure_identity_signal import build_masks

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR", "STS"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=25,
                    help="Nearest general faces to match to each target photo.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(OUT_DIR / "attributes.npz", allow_pickle=True)
    A = d["attrs"].astype(np.float64)
    P = d["preds"].astype(np.float64)
    is_t = d["is_target"].astype(bool)
    det = d["detected"].astype(bool)
    masks = build_masks()
    rois = [r for r in ROIS if r in masks]

    g, t = det & ~is_t, det & is_t
    mu, sd = A[g].mean(0), np.maximum(A[g].std(0), 1e-9)
    Zg, Zt = (A[g] - mu) / sd, (A[t] - mu) / sd
    resp = {r: P[:, masks[r]].mean(1) for r in rois}

    print(f"{t.sum()} target photos, {g.sum()} general faces, "
          f"{len(ATTR_NAMES)} attributes\n")

    # ---- common support ------------------------------------------------
    print("Common support: how far outside the general population is she?")
    print(f"{'attribute':>20s} {'z(target)':>10s} {'general faces beyond her':>26s}")
    print("-" * 60)
    zt_mean = Zt.mean(0)
    order = np.argsort(-np.abs(zt_mean))
    for i in order[:8]:
        if zt_mean[i] >= 0:
            frac = float((Zg[:, i] >= zt_mean[i]).mean())
        else:
            frac = float((Zg[:, i] <= zt_mean[i]).mean())
        print(f"{ATTR_NAMES[i]:>20s} {zt_mean[i]:+10.2f} "
              f"{100 * frac:24.2f}%")

    # ---- nearest-neighbour matching ------------------------------------
    # Euclidean in z space over all attributes; each target photo pulls its own
    # controls, so matching respects the spread within her set rather than
    # matching one averaged prototype.
    D = np.linalg.norm(Zt[:, None, :] - Zg[None, :, :], axis=2)
    nn = np.argsort(D, axis=1)[:, :args.k]
    nn_d = np.take_along_axis(D, nn, axis=1)

    typ = np.linalg.norm(Zg[:, None, :] - Zg[None, :, :], axis=2)
    np.fill_diagonal(typ, np.inf)
    typical = float(np.sort(typ, axis=1)[:, :args.k].mean())

    print(f"\nMatching quality (Euclidean distance in {len(ATTR_NAMES)}-d z space)")
    print(f"  target photo -> its {args.k} nearest general faces: "
          f"mean {nn_d.mean():.2f}, best {nn_d.min():.2f}")
    print(f"  general face -> its {args.k} nearest general faces: "
          f"mean {typical:.2f}   (the yardstick)")
    ratio = nn_d.mean() / max(typical, 1e-9)
    print(f"  ratio {ratio:.2f}x -- "
          + ("matches are as close as ordinary neighbours, comparison is sound"
             if ratio < 1.5 else
             "her nearest matches are far worse than ordinary neighbours, so "
             "matching is still extrapolating"))

    # ---- matched comparison --------------------------------------------
    print(f"\nResponse against attribute-matched controls (k={args.k})")
    print(f"{'ROI':>14s} {'target':>10s} {'matched':>10s} {'all general':>12s} "
          f"{'gap vs matched':>15s} {'gap vs all':>11s}")
    print("-" * 78)
    for r in rois:
        yt = resp[r][t]
        yg = resp[r][g]
        matched = yg[nn].mean(1)          # per target photo, its controls' mean
        gap_m = float((yt - matched).mean())
        gap_a = float(yt.mean() - yg.mean())
        print(f"{r:>14s} {yt.mean():+10.5f} {matched.mean():+10.5f} "
              f"{yg.mean():+12.5f} {gap_m:+15.5f} {gap_a:+11.5f}")

    # paired significance on the face ROI
    yt = resp["FACE(OFA+FFA)"][t]
    matched = resp["FACE(OFA+FFA)"][g][nn].mean(1)
    diff = yt - matched
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
    gap_all = float(yt.mean() - resp["FACE(OFA+FFA)"][g].mean())
    print(f"\nFACE gap vs matched controls: {diff.mean():+.5f} "
          f"[95% CI {lo:+.5f}, {hi:+.5f}]")
    print(f"FACE gap vs all general:      {gap_all:+.5f}")
    if abs(gap_all) > 1e-9:
        print(f"matching removes {100 * (gap_all - diff.mean()) / gap_all:.0f}% "
              f"of the raw gap")

    print("\n" + "=" * 78)
    if ratio >= 1.5:
        print("VERDICT: cannot be answered with this photo set.")
        print("  Her photographs have no close counterparts in the general")
        print("  population, so regression, normalisation and matching are all")
        print("  extrapolating. The 112% and the 24% are both artefacts of that.")
        print("  What is needed is target images shot like ordinary snapshots,")
        print("  not more analysis of these 175.")
    elif lo * hi <= 0:
        print("VERDICT: no elevation once photography is matched.")
        print("  Against general faces photographed like hers, her response is")
        print("  indistinguishable. Nothing identity-specific to suppress.")
    else:
        print("VERDICT: a real residual survives matching.")
        print(f"  {diff.mean():+.5f}, CI excluding zero, against comparable")
        print("  photographs. That residual is the honest abliteration target.")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""
part_occlusion_report.py

Reads part_occlusion.npy and answers the two questions the raw table cannot:

  Are the face ROIs doing anything V1 is not? The part-profile correlation
  between FACE/OFA and V1 across the seven parts. A high value means the face
  pathway ranks parts the same way primary visual cortex does, i.e. it is
  tracking low-level image change rather than face structure.

  Is any part genuinely target-SPECIFIC? A Welch t on target-vs-general for the
  same part, so a difference has to survive both groups' spread rather than
  just look large. At n=17/14 several "differences" did not.
"""

import sys
from pathlib import Path
import numpy as np

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR", "STS"]
def parts_in(rows):
    """Derived from the data, not hardcoded: runs restricted with --only-parts
    (e.g. the forehead sub-band follow-up) measure a different set."""
    return sorted({k.split(":", 1)[1] for _, k, _, _ in rows if k.startswith("part:")})


def adjusted(rows, sel, PARTS):
    """Per-part residuals after regressing the random controls' effect on area,
    fitted within the same group."""
    ctrl = [(a, d) for t, k, a, d in rows if k.startswith("ctrl") and sel(t)]
    out = {}
    for roi in ROIS:
        A = np.array([[a, 1.0] for a, _ in ctrl])
        y = np.array([d[roi] for _, d in ctrl])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        for p in PARTS:
            v = [(a, d[roi]) for t, k, a, d in rows if k == f"part:{p}" and sel(t)]
            out.setdefault(p, {})[roi] = np.array([x - (coef[0] * a + coef[1]) for a, x in v])
    return out


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "abliterated/part_occlusion.npy")
    rows = np.load(path, allow_pickle=True)
    PARTS0 = parts_in(rows)
    key0 = f"part:{PARTS0[0]}"
    nt = sum(1 for r in rows if r[0] and r[1] == key0)
    ng = sum(1 for r in rows if not r[0] and r[1] == key0)
    print(f"images used -> target {nt}, general {ng}\n")

    PARTS = parts_in(rows)
    T = adjusted(rows, lambda t: t, PARTS)
    G = adjusted(rows, lambda t: not t, PARTS)

    if len(PARTS) >= 5:
        print("Part-profile correlation with V1:")
    for lab, D in [("TARGET ", T), ("GENERAL", G)]:
        # a correlation over 3 points is not a profile
        if len(PARTS) < 5:
            continue
        prof = {r: np.array([D[p][r].mean() for p in PARTS]) for r in ROIS}
        print(f"  {lab}  FACE~V1 {np.corrcoef(prof['FACE(OFA+FFA)'], prof['V1'])[0,1]:+.3f}"
              f"   OFA~V1 {np.corrcoef(prof['OFA'], prof['V1'])[0,1]:+.3f}"
              f"   FFA~V1 {np.corrcoef(prof['FFA'], prof['V1'])[0,1]:+.3f}")

    print("\nTarget vs general, same part (Welch t; * = |t| > 2):")
    for roi in ["FACE(OFA+FFA)", "OFA", "FFA"]:
        print(f"\n  {roi}")
        print(f"    {'part':10s} {'target':>10s} {'general':>10s} {'diff':>10s} {'t':>7s}")
        for p in PARTS:
            a, b = T[p][roi], G[p][roi]
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            t = (a.mean() - b.mean()) / max(se, 1e-12)
            print(f"    {p:10s} {a.mean():+10.5f} {b.mean():+10.5f} "
                  f"{a.mean()-b.mean():+10.5f} {t:+7.2f}{'*' if abs(t) > 2 else ' '}")




    # Within-group band comparison. For the forehead sub-bands this is the
    # comparison that is actually clean: the three target bands cover similar
    # image fractions (3.1 / 3.7 / 4.0%), whereas target-vs-general does not
    # (target forehead_high is 3.1% of frame, general 0.5%), and no area
    # regression should be asked to extrapolate across a 6x gap.
    if "forehead_high" in PARTS and "forehead_low" in PARTS:
        print("\n\nWithin-group: hairline band (high) vs skin band (low)")
        print(f"  {'group':8s} {'ROI':6s} {'high':>10s} {'low':>10s} {'diff':>10s} {'t':>7s}")
        for lab, D in [("target", T), ("general", G)]:
            for roi in ["FACE(OFA+FFA)", "OFA", "FFA"]:
                hi, lo = D["forehead_high"][roi], D["forehead_low"][roi]
                n = min(len(hi), len(lo))
                d = hi[:n] - lo[:n]          # paired: same images
                t = d.mean() / max(d.std(ddof=1) / np.sqrt(n), 1e-12)
                print(f"  {lab:8s} {roi.split('(')[0]:6s} {hi.mean():+10.5f} "
                      f"{lo.mean():+10.5f} {d.mean():+10.5f} {t:+7.2f}"
                      f"{'*' if abs(t) > 2 else ' '}")


if __name__ == "__main__":
    main()


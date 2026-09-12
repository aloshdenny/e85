"""
dosage_report.py

Three questions the per-level tables cannot answer on their own.

  Is this localised? 17,000 of 20,484 vertices pass FDR at n=1000 pairs. With
  that much power a trivially small effect clears the bar, so "significant"
  stops being informative and the honest measure is where the effect is
  LARGEST, and how face regions rank against everything else.

  Does it dilute? An identity signal competing with N other faces should shrink
  roughly as 1/(N+1). A fixed image-level difference between two tiles should
  not care how many other faces are present. Fitting both models says which.

  Is it one thing? If the effect map at N=1 correlates highly with N=5, the
  same spatial pattern is being driven regardless of face count.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR
from dosage_composites import destrieux_regions

d = np.load(OUT_DIR / "dosage_composites.npz")
levels = sorted({int(k.split("_")[1]) for k in d.files})
regions = destrieux_regions()
FACE = ["G_oc-temp_lat-fusifor", "G_and_S_occipital_inf", "Pole_occipital"]

print(f"levels {levels}, {len(regions)} regions\n")

print("Rank of face-selective regions among all 74, by |mean t| (1 = strongest):")
print(f"  {'region':32s}" + "".join(f"{'N=' + str(N):>8s}" for N in levels))
for nm in FACE:
    line = f"  {nm:32s}"
    for N in levels:
        t = d[f"t_{N}"]
        order = sorted(regions, key=lambda k: -abs(t[regions[k]].mean()))
        line += f"{order.index(nm) + 1:8d}"
    print(line)
for nm in ["S_central", "S_front_middle", "G_and_S_cingul-Mid-Ant"]:
    line = f"  {nm:32s}"
    for N in levels:
        t = d[f"t_{N}"]
        order = sorted(regions, key=lambda k: -abs(t[regions[k]].mean()))
        line += f"{order.index(nm) + 1:8d}"
    print(line + "   <- not visual")

print("\nSpatial correlation between effect maps across levels (mean delta per vertex):")
print(f"  {'':6s}" + "".join(f"{'N=' + str(N):>8s}" for N in levels))
for a in levels:
    row = f"  N={a:<4d}"
    for b in levels:
        row += f"{np.corrcoef(d[f'mean_{a}'], d[f'mean_{b}'])[0,1]:8.3f}"
    print(row)

print("\nDilution test on the strongest regions:")
print("  constant model vs 1/(N+1) dilution, fitted per region (lower SSE wins)")
print(f"  {'region':32s} {'const SSE':>11s} {'dilute SSE':>11s} {'verdict':>12s}")
t0 = d[f"t_{levels[0]}"]
top = sorted(regions, key=lambda k: -abs(t0[regions[k]].mean()))[:10]
inv = np.array([1.0 / (N + 1) for N in levels])
for nm in top:
    m = regions[nm]
    y = np.array([d[f"mean_{N}"][m].mean() for N in levels])
    sse_c = float(((y - y.mean()) ** 2).sum())
    A = np.stack([inv, np.ones_like(inv)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sse_d = float(((y - A @ coef) ** 2).sum())
    print(f"  {nm:32s} {sse_c:11.3e} {sse_d:11.3e} "
          f"{'dilutes' if sse_d < sse_c * 0.5 else 'flat':>12s}")

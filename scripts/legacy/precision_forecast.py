"""
precision_forecast.py

What does adding more GENERAL images actually buy, given the target set is
fixed at 175 photos (150 usable)?

A Welch t on target-vs-general has
    se = sqrt(var_target/n_target + var_general/n_general)
and n_target cannot grow -- there are only 175 photos of her. Once
var_general/n_general is small next to var_target/n_target, more general
images move nothing. This computes the real ratio from the variances actually
measured, rather than assuming them equal.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from part_occlusion_report import adjusted, PARTS

rows = np.load(sys.argv[1] if len(sys.argv) > 1 else "abliterated/part_occlusion.npy",
               allow_pickle=True)
T = adjusted(rows, lambda t: t)
G = adjusted(rows, lambda t: not t)

roi = "FFA"
part = "forehead"          # the one surviving target-specific effect
a, b = T[part][roi], G[part][roi]
nt, ng = len(a), len(b)
vt, vg = a.var(ddof=1), b.var(ddof=1)
diff = a.mean() - b.mean()

print(f"{part} -> {roi}:  target n={nt}  general n={ng}")
print(f"  diff = {diff:+.5f}   var_target = {vt:.3e}   var_general = {vg:.3e}\n")

print(f"{'n_general':>12s} {'se':>12s} {'t':>8s} {'GPU-days':>10s} {'t gain':>8s}")
print("-" * 56)
base_t = None
for n in [ng, 5000, 20000, 101698]:
    se = np.sqrt(vt / nt + vg / n)
    t = diff / se
    if base_t is None:
        base_t = t
    days = (n - ng) * 13 * 0.96 / 3600 / 24 if n > ng else 0.0
    print(f"{n:12,d} {se:12.3e} {t:8.2f} {days:10.1f} {t/base_t - 1:+7.1%}")

print(f"\nCeiling with an infinite general set (var_general/n -> 0): "
      f"t = {diff / np.sqrt(vt / nt):.2f}")
print("The target set is the binding constraint. var_target/n_target is "
      f"{100 * (vt/nt) / (vt/nt + vg/ng):.0f}% of the current error term.")

"""
token_direction_compare.py

Does face-token pooling produce a better ABLATION DIRECTION, not merely a
better identity probe?

token_pooling_probe.py found face tokens carry 4-5x more decodable identity
than background tokens (L30: 35.7% vs 8.4% retrieval), which is a real
dissociation -- but only ~2 points more than the mean over all tokens, because
the face box already covers 78.8% of the token grid on these pre-cropped
photos. There was little dilution left to remove.

So the question is whether that small representational gain survives into the
quantity that matters: a direction whose removal actually changes predicted
cortex. Same four metrics used on every previous candidate --

  AUC_ho      does it separate held-out target photos
  PCleak      how much of it lies in the general population's top-20 variance
              (high = removing it damages generic face processing)
  proj_ratio  does ablating it cost the target more than the general population
  r_facenet   do held-out general faces that project high on it actually look
              more like the target to facenet -- the test every contrast-derived
              direction failed at r = -0.06..+0.08

Usage:
  python scripts/token_direction_compare.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR
from analyze_directions import auc, unit, lda_direction
from identity_direction import fit_ridge

d = np.load(OUT_DIR / "token_pooled.npz")
a = np.load(OUT_DIR / "attributes.npz", allow_pickle=True)
F = a["facenet"].astype(np.float64)
n_target = int(d["n_target"])
det = d["detected"]
layers = sorted({int(k.split("_L")[1]) for k in d.files if k.startswith("mean_all_L")})

rng = np.random.default_rng(0)
ti = rng.permutation(n_target)
n_gen = len(det) - n_target
gi = rng.permutation(n_gen)
nt_te, ng_te = int(n_target * 0.3), int(n_gen * 0.3)
t_te, t_tr = ti[:nt_te], ti[nt_te:]
g_te, g_tr = gi[:ng_te], gi[ng_te:]
det_t, det_g = det[:n_target], det[n_target:]
Ft, Fg = F[:n_target], F[n_target:]

a_t = unit(Ft[t_tr][det_t[t_tr]].mean(0) - Fg[g_tr][det_g[g_tr]].mean(0))
fn_sim = (Fg / np.maximum(np.linalg.norm(Fg, axis=1, keepdims=True), 1e-12)) @ \
         unit(Ft[t_tr][det_t[t_tr]].mean(0))

print(f"target {n_target} ({len(t_tr)}/{len(t_te)}), general {n_gen} "
      f"({len(g_tr)}/{len(g_te)})\n")
print(f"{'layer':>5s} {'pooling':>9s} {'method':>9s} {'AUC_ho':>7s} {'proj_ratio':>11s} "
      f"{'PCleak':>8s} {'r_facenet':>10s}")
print("-" * 66)

best = None
saved = {}
for L in layers:
    for pool in ["mean_all", "face"]:
        X = d[f"{pool}_L{L}"].astype(np.float64)
        Xt, Xg = X[:n_target], X[n_target:]
        fit_rows = g_tr[det_g[g_tr]]
        W, mu, sd, _ = fit_ridge(Xg[fit_rows], Fg[fit_rows], 0.1)
        q_fn = unit((W @ a_t) / sd)
        q_dm = unit(Xt[t_tr].mean(0) - Xg[g_tr].mean(0))
        Vg = np.linalg.svd(Xg[g_tr] - Xg[g_tr].mean(0), full_matrices=False)[2][:20]
        ho = g_te[det_g[g_te]]
        for meth, q in [("diffmean", q_dm), ("facenet", q_fn)]:
            pt, pg = Xt[t_te] @ q, Xg[g_te] @ q
            ratio = float(np.abs(pt).mean()) / max(float(np.abs(pg).mean()), 1e-12)
            leak = float(np.sum((Vg @ q) ** 2))
            r = float(np.corrcoef(Xg[ho] @ q, fn_sim[ho])[0, 1])
            A = auc(pt, pg)
            print(f"{L:5d} {pool:>9s} {meth:>9s} {A:7.3f} {ratio:11.2f} {leak:8.3f} {r:+10.3f}")
            if pool == "face":
                saved[f"{meth}_L{L}"] = q.astype(np.float32)[None, :]
            if meth == "facenet" and (best is None or r > best[0]):
                best = (r, L, pool, q.astype(np.float32))
    print()

r, L, pool, q = best
np.savez(OUT_DIR / "token_directions.npz", **saved)
print(f"best by r_facenet: L{L} {pool}  r={r:+.3f}")
print(f"saved {len(saved)} face-pooled directions -> {OUT_DIR/'token_directions.npz'}")

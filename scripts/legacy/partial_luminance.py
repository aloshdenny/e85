"""
partial_luminance.py

Two questions about the full-scale feature -> cortex map, both answered from
files already on disk.

1. Did 50x more data change the map? Vertex-level agreement between the
   n=1,959 version and the n=99,408 one. If they agree, that is direct
   confirmation of the precision argument -- the map was converged long before
   the full set, and the extra images bought nothing.

2. Is ANY of the face-ROI signal face-STRUCTURE rather than brightness?
   Every large correlation in that map is brightness-family (luminance +0.52/
   +0.65, skin_L, forehead_luma, hair_darkness, eye_edge), while geometry sits
   near zero (yaw +0.008, roll +0.049, face_cx -0.026). So partial out
   luminance and see what survives:

       r_jv.L = (r_jv - r_jL r_Lv) / sqrt((1 - r_jL^2)(1 - r_Lv^2))

   This needs no new I/O: r_jv is the saved correlation matrix, r_Lv its
   luminance row, and r_jL comes from the saved per-image attribute table. If
   the face ROIs are a photometric readout, the surviving partials collapse
   toward zero; if they carry face structure, something stays.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))
from measure_identity_signal import build_masks
from abliteration import OUT_DIR

ROIS = ["OFA", "FFA", "FACE(OFA+FFA)", "V1", "AUD", "MOTOR"]

d = np.load(OUT_DIR / "feature_cortex_map_full.npz", allow_pickle=True)
corr = d["corr"].astype(np.float64)              # (21, 20484)
names = [str(s) for s in d["attr_names"]]
A, det = d["attrs"].astype(np.float64), d["detected"]
masks = build_masks()
print(f"full map: n={int(d['n']):,}\n")

# ---- 1. did the extra 50x change anything? --------------------------------
old_p = OUT_DIR / "attribute_cortical_maps.npz"
if old_p.exists():
    old = np.load(old_p, allow_pickle=True)
    common = [nm for nm in names if nm in old.files]
    A_old = np.stack([old[nm] for nm in common])
    A_new = np.stack([corr[names.index(nm)] for nm in common])
    r = float(np.corrcoef(A_old.ravel(), A_new.ravel())[0, 1])
    print(f"n=1,959 map vs n=99,408 map: vertex-level r = {r:.4f}, "
          f"max |diff| = {np.abs(A_old - A_new).max():.3f}, "
          f"mean |diff| = {np.abs(A_old - A_new).mean():.4f}")
    print("  (50x the data; the map was already converged)\n")

# ---- 2. partial out luminance ---------------------------------------------
Ad = A[det]
Ad = (Ad - Ad.mean(0)) / np.maximum(Ad.std(0), 1e-12)
r_jL = (Ad.T @ Ad[:, names.index("luminance")]) / len(Ad)      # attr vs luminance
r_Lv = corr[names.index("luminance")]                          # luminance vs vertex

print(f"{'attribute':20s} {'r(luma)':>8s}" +
      "".join(f"{x.split('(')[0]:>9s}{'':>8s}" for x in ROIS))
print(f"{'':20s} {'':>8s}" + "".join(f"{'raw':>9s}{'partial':>8s}" for _ in ROIS))
print("-" * (29 + 17 * len(ROIS)))
for j, nm in enumerate(names):
    denom = np.sqrt(max(1 - r_jL[j] ** 2, 1e-12) * np.maximum(1 - r_Lv ** 2, 1e-12))
    part = (corr[j] - r_jL[j] * r_Lv) / denom
    line = f"{nm:20s} {r_jL[j]:+8.2f}"
    for roi in ROIS:
        m = masks[roi]
        line += f"{corr[j][m].mean():+9.3f}{part[m].mean():+8.3f}"
    print(line)

print("\nr(luma) is how much each attribute is itself a brightness measurement.")
print("raw -> partial is what survives once luminance is removed.")


# ---- 3. all attributes at once --------------------------------------------
# Partialling out luminance alone is not a deconfounding: the 21 attributes are
# mutually correlated (skin_L, forehead_luma and hair_darkness are all largely
# the same brightness factor), so a pairwise partial still leaves each column
# carrying its neighbours. The exact fix, and it costs nothing more, is the
# standardised multiple regression of every vertex on ALL 21 at once:
#
#     beta_v = R_aa^-1 r_av
#
# with R_aa the attribute-attribute correlation matrix and r_av the saved
# attribute-vertex correlations. beta is then each attribute's UNIQUE
# contribution, holding all twenty others fixed.
print("\n\nStandardised multiple-regression betas (each attribute's unique share)")
R_aa = (Ad.T @ Ad) / len(Ad)
R_inv = np.linalg.pinv(R_aa + 1e-8 * np.eye(len(names)))
beta = R_inv @ corr                                   # (21, 20484)

print(f"{'attribute':20s}" + "".join(f"{r.split('(')[0]:>10s}" for r in ROIS))
print("-" * (20 + 10 * len(ROIS)))
order = np.argsort(-np.abs(np.stack([beta[j][masks['FACE(OFA+FFA)']].mean()
                                     for j in range(len(names))])))
for j in order:
    print(f"{names[j]:20s}" + "".join(f"{beta[j][masks[r]].mean():+10.3f}" for r in ROIS))

r2 = np.einsum("jv,jv->v", beta, corr)
for roi in ["OFA", "FFA", "FACE(OFA+FFA)", "V1", "MOTOR"]:
    print(f"  variance of {roi} explained by all 21 attributes: "
          f"R^2 = {r2[masks[roi]].mean():.3f}")

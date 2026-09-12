"""
ds007369_real_rsa.py

The ground-truth half of the comparison: build representational geometries from
the REAL human fMRI for the same 9 conditions TRIBE was shown, and ask three
things in order.

  1. Is the real geometry measurable at all? Split-half reliability across
     trials gives the noise ceiling. A model cannot be blamed for failing to
     predict structure the data itself does not reliably contain, and with 72
     trials per condition in a handful of subjects this has to be checked
     before anything else is interpreted.

  2. Does real face cortex carry identity structure here? Same permutation test
     used on TRIBE's RDMs, so the numbers are directly comparable.

  3. Does TRIBE's predicted geometry match the real one? Correlating the two
     RDMs is space-agnostic, which is the whole reason RSA is the right tool:
     the real ROIs are functionally defined in subject volume space and TRIBE
     outputs fsaverage5 vertices, and no correspondence between them is needed.

Beta rows align one-to-one with event rows (648 <-> 648, 646 <-> 646), so each
row is a single trial carrying a Condition label like "id2-mov3".

Usage:
  python scripts/ds007369_real_rsa.py
"""

import sys, re, itertools
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

CONDS = [f"id{i}-mov{m}" for i in (1, 2, 3) for m in (1, 2, 3)]


def rdm(P):
    Z = (P - P.mean(1, keepdims=True)) / np.maximum(P.std(1, keepdims=True), 1e-12)
    return 1 - (Z @ Z.T) / Z.shape[1]


def offdiag(M):
    return M[np.triu_indices(M.shape[0], 1)]


def model_vecs(conds):
    n = len(conds)
    ident = np.array([[0.0 if conds[i].split("-")[0] == conds[j].split("-")[0] else 1.0
                       for j in range(n)] for i in range(n)])
    motion = np.array([[0.0 if conds[i].split("-")[1] == conds[j].split("-")[1] else 1.0
                        for j in range(n)] for i in range(n)])
    return offdiag(ident), offdiag(motion)


def perm_p(D, target, rng, P=10000):
    v = offdiag(D)
    if v.std() < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(v, target)[0, 1])
    n = D.shape[0]
    hits = 0
    for _ in range(P):
        p = rng.permutation(n)
        if abs(np.corrcoef(offdiag(D[np.ix_(p, p)]), target)[0, 1]) >= abs(r):
            hits += 1
    return r, (hits + 1) / (P + 1)


def main():
    base = Path("./ds007369/osf")
    beta_files = sorted((base / "betas").glob("*.npz"))
    rng = np.random.default_rng(0)
    vi, vm = model_vecs(CONDS)

    tribe = np.load(Path("./abliterated") / "ds007369_rsa.npz", allow_pickle=True)
    tribe_conds = [str(c) for c in tribe["conds"]]
    order = [tribe_conds.index(c) for c in CONDS]

    print(f"{'subject/ROI':34s} {'vox':>5s} {'rel':>6s} {'r_id':>7s} {'p':>7s} "
          f"{'r_mot':>7s} {'p':>7s} {'vs TRIBE':>9s}")
    print("-" * 88)

    per_roi = {}
    for bf in beta_files:
        m = re.match(r"sub-(\w+)_ses-(\d)_(\w+?)_", bf.name)
        subj, ses, roi = m.group(1), m.group(2), m.group(3)
        ev = base / "events" / f"{subj}_events_concatenated_ses-{ses}.tsv"
        if not ev.exists():
            print(f"  [skip] no events for {bf.name}")
            continue
        B = np.load(bf)["betas"]
        E = pd.read_csv(ev)
        if len(E) != len(B):
            print(f"  [skip] {bf.name}: {len(B)} betas vs {len(E)} events")
            continue

        idx = {c: np.where(E.Condition.values == c)[0] for c in CONDS}
        if any(len(v) < 4 for v in idx.values()):
            print(f"  [skip] {bf.name}: a condition has too few trials")
            continue

        P = np.stack([B[idx[c]].mean(0) for c in CONDS])
        # split-half over trials -> how much geometry is reliably there at all
        A = np.stack([B[idx[c][0::2]].mean(0) for c in CONDS])
        Bh = np.stack([B[idx[c][1::2]].mean(0) for c in CONDS])
        rel = float(np.corrcoef(offdiag(rdm(A)), offdiag(rdm(Bh)))[0, 1])

        D = rdm(P)
        r_i, p_i = perm_p(D, vi, rng)
        r_m, p_m = perm_p(D, vm, rng)

        key = f"rdm_{roi}" if f"rdm_{roi}" in tribe.files else None
        if roi == "FFA":
            key = "rdm_FFA"
        elif roi == "OFA":
            key = "rdm_OFA"
        xr = float("nan")
        if key in tribe.files:
            T = tribe[key][np.ix_(order, order)]
            xr = float(np.corrcoef(offdiag(D), offdiag(T))[0, 1])

        print(f"{('sub-' + subj + ' ses-' + ses + ' ' + roi):34s} {B.shape[1]:5d} "
              f"{rel:6.3f} {r_i:+7.3f} {p_i:7.3f} {r_m:+7.3f} {p_m:7.3f} {xr:+9.3f}")
        per_roi.setdefault(roi, []).append((r_i, r_m, rel, xr))

    print("\nGroup means by ROI (real fMRI):")
    print(f"  {'ROI':6s} {'n':>3s} {'reliability':>12s} {'r_identity':>11s} "
          f"{'r_motion':>10s} {'r vs TRIBE':>11s}")
    for roi, rows in sorted(per_roi.items()):
        a = np.array(rows, dtype=float)
        print(f"  {roi:6s} {len(rows):3d} {np.nanmean(a[:,2]):12.3f} "
              f"{np.nanmean(a[:,0]):+11.3f} {np.nanmean(a[:,1]):+10.3f} "
              f"{np.nanmean(a[:,3]):+11.3f}")

    print("\nTRIBE, same stimuli (from ds007369_rsa.py):")
    for roi in ["FFA", "OFA"]:
        k = f"rdm_{roi}"
        if k in tribe.files:
            T = tribe[k][np.ix_(order, order)]
            r_i, p_i = perm_p(T, vi, rng)
            print(f"  {roi:6s} r_identity={r_i:+.3f} (p={p_i:.3f})")

    print("\nreliability is the ceiling: TRIBE cannot be expected to predict "
          "geometry\nthe measurement itself does not reproduce across trial halves.")


if __name__ == "__main__":
    main()

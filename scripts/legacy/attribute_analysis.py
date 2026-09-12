"""
attribute_analysis.py

Two questions, both answered offline from data already on disk.

1. WHICH CORTICAL REGIONS TRACK WHICH FACIAL FEATURE.
   For every named attribute, the vertex-wise correlation across the general
   population between that attribute and predicted activity, summarised per
   ROI. This is the "map features to visual regions" map.

2. HOW MUCH OF THE "MIA DIRECTION" IS JUST THOSE FEATURES.
   In vjepa2 activation space, each attribute gets a direction (ridge fit on
   GENERAL images only, so the subspace describes how faces normally vary, not
   how Mia differs). The Mia contrast is then decomposed against that subspace.

   The point of (2) is the inversion. Ablating an attribute direction would
   suppress everyone who has that attribute -- the least surgical thing
   available. What we want is the part of the Mia contrast that named,
   shared attributes CANNOT explain, so the subspace gets projected OUT:

       q_resid = normalize(q_mia - P_attr q_mia)

   and q_resid is scored on the same three numbers analyze_directions.py uses:
   held-out AUC (does it still separate Mia), top-20-general-PC leak (does it
   avoid generic face variance), and proj_ratio (does ablating it cost the
   target more than the general population).

   facenet (VGGFace2) is the independent referee. It is a real face-identity
   space that never saw a brain map, so if a direction is identity rather than
   photographic style, general faces that project high on it should also look
   more like Mia to facenet. A direction that fails that check is style.

Usage:
  python scripts/attribute_analysis.py --layers 5 10 15 20 25 30 35
"""

import sys, argparse
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR
from analyze_directions import auc, unit, lda_direction
from measure_identity_signal import build_masks


def zs(v):
    v = np.asarray(v, dtype=np.float64)
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def ridge_direction(X, t, alpha_frac=0.1):
    """Direction in X-space that linearly predicts target t. Ridge rather than a
    plain correlation vector because the activation covariance is strongly
    anisotropic: the correlation vector would point at the top PCs whatever t
    is, which is the exact failure mode being diagnosed."""
    Xc = X - X.mean(0)
    t = zs(t)
    G = Xc.T @ Xc
    alpha = alpha_frac * np.trace(G) / G.shape[0]
    w = np.linalg.solve(G + alpha * np.eye(G.shape[0]), Xc.T @ t)
    return unit(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", type=Path, default=OUT_DIR / "attributes.npz")
    ap.add_argument("--cache-dir", type=Path, default=OUT_DIR / "raw_activations_norm")
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35])
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-attrs", type=int, default=6)
    args = ap.parse_args()

    d = np.load(args.attrs, allow_pickle=True)
    A, names = d["attrs"].astype(np.float64), [str(s) for s in d["attr_names"]]
    P, is_t, det = d["preds"].astype(np.float64), d["is_target"], d["detected"]
    F = d["facenet"].astype(np.float64)
    print(f"{A.shape[0]} rows, {A.shape[1]} attributes, {int(is_t.sum())} target, "
          f"face detected in {det.mean():.1%}")

    keep = det                      # geometry columns are meaningless without a detection
    masks = build_masks()
    rois = ["OFA", "FFA", "FACE(OFA+FFA)", "V1", "AUD", "MOTOR", "STS"]

    # ---------------------------------------------------------------- part 1
    print("\n" + "=" * 92)
    print("WHICH REGIONS TRACK WHICH FEATURE  (corr across general population, "
          "mean over ROI vertices)")
    print("=" * 92)
    gen = (~is_t) & keep
    Pg = P[gen]
    print(f"{'attribute':20s}" + "".join(f"{r:>12s}" for r in rois))
    print("-" * 92)
    attr_roi = {}
    for j, nm in enumerate(names):
        a = zs(A[gen, j])
        if a.std() < 1e-9:
            continue
        # vertex-wise correlation, vectorised
        Pz = (Pg - Pg.mean(0)) / np.maximum(Pg.std(0), 1e-12)
        cmap = (Pz * a[:, None]).mean(0)
        attr_roi[nm] = cmap
        print(f"{nm:20s}" + "".join(f"{cmap[masks[r]].mean():+12.3f}" for r in rois))

    # ---------------------------------------------------------------- part 2
    print("\n" + "=" * 92)
    print("HOW MUCH OF THE MIA DIRECTION IS NAMED, SHARED FEATURES")
    print("=" * 92)

    rng = np.random.default_rng(args.seed)
    n_target = int(is_t.sum())
    ti = rng.permutation(n_target)
    gi = rng.permutation(int((~is_t).sum()))
    nt_te, ng_te = int(n_target * args.holdout), int(len(gi) * args.holdout)

    rows = []
    for L in args.layers:
        xp = args.cache_dir / f"raw_X_L{L}.npy"
        if not xp.exists():
            continue
        X = np.load(xp).astype(np.float64)
        Xt_all, Xg_all = X[:n_target], X[n_target:]
        Xt_te, Xt_tr = Xt_all[ti[:nt_te]], Xt_all[ti[nt_te:]]
        Xg_te, Xg_tr = Xg_all[gi[:ng_te]], Xg_all[gi[ng_te:]]

        # attribute subspace, fitted on GENERAL TRAIN rows only
        gen_tr_mask = np.zeros(len(Xg_all), dtype=bool)
        gen_tr_mask[gi[ng_te:]] = True
        usable = gen_tr_mask & det[n_target:]
        Xg_fit, A_fit = Xg_all[usable], A[n_target:][usable]

        dirs, used = [], []
        for j, nm in enumerate(names):
            if A_fit[:, j].std() < 1e-9:
                continue
            dirs.append(ridge_direction(Xg_fit, A_fit[:, j]))
            used.append(nm)
        D = np.stack(dirs)
        # orthonormal basis of the attribute subspace
        U, S, _ = np.linalg.svd(D.T, full_matrices=False)
        B = U[:, S > 1e-8 * S[0]].T

        Vg = np.linalg.svd(Xg_tr - Xg_tr.mean(0), full_matrices=False)[2][:20]

        q_dm = unit(Xt_tr.mean(0) - Xg_tr.mean(0))
        q_lda = lda_direction(Xt_tr, Xg_tr, 0.1)
        explained = float(np.sum((B @ q_dm) ** 2))
        q_res = unit(q_dm - B.T @ (B @ q_dm))

        print(f"\nL{L}:  attribute subspace rank {B.shape[0]} / {len(used)} attributes")
        print(f"  fraction of the diff-of-means Mia direction explained by named "
              f"attributes: {explained:.1%}")
        order = np.argsort(-np.abs(D @ q_dm))[: args.top_attrs]
        print("  biggest named contributors: " +
              ", ".join(f"{used[k]} {float(D[k] @ q_dm):+.2f}" for k in order))

        for lab, q in [("diffmean", q_dm), ("lda", q_lda), ("attr_resid", q_res)]:
            a_ho = auc(Xt_te @ q, Xg_te @ q)
            et = float(np.abs(Xt_te @ q).mean())
            eg = float(np.abs(Xg_te @ q).mean())
            leak = float(np.sum((Vg @ q) ** 2))

            # facenet referee, GENERAL images only: do faces that project high
            # on this vjepa2 direction also look more like Mia to facenet?
            fm = unit(F[is_t].mean(0))
            Fg = F[n_target:]
            fn_sim = (Fg / np.maximum(np.linalg.norm(Fg, axis=1, keepdims=True), 1e-12)) @ fm
            r_fn = float(np.corrcoef(Xg_all @ q, fn_sim)[0, 1])

            print(f"    {lab:11s} AUC_ho={a_ho:.3f}  proj_ratio={et / max(eg, 1e-12):5.2f}  "
                  f"PCleak={leak:.3f}  r_facenet={r_fn:+.3f}")
            rows.append(dict(layer=L, dir=lab, auc=a_ho, ratio=et / max(eg, 1e-12),
                             leak=leak, r_facenet=r_fn, explained=explained))
        del X

    print("\n" + "=" * 92)
    print(f"{'layer':>5s} {'dir':12s} {'AUC_ho':>7s} {'proj_ratio':>11s} {'PCleak':>8s} "
          f"{'r_facenet':>10s}")
    print("=" * 92)
    for r in sorted(rows, key=lambda r: -r["ratio"])[:24]:
        print(f"{r['layer']:5d} {r['dir']:12s} {r['auc']:7.3f} {r['ratio']:11.2f} "
              f"{r['leak']:8.3f} {r['r_facenet']:+10.3f}")

    np.save(OUT_DIR / "attribute_analysis.npy", np.array(rows, dtype=object),
            allow_pickle=True)
    np.savez_compressed(OUT_DIR / "attribute_cortical_maps.npz", **attr_roi)
    print(f"\nSaved -> {OUT_DIR/'attribute_analysis.npy'} and "
          f"{OUT_DIR/'attribute_cortical_maps.npz'}")


if __name__ == "__main__":
    main()

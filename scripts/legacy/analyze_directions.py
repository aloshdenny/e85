"""
analyze_directions.py

Pure-numpy post-mortem on the CACHED layer activations -- no GPU, no
predict(), seconds per layer. It exists because ab_surgery_modes.py showed the
surgery moving target and general face responses by nearly identical amounts in
every mode and at every depth: the problem is upstream of the surgery, in the
direction itself.

The surgery removes the component of every token along q. So the quantity that
mechanically determines selectivity is not "does q separate target from
general" (a CENTERED question) but "how much of each group's activation energy
actually SITS on q" (an UNCENTERED one). If general faces project onto q about
as strongly as the target does, removing q must damage both -- exactly what was
measured. That is `proj_ratio` below, and unlike every proxy this project has
discarded, it is tied to the arithmetic of the edit rather than to a hoped-for
correlate of it.

Directions compared:
  diffmean  normalize(mean(X_t) - mean(X_g))          <- current primary
  lda       normalize((Sigma_g + lam I)^-1 (mu_t - mu_g))
            Mahalanobis discriminant. diffmean is only optimal when the noise
            is isotropic; a ViT residual stream is the opposite, so diffmean
            loads heavily on whatever the top shared PCs are -- i.e. on
            generic face variance, which is precisely what must be preserved.
  wpca1     top weighted-PCA component (the secondary-direction machinery)

Usage:
  python scripts/analyze_directions.py --layers 0 5 10 15 20 25 30 35 39
"""

import sys, argparse
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import (
    build_face_mask, load_target_images, find_directions, OUT_DIR,
)


def auc(pos, neg):
    """Rank-based AUC: P(a random target scores above a random general)."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def lda_direction(Xt, Xg, shrink=0.1):
    """(Sigma + lam I)^-1 (mu_t - mu_g), with Sigma pooled over BOTH groups and
    lam set as a fraction of its mean eigenvalue (Ledoit-Wolf style shrinkage
    toward the identity, which is what makes this stable at d=1408 with only a
    few thousand samples)."""
    mu_t, mu_g = Xt.mean(0), Xg.mean(0)
    Xc = np.concatenate([Xt - mu_t, Xg - mu_g], 0)
    S = (Xc.T @ Xc) / max(len(Xc) - 2, 1)
    lam = shrink * np.trace(S) / S.shape[0]
    return unit(np.linalg.solve(S + lam * np.eye(S.shape[0]), mu_t - mu_g))


def report(label, q, Xt_tr, Xg_tr, Xt_te, Xg_te, Vg):
    pt_tr, pg_tr = Xt_tr @ q, Xg_tr @ q
    pt_te, pg_te = Xt_te @ q, Xg_te @ q

    # Centered separability (what "does it discriminate" usually means).
    a_tr, a_te = auc(pt_tr, pg_tr), auc(pt_te, pg_te)

    # Uncentered energy on q -- what the surgery actually deletes.
    et, eg = float(np.abs(pt_te).mean()), float(np.abs(pg_te).mean())
    proj_ratio = et / max(eg, 1e-12)

    # How much of q lies in the general population's dominant variance
    # subspace, i.e. how much generic face structure the edit destroys.
    leak = float(np.sum((Vg @ q) ** 2))

    print(f"  {label:9s} AUC_train={a_tr:.3f} AUC_heldout={a_te:.3f}  "
          f"|proj| target={et:7.3f} general={eg:7.3f}  proj_ratio={proj_ratio:5.2f}  "
          f"top20PC_leak={leak:.3f}")
    return dict(label=label, auc_train=a_tr, auc_heldout=a_te,
                proj_target=et, proj_general=eg, proj_ratio=proj_ratio, pc_leak=leak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 5, 10, 15, 20, 25, 30, 35, 39])
    ap.add_argument("--cache-dir", type=Path, default=OUT_DIR / "raw_activations_norm")
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mask = build_face_mask(include_secondary=False)
    n_target = len(load_target_images(args.target_preds_npz, args.target_zip, mask))
    print(f"n_target={n_target}\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    for l in args.layers:
        xp = args.cache_dir / f"raw_X_L{l}.npy"
        if not xp.exists():
            print(f"L{l}: no cache, skipping")
            continue
        X = np.load(xp).astype(np.float64)
        y = np.load(args.cache_dir / f"raw_y_L{l}.npy").astype(np.float64)
        Xt, Xg = X[:n_target], X[n_target:]

        ti, gi = rng.permutation(len(Xt)), rng.permutation(len(Xg))
        nt_te, ng_te = int(len(Xt) * args.holdout), int(len(Xg) * args.holdout)
        Xt_te, Xt_tr = Xt[ti[:nt_te]], Xt[ti[nt_te:]]
        Xg_te, Xg_tr = Xg[gi[:ng_te]], Xg[gi[ng_te:]]

        # Dominant variance subspace of the GENERAL population = the structure
        # that must survive. Computed on train only.
        Gc = Xg_tr - Xg_tr.mean(0)
        _, _, Vt_g = np.linalg.svd(Gc, full_matrices=False)
        Vg = Vt_g[:20]

        print(f"L{l}:")
        q_dm = unit(Xt_tr.mean(0) - Xg_tr.mean(0))
        q_lda = lda_direction(Xt_tr, Xg_tr, args.shrink)
        Xtr = np.concatenate([Xt_tr, Xg_tr])
        ytr = np.concatenate([y[:n_target][ti[nt_te:]], y[n_target:][gi[ng_te:]]])
        q_w = unit(find_directions(Xtr, ytr, 1, f"L{l}-wpca")[0])

        for lab, q in [("diffmean", q_dm), ("lda", q_lda), ("wpca1", q_w)]:
            r = report(lab, q, Xt_tr, Xg_tr, Xt_te, Xg_te, Vg)
            r["layer"] = l
            rows.append(r)
        print(f"  cos(diffmean, lda)={float(q_dm @ q_lda):+.3f}   "
              f"cos(diffmean, wpca1)={float(q_dm @ q_w):+.3f}")
        print()

    print("=" * 100)
    print("Ranked by proj_ratio (higher = removing q costs the target more than "
          "the general population)")
    print("=" * 100)
    print(f"{'layer':>5s} {'dir':9s} {'AUC_ho':>7s} {'proj_t':>8s} {'proj_g':>8s} "
          f"{'ratio':>6s} {'PCleak':>7s}")
    for r in sorted(rows, key=lambda r: -r["proj_ratio"])[:25]:
        print(f"{r['layer']:5d} {r['label']:9s} {r['auc_heldout']:7.3f} "
              f"{r['proj_target']:8.3f} {r['proj_general']:8.3f} "
              f"{r['proj_ratio']:6.2f} {r['pc_leak']:7.3f}")

    out = OUT_DIR / "direction_analysis.npy"
    np.save(out, np.array(rows, dtype=object), allow_pickle=True)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

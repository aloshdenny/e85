"""
identity_direction.py

Derives the ablation direction from the part of activation space that provably
encodes IDENTITY, instead of from a raw target-vs-general contrast.

The two measurements this is built on:

  probe_identity_content.py  the pooled activations DO carry identity. A ridge
      fit from activations to facenet's VGGFace2 embedding retrieves the right
      face out of a ~391-image held-out fold 28.1% of the time at L35, against
      0.26% chance, and far above the 0.9% you get from 21 photometric scalars.

  attribute_analysis.py      yet every direction taken from mean(X_t)-mean(X_g)
      correlates with facenet Mia-similarity at r = -0.03..+0.07, including the
      LDA one that separates held-out Mia photos at AUC 1.000.

Both being true means the identity subspace is present but the target-vs-general
contrast does not point along it: the contrast is dominated by the target set's
photographic provenance, which is perfectly confounded with the target because
all 175 photos come from one narrow source. Sharper statistics cannot fix a
confound this complete -- but they can be routed around.

The route: learn activations -> facenet with a ridge fit on GENERAL images ONLY,
so the map never sees the target and cannot absorb its provenance. Take the
target axis in FACENET space, where it means "looks like this person" and is
trained to be invariant to lighting, pose and camera. Pull that axis back
through the map:

    q = normalize(W @ a_target)

q is by construction inside the identity-encoding subspace, and its definition
of "target" was supplied by a network that has never seen a brain map or this
photo set's provenance.

Evaluated the same way as every other candidate, plus a generalisation check the
contrast-based directions failed: on HELD-OUT general faces, do the ones that
project high on q also look more like the target to facenet? W was fit without
those rows, so that correlation is a real test rather than a restatement.

Usage:
  python scripts/identity_direction.py --layers 15 25 30 35 39
"""

import sys, argparse
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR
from analyze_directions import auc, unit, lda_direction


def fit_ridge(X, Y, alpha_frac=0.1):
    """Returns (W, mu, sd, ymu) such that pred = ((X - mu)/sd) @ W + ymu."""
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
    Xs = (X - mu) / sd
    ymu = Y.mean(0)
    G = Xs.T @ Xs
    alpha = alpha_frac * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + alpha * np.eye(G.shape[0]), Xs.T @ (Y - ymu))
    return W, mu, sd, ymu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", type=Path, default=OUT_DIR / "attributes.npz")
    ap.add_argument("--cache-dir", type=Path, default=OUT_DIR / "raw_activations_norm")
    ap.add_argument("--layers", type=int, nargs="+", default=[15, 25, 30, 35, 39])
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--alpha-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dirs", type=Path, default=OUT_DIR / "identity_directions.npz")
    args = ap.parse_args()

    d = np.load(args.attrs, allow_pickle=True)
    F = d["facenet"].astype(np.float64)
    is_t, det = d["is_target"], d["detected"]
    n_target = int(is_t.sum())

    rng = np.random.default_rng(args.seed)
    ti = rng.permutation(n_target)
    n_gen = len(is_t) - n_target
    gi = rng.permutation(n_gen)
    nt_te, ng_te = int(n_target * args.holdout), int(n_gen * args.holdout)
    t_te, t_tr = ti[:nt_te], ti[nt_te:]
    g_te, g_tr = gi[:ng_te], gi[ng_te:]

    det_t, det_g = det[:n_target], det[n_target:]
    Ft, Fg = F[:n_target], F[n_target:]

    # target axis in facenet space, from TRAIN rows only
    a_t = unit(Ft[t_tr][det_t[t_tr]].mean(0) - Fg[g_tr][det_g[g_tr]].mean(0))
    fn_unit = Fg / np.maximum(np.linalg.norm(Fg, axis=1, keepdims=True), 1e-12)
    fn_sim = fn_unit @ unit(Ft[t_tr][det_t[t_tr]].mean(0))

    print(f"target {n_target} ({len(t_tr)} train / {len(t_te)} held out), "
          f"general {n_gen} ({len(g_tr)} / {len(g_te)})\n")
    print(f"{'layer':>5s} {'dir':10s} {'AUC_ho':>7s} {'proj_ratio':>11s} {'PCleak':>8s} "
          f"{'r_facenet_ho':>13s}")
    print("-" * 62)

    saved = {}
    for L in args.layers:
        xp = args.cache_dir / f"raw_X_L{L}.npy"
        if not xp.exists():
            continue
        X = np.load(xp).astype(np.float64)
        Xt, Xg = X[:n_target], X[n_target:]

        # activations -> facenet, GENERAL TRAIN rows only: the map never sees
        # the target, so it cannot learn the target set's provenance.
        fit_rows = g_tr[det_g[g_tr]]
        W, mu, sd, _ = fit_ridge(Xg[fit_rows], Fg[fit_rows], args.alpha_frac)

        # pull the facenet target axis back into activation space
        q_fn = unit((W @ a_t) / sd)

        q_dm = unit(Xt[t_tr].mean(0) - Xg[g_tr].mean(0))
        q_lda = lda_direction(Xt[t_tr], Xg[g_tr], 0.1)

        Vg = np.linalg.svd(Xg[g_tr] - Xg[g_tr].mean(0), full_matrices=False)[2][:20]
        ho_rows = g_te[det_g[g_te]]

        for lab, q in [("diffmean", q_dm), ("lda", q_lda), ("facenet", q_fn)]:
            pt, pg = Xt[t_te] @ q, Xg[g_te] @ q
            a_ho = auc(pt, pg)
            ratio = float(np.abs(pt).mean()) / max(float(np.abs(pg).mean()), 1e-12)
            leak = float(np.sum((Vg @ q) ** 2))
            r_ho = float(np.corrcoef(Xg[ho_rows] @ q, fn_sim[ho_rows])[0, 1])
            print(f"{L:5d} {lab:10s} {a_ho:7.3f} {ratio:11.2f} {leak:8.3f} {r_ho:+13.3f}")
            if lab == "facenet":
                saved[f"L{L}"] = q.astype(np.float32)
        print()
        del X

    if saved:
        np.savez(args.save_dirs, **saved)
        print(f"Saved facenet-derived directions -> {args.save_dirs}")
    print("r_facenet_ho is the test the contrast-based directions failed: held-out "
          "general faces\nprojecting high on q should look more like the target to "
          "facenet. Rows the ridge never saw.")


if __name__ == "__main__":
    main()

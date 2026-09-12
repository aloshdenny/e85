"""
probe_identity_content.py

Asks whether the activations we have been searching for an "identity direction"
in contain identity information at all.

Why. attribute_analysis.py's facenet referee came back at r = -0.03..+0.07 for
EVERY candidate direction, including the LDA one that separates Mia's held-out
photos from held-out general photos at AUC 1.000. Perfect separation with no
identity correlation means the thing being separated is not identity -- most
likely the target set's own photographic provenance, which is perfectly
confounded with the target's identity because all 175 photos come from the same
narrow source. No direction-finding method can undo that from this data.

So: a direct probe. Ridge from the cached activations to facenet's (VGGFace2)
512-d identity embedding, cross-validated on GENERAL images only, scored by
held-out cosine and by identity retrieval (does the predicted embedding pick the
right face out of the held-out fold). Two reference points make the number
readable:

  attrs->facenet   the same probe from only the 21 hand-made photometric and
                   geometric scalars. Whatever identity they explain is
                   identity you could get from brightness and crop alone.
  acts->attrs      the reverse check, that the activations are informative at
                   all. If they predict brightness well but identity badly, the
                   pooled representation kept image statistics and discarded
                   the face.

That last contrast is the one that matters, because collect_layer_activations
stores hidden.mean(dim=1) -- the mean over all ~8k tokens. Mean-pooling a ViT is
exactly the operation that would keep global image statistics and throw identity
away, and every direction searched so far has lived in that pooled space.

Usage:
  python scripts/probe_identity_content.py --layers 5 15 25 35
"""

import sys, argparse
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR


def ridge_cv(X, Y, n_folds=5, alpha_frac=0.1, seed=0):
    """Returns (mean held-out cosine between predicted and true rows,
    top-1 retrieval accuracy within each held-out fold)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, n_folds)
    cos, hits, total = [], 0, 0
    for f in folds:
        te = np.zeros(len(X), dtype=bool)
        te[f] = True
        Xtr, Xte, Ytr, Yte = X[~te], X[te], Y[~te], Y[te]
        mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-9)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ymu = Ytr.mean(0)
        G = Xtr.T @ Xtr
        alpha = alpha_frac * np.trace(G) / G.shape[0]
        W = np.linalg.solve(G + alpha * np.eye(G.shape[0]), Xtr.T @ (Ytr - ymu))
        pred = Xte @ W + ymu

        pn = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        tn = Yte / np.maximum(np.linalg.norm(Yte, axis=1, keepdims=True), 1e-12)
        cos.append(float((pn * tn).sum(1).mean()))
        # retrieval: nearest true embedding in this fold for each prediction
        sim = pn @ tn.T
        hits += int((sim.argmax(1) == np.arange(len(sim))).sum())
        total += len(sim)
    return float(np.mean(cos)), hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", type=Path, default=OUT_DIR / "attributes.npz")
    ap.add_argument("--cache-dir", type=Path, default=OUT_DIR / "raw_activations_norm")
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 15, 25, 35])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.attrs, allow_pickle=True)
    A = d["attrs"].astype(np.float64)
    F = d["facenet"].astype(np.float64)
    is_t, det = d["is_target"], d["detected"]
    names = [str(s) for s in d["attr_names"]]

    gen = (~is_t) & det
    n_target = int(is_t.sum())
    Ag, Fg = A[gen], F[gen]
    Ag = (Ag - Ag.mean(0)) / np.maximum(Ag.std(0), 1e-9)
    print(f"general images with a detected face: {gen.sum()}")

    c, r = ridge_cv(Ag, Fg, args.folds)
    print(f"\nreference  21 photometric/geometric attrs -> facenet identity: "
          f"cos={c:+.3f}  retrieval={r:.1%}")
    print(f"chance retrieval within a fold of ~{int(gen.sum()/args.folds)}: "
          f"{args.folds / gen.sum():.2%}\n")

    print(f"{'layer':>5s} {'acts->facenet cos':>18s} {'retrieval':>10s} "
          f"{'acts->attrs cos':>16s}")
    print("-" * 54)
    for L in args.layers:
        xp = args.cache_dir / f"raw_X_L{L}.npy"
        if not xp.exists():
            print(f"L{L}: no cache")
            continue
        X = np.load(xp).astype(np.float64)
        Xg = X[n_target:][det[n_target:]]
        if len(Xg) != len(Fg):
            Xg = X[n_target:][(det[n_target:])][: len(Fg)]
        c_id, r_id = ridge_cv(Xg, Fg, args.folds)
        c_at, _ = ridge_cv(Xg, Ag, args.folds)
        print(f"{L:5d} {c_id:18.3f} {r_id:10.1%} {c_at:16.3f}")
        del X

    print("\ncos ~0 with chance-level retrieval means the pooled activations carry "
          "no linearly decodable identity, whatever they do carry.")


if __name__ == "__main__":
    main()

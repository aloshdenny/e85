"""
ab_surgery_modes.py

One question, measured end-to-end rather than argued about: does the surgery
BASIS fix change real selectivity?

`q` is extracted by hooking VJEPA2Layer's output, so it lives in the residual
stream basis. abliteration.py used to remove it from attention.proj's INPUT
space -- but proj is nn.Linear(all_head_size, hidden) whose input is the
concatenated-head basis, not the residual basis. That edit is an arbitrary
rank-1 perturbation, which would show up as exactly the symptom this project
kept hitting: general faces dropping as much as the target.

Modes compared (see abliteration.SURGERY_MODES):
  legacy  value(read) + proj(read)     <- what produced the 1.75x result
  both    value(read) + proj(write) + mlp.fc2(write)
  write   proj(write) + mlp.fc2(write)
  read    value(read)

Scoring reuses the same real TribeModel.predict() path the greedy search uses,
on the same val images, so numbers here are directly comparable to a search
round. Per-image deltas are printed too -- with 6 target / 8 general images the
spread matters as much as the mean, and the aggregate ratio hides it.

Usage:
  python scripts/ab_surgery_modes.py --layers 5 15 25 35 --modes legacy both
"""

import os, sys, argparse, tempfile, shutil
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))

import abliteration as abl
from abliteration import (
    build_face_mask, load_target_images, collect_layer_activations,
    find_directions_for_method, apply_surgery, make_trial_checkpoint,
    assert_redirectable_path, pick_trial_scratch_root,
    MODEL_FAMILY_TOKEN, SURGERY_MODES, OUT_DIR, free, TribeModel,
)
from validation import (
    get_tmp_root, discover_val_images, build_clips_once,
    run_predict_on_clips, redirect_model_name,
)


def per_image_deltas(preds_before, preds_after, paths, mask):
    out = {}
    for p in paths:
        n = p.name
        if n in preds_before and n in preds_after:
            out[n] = float(preds_after[n][mask].mean()) - float(preds_before[n][mask].mean())
    return out


def cohens_d(target_deltas, general_deltas):
    """Positive = target dropped MORE than general, scaled by within-group
    noise. Unlike |target|/|general| this cannot be inflated by a general_delta
    that lands near zero by chance, and it respects the sign of the effect
    (suppression is the goal; amplification should not score well)."""
    t, g = np.array(list(target_deltas.values())), np.array(list(general_deltas.values()))
    if len(t) < 2 or len(g) < 2:
        return float("nan")
    nt, ng = len(t), len(g)
    pooled = np.sqrt(((nt - 1) * t.var(ddof=1) + (ng - 1) * g.var(ddof=1)) / (nt + ng - 2))
    if pooled < 1e-12:
        return float("nan")
    return float((g.mean() - t.mean()) / pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 15, 25, 35])
    ap.add_argument("--modes", nargs="+", default=["legacy", "both"], choices=sorted(SURGERY_MODES))
    ap.add_argument("--n_components", type=int, default=5)
    ap.add_argument("--directions", nargs="+", default=["contrastive"],
                    help="Direction methods to compare (contrastive/lda/wpca).")
    ap.add_argument("--lda-shrink", type=float, default=0.1)
    ap.add_argument("--tolerances", type=float, nargs="+", default=[-1.0],
                    help="Over-subtraction sweep. A direction orthogonal to general face "
                         "variance perturbs the representation only weakly at -1, so "
                         "stronger values buy magnitude at (hopefully) little collateral.")
    ap.add_argument("--val-dir", type=Path, default=Path("./val"))
    ap.add_argument("--target-prefix", default="mia")
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--source-model-name", default="facebook/vjepa2-vitg-fpc64-256")
    ap.add_argument("--max-shard-size", default="2GB")
    ap.add_argument("--trial-scratch-dir", type=Path, default=None)
    ap.add_argument("--combo", action="store_true",
                    help="Also run all --layers together as one combined set.")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    mask = build_face_mask(include_secondary=False)
    target_paths, general_paths = discover_val_images(args.val_dir, args.target_prefix)
    print(f"val: {len(target_paths)} target, {len(general_paths)} general")

    scratch_root = pick_trial_scratch_root(OUT_DIR, args.trial_scratch_dir)
    assert_redirectable_path(scratch_root / f"{MODEL_FAMILY_TOKEN}_ab_probe", "trial dir")
    print(f"trial checkpoints -> {scratch_root}")

    print("\nLoading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    vjepa2_module = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    vjepa2_module.eval().to(abl.DEVICE)

    # n_target defines the target/general split point in the cached activation
    # matrices -- they were written in that order by collect_layer_activations.
    n_target = len(load_target_images(args.target_preds_npz, args.target_zip, mask))

    cache_dir = OUT_DIR / "raw_activations_norm"
    dirs_by_method = {m: {} for m in args.directions}
    for l in args.layers:
        if not (cache_dir / f"raw_X_L{l}.npy").exists():
            raise FileNotFoundError(f"No cached activations for L{l} in {cache_dir}")
        X, y = collect_layer_activations(vjepa2_module, encoder_blocks, l, [], [],
                                         cache_dir=cache_dir)
        if X.shape[0] <= n_target:
            raise ValueError(f"L{l}: cached X has {X.shape[0]} rows but n_target={n_target}")
        for m in args.directions:
            dirs_by_method[m][l] = find_directions_for_method(
                m, X, y, n_target, args.n_components, f"L{l}-{m}", args.lda_shrink)
        del X, y
        free()

    original_state = {k: v.clone() for k, v in vjepa2_module.state_dict().items()}
    original_snapshot_dir = Path(snapshot_download(repo_id=args.source_model_name))
    tmp_root = get_tmp_root()
    all_paths = target_paths + general_paths

    print("\nBaseline (pre-surgery) predictions, reused by every trial...")
    d = Path(tempfile.mkdtemp(prefix="ab_base_", dir=tmp_root))
    try:
        rows = build_clips_once(all_paths, d, duration=1.0, fps=2)
        m0 = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
        preds_before = run_predict_on_clips(m0, rows, duration=1.0)
        del m0
        free()
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # Scale context. A delta of -0.002 means nothing without knowing what the
    # face-mask response IS at baseline, and how far above the general
    # population the target already sits -- "completely forgotten" should mean
    # the target's face-mask response falls to the general level, not merely
    # that it moved.
    base_t = np.array([float(preds_before[p.name][mask].mean())
                       for p in target_paths if p.name in preds_before])
    base_g = np.array([float(preds_before[p.name][mask].mean())
                       for p in general_paths if p.name in preds_before])
    baseline_scale = float(np.abs(np.concatenate([base_t, base_g])).mean())
    identity_gap = float(base_t.mean() - base_g.mean())
    print(f"\nBaseline face-mask response: target mean={base_t.mean():+.5f} "
          f"general mean={base_g.mean():+.5f}")
    print(f"  identity gap (target - general) = {identity_gap:+.5f}")
    print(f"  |response| scale = {baseline_scale:.5f}  -- deltas below are also "
          f"reported as a fraction of the identity gap, which is what a complete "
          f"erasure would have to close.")

    results = []
    seen = []
    tag = 0

    def trial(layers, mode, method, tol):
        nonlocal tag
        tag += 1
        dirs_by_layer = dirs_by_method[method]
        vjepa2_module.load_state_dict(original_state)
        apply_surgery(encoder_blocks, {l: dirs_by_layer[l] for l in layers},
                      tol, mode=mode, verbose=False)

        tdir = scratch_root / f"{MODEL_FAMILY_TOKEN}_ab_{tag}_{os.getpid()}"
        try:
            make_trial_checkpoint(vjepa2_module, args.source_model_name, tdir,
                                  original_snapshot_dir, max_shard_size=args.max_shard_size)
            cm = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
            redirect_model_name(cm, assert_redirectable_path(tdir, "trial dir"))
            td = Path(tempfile.mkdtemp(prefix=f"ab_{tag}_", dir=tmp_root))
            try:
                rows = build_clips_once(all_paths, td, duration=1.0, fps=2)
                preds_after = run_predict_on_clips(cm, rows, duration=1.0)
            finally:
                shutil.rmtree(td, ignore_errors=True)
            del cm
            free()
        finally:
            shutil.rmtree(tdir, ignore_errors=True)

        # Same staleness invariants the search enforces: a reused/stale model
        # returns well-formed predictions for the WRONG weights.
        for name, ref in [("baseline", preds_before)] + seen:
            common = set(preds_after) & set(ref)
            if common and all(np.array_equal(preds_after[k], ref[k]) for k in common):
                raise RuntimeError(f"trial {tag} ({method}/{mode}/tol={tol}, {layers}) is "
                                   f"bit-identical to {name} -- stale model or cached preds.")
        seen.append((f"trial {tag}", preds_after))

        tds = per_image_deltas(preds_before, preds_after, target_paths, mask)
        gds = per_image_deltas(preds_before, preds_after, general_paths, mask)
        t_mean, g_mean = float(np.mean(list(tds.values()))), float(np.mean(list(gds.values())))
        ratio = abs(t_mean) / max(abs(g_mean), 1e-9)
        d_stat = cohens_d(tds, gds)

        print(f"\n  {method:11s} {mode:7s} tol={tol:<6g} L{layers}")
        print(f"    target : mean={t_mean:+.6f}  " +
              " ".join(f"{v:+.5f}" for v in tds.values()))
        print(f"    general: mean={g_mean:+.6f}  " +
              " ".join(f"{v:+.5f}" for v in gds.values()))
        # Fraction of the identity gap closed: the target's excess over the
        # general population is what identity-specific erasure has to remove,
        # so (target_delta - general_delta) / gap is the progress measure.
        closed = (t_mean - g_mean) / identity_gap if abs(identity_gap) > 1e-12 else float("nan")
        print(f"    ratio={ratio:.2f}x   cohens_d={d_stat:+.2f}   "
              f"gap_closed={closed:+.1%}   suppressive={'YES' if t_mean < 0 else 'NO'}")
        results.append(dict(mode=mode, method=method, tol=tol, layers=list(layers),
                            target=t_mean, general=g_mean, ratio=ratio, cohens_d=d_stat,
                            gap_closed=closed))

    layer_sets = [[l] for l in args.layers]
    if args.combo:
        layer_sets.append(list(args.layers))

    for layers in layer_sets:
        for method in args.directions:
            for tol in args.tolerances:
                for mode in args.modes:
                    trial(layers, mode, method, tol)

    print("\n" + "=" * 78)
    print(f"{'method':11s} {'tol':>6s} {'layers':14s} {'target':>10s} "
          f"{'general':>10s} {'ratio':>7s} {'d':>7s} {'gap':>8s}")
    print("=" * 78)
    for r in sorted(results, key=lambda r: -(r["gap_closed"] if np.isfinite(r["gap_closed"]) else -9)):
        print(f"{r['method']:11s} {r['tol']:6g} {str(r['layers']):14s} "
              f"{r['target']:+10.6f} {r['general']:+10.6f} {r['ratio']:7.2f} "
              f"{r['cohens_d']:+7.2f} {r['gap_closed']:+8.1%}")

    out = OUT_DIR / "ab_surgery_modes.npy"
    np.save(out, np.array(results, dtype=object), allow_pickle=True)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

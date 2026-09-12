"""
eval_erasure.py

End-to-end erasure test on an evaluation set that is actually representative.

Why not reuse ./val: it holds 6 target and 8 general images, and its baseline
face-mask response runs target 0.0819 < general 0.0872 -- the OPPOSITE ordering
to the real data, where over 175 target and 1200 general predictions it is
target 0.0902 > general 0.0673. Every greedy round so far was scored on six
images pointing the wrong way. Images here are drawn from the same zips the
predictions came from, and the general ones come from npz files NOT in the
training sample, so they are genuinely held out.

Scored on the measure that was shown to be face-specific rather than confound-
saturated. Target-vs-general AUC of preds[FACE].mean() is 0.771, while the
controls sit at or below chance (V1 0.446, AUD 0.440, MOTOR 0.281). The
gain-free multivariate pattern is NOT usable for this: it reads 0.93 in face
ROIs but also 0.87-0.89 in V1 and motor cortex.

So success is: FACE AUC falls toward 0.50 (the target stops being distinguishable
from other faces in face cortex) while the control ROI AUCs and the general
population's own response level stay where they were.

Usage:
  python scripts/eval_erasure.py --direction facenet --layers 25 30 35 \
      --tolerances -1 -5 --combo
"""

import os, sys, argparse, random, zipfile, tempfile, shutil
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
import abliteration as abl
from abliteration import (
    build_face_mask, collect_layer_activations, find_directions_for_method,
    apply_surgery, make_trial_checkpoint, assert_redirectable_path,
    pick_trial_scratch_root, decode_image_from_zip, load_target_images,
    MODEL_FAMILY_TOKEN, OUT_DIR, free, TribeModel,
)
from chunk_utils import (
    discover_npz, load_npz, preds_as_image_vectors, npz_image_names, ensure_fused_zip,
)
from validation import (
    get_tmp_root, build_clips_once, run_predict_on_clips, redirect_model_name,
)
from measure_identity_signal import build_masks, auc

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR"]


def build_eval_set(target_zip, target_preds_npz, general_preds_dir, general_zip,
                   n_target, n_general, train_seed, train_size, eval_seed, out_dir):
    """Target images from the target zip; general images from npz files the
    training sample did not touch, so the general side is genuinely held out."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(eval_seed)

    tzip = ensure_fused_zip(target_zip)
    tnames = npz_image_names(load_npz(target_preds_npz))
    chosen_t = rng.sample(list(tnames), min(n_target, len(tnames)))
    tpaths = []
    with zipfile.ZipFile(tzip, "r") as zf:
        for i, nm in enumerate(chosen_t):
            img = decode_image_from_zip(zf, nm)
            p = out_dir / f"tgt_{i:03d}.jpg"
            cv2.imwrite(str(p), img)
            tpaths.append(p)

    all_npz = discover_npz(general_preds_dir)
    used = set(random.Random(train_seed).sample(all_npz, min(train_size, len(all_npz))))
    pool = [f for f in all_npz if f not in used]
    gzip_path = ensure_fused_zip(general_zip)
    gpaths = []
    with zipfile.ZipFile(gzip_path, "r") as zf:
        for f in rng.sample(pool, min(n_general * 2, len(pool))):
            if len(gpaths) >= n_general:
                break
            try:
                for nm in npz_image_names(load_npz(f)):
                    if len(gpaths) >= n_general:
                        break
                    img = decode_image_from_zip(zf, nm)
                    p = out_dir / f"gen_{len(gpaths):03d}.jpg"
                    cv2.imwrite(str(p), img)
                    gpaths.append(p)
            except Exception as e:
                print(f"  [WARN] {f.name}: {e}")
    print(f"eval set: {len(tpaths)} target, {len(gpaths)} general "
          f"(general drawn from {len(pool)} npz files outside the training sample)")
    return tpaths, gpaths


def score(preds, tpaths, gpaths, masks):
    out = {}
    for roi in ROIS:
        m = masks[roi]
        t = np.array([preds[p.name][m].mean() for p in tpaths if p.name in preds])
        g = np.array([preds[p.name][m].mean() for p in gpaths if p.name in preds])
        out[roi] = dict(auc=auc(t, g), t=float(t.mean()), g=float(g.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="facenet")
    ap.add_argument("--direction-file", type=Path, default=None,
                    help="npz of precomputed directions keyed {method}_L{layer}.")
    ap.add_argument("--layers", type=int, nargs="+", default=[25, 30, 35])
    ap.add_argument("--tolerances", type=float, nargs="+", default=[-1.0, -5.0])
    ap.add_argument("--n_components", type=int, default=1)
    ap.add_argument("--combo", action="store_true")
    ap.add_argument("--surgery-mode", default="both")
    ap.add_argument("--n-target", type=int, default=20)
    ap.add_argument("--n-general", type=int, default=20)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path, default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--source-model-name", default="facebook/vjepa2-vitg-fpc64-256")
    ap.add_argument("--max-shard-size", default="2GB")
    ap.add_argument("--trial-scratch-dir", type=Path, default=None)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    masks = build_masks()
    face_mask = build_face_mask(include_secondary=False)
    tmp_root = get_tmp_root()
    img_dir = Path(tempfile.mkdtemp(prefix="evalimgs_", dir=tmp_root))
    scratch_root = pick_trial_scratch_root(OUT_DIR, args.trial_scratch_dir)
    assert_redirectable_path(scratch_root / f"{MODEL_FAMILY_TOKEN}_eval_probe", "trial dir")

    try:
        tpaths, gpaths = build_eval_set(
            args.target_zip, args.target_preds_npz, args.general_preds_dir,
            args.general_zip, args.n_target, args.n_general, args.train_seed,
            args.train_size, args.eval_seed, img_dir)
        all_paths = tpaths + gpaths

        print("\nLoading TribeModel...")
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
        vjepa2 = model.data.video_feature.image.model.model
        blocks = vjepa2.encoder.layer
        vjepa2.eval().to(abl.DEVICE)
        n_target_train = len(load_target_images(args.target_preds_npz, args.target_zip, face_mask))

        dirs_by_layer = {}
        if args.direction_file:
            # Directions computed elsewhere -- e.g. from face-token pooling, which
            # is not what raw_activations_norm holds (that cache is the mean over
            # ALL tokens). Keyed "{method}_L{layer}".
            src = np.load(args.direction_file)
            for L in args.layers:
                key = f"{args.direction}_L{L}"
                if key not in src.files:
                    raise SystemExit(f"{key} not in {args.direction_file} "
                                     f"(has {sorted(src.files)})")
                dirs_by_layer[L] = src[key]
                print(f"  loaded {key} {src[key].shape} from {args.direction_file.name}")
        else:
            cache_dir = OUT_DIR / "raw_activations_norm"
            for L in args.layers:
                X, y = collect_layer_activations(vjepa2, blocks, L, [], [],
                                                 cache_dir=cache_dir)
                dirs_by_layer[L] = find_directions_for_method(
                    args.direction, X, y, n_target_train, args.n_components, f"L{L}")
                del X, y
                free()

        original_state = {k: v.clone() for k, v in vjepa2.state_dict().items()}
        snap = Path(snapshot_download(repo_id=args.source_model_name))

        print("\nBaseline predictions...")
        d = Path(tempfile.mkdtemp(prefix="eval_base_", dir=tmp_root))
        try:
            rows = build_clips_once(all_paths, d, duration=1.0, fps=2)
            m0 = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
            preds_before = run_predict_on_clips(m0, rows, duration=1.0)
            del m0
            free()
        finally:
            shutil.rmtree(d, ignore_errors=True)

        s0 = score(preds_before, tpaths, gpaths, masks)
        print("\nBASELINE   " + "  ".join(f"{r}:AUC={s0[r]['auc']:.3f}" for r in ROIS))
        print(f"  FACE target mean={s0['FACE(OFA+FFA)']['t']:+.5f}  "
              f"general mean={s0['FACE(OFA+FFA)']['g']:+.5f}")

        results, seen, tag = [], [], 0
        layer_sets = [[L] for L in args.layers] + ([list(args.layers)] if args.combo else [])

        for layers in layer_sets:
            for tol in args.tolerances:
                tag += 1
                vjepa2.load_state_dict(original_state)
                apply_surgery(blocks, {L: dirs_by_layer[L] for L in layers}, tol,
                              mode=args.surgery_mode, verbose=False)
                tdir = scratch_root / f"{MODEL_FAMILY_TOKEN}_eval_{tag}_{os.getpid()}"
                try:
                    make_trial_checkpoint(vjepa2, args.source_model_name, tdir, snap,
                                          max_shard_size=args.max_shard_size)
                    cm = TribeModel.from_pretrained("facebook/tribev2",
                                                    cache_folder=args.cache_folder)
                    redirect_model_name(cm, assert_redirectable_path(tdir, "trial dir"))
                    td = Path(tempfile.mkdtemp(prefix=f"eval_{tag}_", dir=tmp_root))
                    try:
                        rows = build_clips_once(all_paths, td, duration=1.0, fps=2)
                        preds_after = run_predict_on_clips(cm, rows, duration=1.0)
                    finally:
                        shutil.rmtree(td, ignore_errors=True)
                    del cm
                    free()
                finally:
                    shutil.rmtree(tdir, ignore_errors=True)

                for nm, ref in [("baseline", preds_before)] + seen:
                    common = set(preds_after) & set(ref)
                    if common and all(np.array_equal(preds_after[k], ref[k]) for k in common):
                        raise RuntimeError(f"trial {tag} bit-identical to {nm} -- stale model.")
                seen.append((f"trial {tag}", preds_after))

                s1 = score(preds_after, tpaths, gpaths, masks)
                f0, f1 = s0["FACE(OFA+FFA)"], s1["FACE(OFA+FFA)"]
                # progress toward indistinguishability, 100% = AUC reached 0.50
                closed = (f0["auc"] - f1["auc"]) / max(f0["auc"] - 0.5, 1e-9)
                gen_shift = f1["g"] - f0["g"]
                ctrl = max(abs(s1[r]["auc"] - s0[r]["auc"]) for r in ["V1", "AUD", "MOTOR"])

                print(f"\n  L{layers} tol={tol:g}")
                print("    " + "  ".join(
                    f"{r.split('(')[0]}:{s0[r]['auc']:.3f}->{s1[r]['auc']:.3f}" for r in ROIS))
                print(f"    FACE AUC {f0['auc']:.3f} -> {f1['auc']:.3f}  "
                      f"({closed:+.0%} of the way to 0.50)")
                print(f"    collateral: general FACE response {gen_shift:+.5f}, "
                      f"largest control AUC shift {ctrl:.3f}")
                results.append(dict(layers=list(layers), tol=tol, auc_before=f0["auc"],
                                    auc_after=f1["auc"], closed=closed,
                                    gen_shift=gen_shift, ctrl=ctrl))

        print("\n" + "=" * 84)
        print(f"{'layers':16s} {'tol':>6s} {'FACE AUC':>16s} {'closed':>8s} "
              f"{'gen shift':>11s} {'ctrl':>7s}")
        print("=" * 84)
        for r in sorted(results, key=lambda r: -r["closed"]):
            print(f"{str(r['layers']):16s} {r['tol']:6g} "
                  f"{r['auc_before']:.3f}->{r['auc_after']:.3f}   {r['closed']:+8.0%} "
                  f"{r['gen_shift']:+11.5f} {r['ctrl']:7.3f}")
        np.save(OUT_DIR / "eval_erasure.npy", np.array(results, dtype=object),
                allow_pickle=True)
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

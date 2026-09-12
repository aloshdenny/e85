"""
ds007369_rsa.py

Does TRIBE v2 represent face IDENTITY at all, tested against a stimulus set
built for exactly that question?

ds007369 (Hosseini & Soto) presents 3 synthetic face identities x 3 motion
patterns, orthogonally crossed, rendered from 3D models. Identity varies while
lighting, camera, background, render pipeline and crop are identical by
construction -- the minimal pair the target photo set could never provide, and
the confound that has dominated every result so far is simply absent. The
stimuli are also real 1.33s videos, so TRIBE receives its native input rather
than a still held as a fake clip.

Representational similarity analysis, which is space-agnostic and therefore
needs no fsaverage5 correspondence: build a 9x9 dissimilarity matrix from
TRIBE's predicted responses, and ask how much of its structure is explained by

  identity model   pairs sharing an identity are similar
  motion model     pairs sharing a motion are similar

A face pathway that codes identity should show identity structure in OFA/FFA
and little in V1/MOTOR. If instead identity structure is absent everywhere, or
present equally in motor cortex, that is the same null this project has hit six
times -- but established against a stimulus set where no provenance confound
exists to blame, and with human fMRI available for the same 9 conditions.

Usage:
  python scripts/ds007369_rsa.py
"""

import sys, argparse, tempfile, shutil, itertools
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from abliteration import TribeModel, free
from infer_fairface_bulk import get_tmp_root, make_multi_row_df, group_preds_by_timeline
from measure_identity_signal import build_masks

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR", "STS"]


def transcode(src, dst, duration, fps=30):
    """Re-encode to mp4 at a fixed fps; the pipeline reads mp4 elsewhere and the
    source is MJPEG avi."""
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {src}")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    need = max(int(round(duration * fps)), len(frames))
    for i in range(need):
        vw.write(frames[i % len(frames)])
    vw.release()
    return len(frames)


def rdm(P):
    """Correlation-distance dissimilarity between condition response patterns."""
    Z = (P - P.mean(1, keepdims=True)) / np.maximum(P.std(1, keepdims=True), 1e-12)
    C = (Z @ Z.T) / Z.shape[1]
    return 1 - C


def model_rdms(conds):
    n = len(conds)
    ident = np.zeros((n, n))
    motion = np.zeros((n, n))
    for i, j in itertools.product(range(n), repeat=2):
        ident[i, j] = 0.0 if conds[i][0] == conds[j][0] else 1.0
        motion[i, j] = 0.0 if conds[i][1] == conds[j][1] else 1.0
    return ident, motion


def offdiag(M):
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return M[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-dir", type=Path,
                    default=Path("./ds007369/Psychopy_Task/main/videos"))
    ap.add_argument("--fallback-dir", type=Path,
                    default=Path("./ds007369/Psychopy_Task/familiarization/videos"))
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--repeats", type=int, default=4,
                    help="Independent presentations per condition, each with its own "
                         "filepath and timeline id so the extractor cannot serve a "
                         "cached result. Averaged, to damp per-clip noise.")
    args = ap.parse_args()

    src = args.stim_dir if args.stim_dir.exists() else args.fallback_dir
    vids = sorted(p for p in src.glob("id*-mov*.avi"))
    if not vids:
        raise SystemExit(f"no id*-mov*.avi under {src}")
    conds = [(p.stem.split("-")[0], p.stem.split("-")[1]) for p in vids]
    print(f"{len(vids)} conditions from {src}:")
    print("  " + ", ".join(f"{i}/{m}" for i, m in conds))

    masks = build_masks()
    tmp_root = get_tmp_root()
    td = Path(tempfile.mkdtemp(prefix="ds7369_", dir=tmp_root))
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)

    try:
        rows = []
        for r in range(args.repeats):
            for k, v in enumerate(vids):
                tl = f"c{k}_r{r}"
                mp4 = td / f"{tl}.mp4"
                transcode(v, mp4, args.duration)
                rows.append((mp4, tl, k))
        df = make_multi_row_df([(p, tl) for p, tl, _ in rows], duration=args.duration)
        with torch.autocast("cuda", dtype=torch.float16):
            preds, segs = model.predict(events=df)
        g = group_preds_by_timeline(preds, segs)
        free()

        acc = {k: [] for k in range(len(vids))}
        for _, tl, k in rows:
            if tl in g:
                acc[k].append(np.asarray(g[tl]))
        P = np.stack([np.mean(acc[k], 0) for k in range(len(vids))])
        print(f"\npredicted responses: {P.shape}")

        ident_m, motion_m = model_rdms(conds)
        vi, vm = offdiag(ident_m), offdiag(motion_m)

        print(f"\n{'ROI':16s} {'r(identity)':>12s} {'r(motion)':>11s} "
              f"{'mean dissim':>12s}")
        print("-" * 55)
        out = {}
        for roi in ROIS:
            m = masks[roi]
            D = rdm(P[:, m])
            d = offdiag(D)
            ri = float(np.corrcoef(d, vi)[0, 1]) if d.std() > 1e-12 else float("nan")
            rm = float(np.corrcoef(d, vm)[0, 1]) if d.std() > 1e-12 else float("nan")
            print(f"{roi.split('(')[0]:16s} {ri:+12.3f} {rm:+11.3f} {d.mean():12.5f}")
            out[roi] = dict(rdm=D, r_identity=ri, r_motion=rm)

        np.savez_compressed(Path("./abliterated") / "ds007369_rsa.npz",
                            preds=P, conds=np.array([f"{i}-{m}" for i, m in conds]),
                            **{f"rdm_{k}": v["rdm"] for k, v in out.items()})
        print("\nr(identity) is how much TRIBE's predicted geometry is organised by WHO "
              "the face is.\nHigh in OFA/FFA and low in V1/MOTOR would mean the face "
              "pathway codes identity.")
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()

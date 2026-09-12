"""
nod_baseline_eval.py

Zero-shot baseline: how well does TRIBE v2 already predict real NOD face-cortex
responses to the 120 shared COCO images, before any fine-tuning?

This is the number that sizes the opportunity. If TRIBE's zero-shot prediction
already correlates strongly with the real, split-half-reliable OFA/FFA betas
(nod_reliability.py: r=0.305/0.292, well above V1/AUD/MOTOR), there's limited
headroom and fine-tuning risks overfitting 120 images. If it's flat or
negative, there's a real gap to close.

Images are held as 1s static clips through TRIBE (matches every other analysis
in this project), predictions averaged over the same face-mask ROIs, and
correlated against the split-half-averaged real betas per vertex.

Usage:
  python scripts/nod_baseline_eval.py
"""

import sys, glob, tempfile, shutil
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from abliteration import TribeModel, free
from infer_fairface_bulk import get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline
from measure_identity_signal import build_masks

NOD_DIR = Path.home() / "nod"


def load_real_betas():
    files = sorted((NOD_DIR / "betas").glob("*/ses-coco*_fsaverage5.npy"))
    labfiles = sorted((NOD_DIR / "betas").glob("*/ses-coco*_labels.txt"))
    by_img = {}
    for bf, lf in zip(files, labfiles):
        B = np.load(bf)
        labs = [l for l in lf.read_text().split("\n") if l]
        for i, lab in enumerate(labs):
            by_img.setdefault(lab, []).append(B[i])
    return {k: np.mean(v, axis=0) for k, v in by_img.items() if len(v) >= 2}


def main():
    real = load_real_betas()
    print(f"{len(real)} images with averaged real betas")

    img_dir = NOD_DIR / "stimuli" / "coco"
    have = [k for k in real if (img_dir / k).exists()]
    print(f"{len(have)} have a downloaded stimulus image")

    masks = build_masks()
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=Path("./cache"))
    tmp_root = get_tmp_root()
    td = Path(tempfile.mkdtemp(prefix="nod_eval_", dir=tmp_root))

    try:
        rows = []
        for i, k in enumerate(have):
            img = cv2.imread(str(img_dir / k))
            if img is None:
                continue
            tl = f"i{i}"
            p = td / f"{tl}.mp4"
            write_static_clip(img, p, duration=1.0, fps=2)
            rows.append((p, tl, k))
        df = make_multi_row_df([(p, tl) for p, tl, _ in rows], duration=1.0)
        with torch.autocast("cuda", dtype=torch.float16):
            preds, segs = model.predict(events=df)
        g = group_preds_by_timeline(preds, segs)
        free()

        pred_by_img = {k: np.asarray(g[tl]) for _, tl, k in rows if tl in g}
        common = [k for k in pred_by_img if k in real]
        print(f"{len(common)} images with both real and predicted responses")

        P = np.stack([pred_by_img[k] for k in common])
        R = np.stack([real[k] for k in common])

        print(f"\n{'ROI':16s} {'r (image-level)':>16s} {'r (per-vertex, mean)':>22s}")
        for roi in ["OFA", "FFA", "FACE(OFA+FFA)", "V1", "AUD", "MOTOR"]:
            m = masks[roi]
            img_r = float(np.corrcoef(P[:, m].mean(1), R[:, m].mean(1))[0, 1])
            vtx_rs = [np.corrcoef(P[:, v], R[:, v])[0, 1] for v in np.where(m)[0]
                     if P[:, v].std() > 1e-9 and R[:, v].std() > 1e-9]
            print(f"{roi:16s} {img_r:+16.3f} {np.mean(vtx_rs):+22.3f}")

        np.savez(NOD_DIR / "baseline_eval.npz", pred=P, real=R,
                 images=np.array(common))
        print(f"\nSaved -> {NOD_DIR/'baseline_eval.npz'}")
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()

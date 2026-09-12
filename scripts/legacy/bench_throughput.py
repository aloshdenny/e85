"""
bench_throughput.py

Where the time actually goes, and what can be bought back.

The GPU is already saturated (93% mean util, 100% peak, 23 GB of 24 GB), so
batching and "more parallelism" buy nothing. Two levers remain, and both are
measured here against the thing that matters -- whether the predictions still
agree with the fp32 / 64-frame reference:

  frames   V-JEPA2 fpc64-256 consumes 64 frames. Our inputs are STILL images
           held as a clip, so all 64 frames are identical and most of that
           compute is spent re-encoding the same picture. Fewer frames should
           cost proportionally less; the question is whether the predicted map
           moves.

  bf16     the model runs in fp32. A 4090 has bf16 tensor cores, and autocast
           needs no checkpoint changes.

A speedup is only usable if r vs the reference is high enough that part-effect
deltas (~0.002 against a ~0.085 baseline) survive it, so correlation is
reported alongside every timing.
"""

import sys, time, tempfile, shutil, argparse
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from abliteration import TribeModel, free
from infer_fairface_bulk import (
    get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline,
)


def set_num_frames(model, n):
    """video_feature is frozen pydantic; escalate via model_copy the same way
    validation.py's redirect does."""
    vf = model.data.video_feature
    new_vf = vf.model_copy(update={"num_frames": n})
    try:
        model.data.video_feature = new_vf
        return "video_feature"
    except Exception:
        pass
    model.data = model.data.model_copy(update={"video_feature": new_vf})
    return "data"


def run(model, imgs, tmp_root, tag, autocast=None):
    td = Path(tempfile.mkdtemp(prefix=f"bench_{tag}_", dir=tmp_root))
    try:
        rows = []
        for i, im in enumerate(imgs):
            tl = f"{tag}_{i}"
            p = td / f"{tl}.mp4"
            write_static_clip(im, p, duration=1.0, fps=2)
            rows.append((p, tl))
        df = make_multi_row_df(rows, duration=1.0)
        t0 = time.time()
        if autocast is not None:
            with torch.autocast("cuda", dtype=autocast):
                preds, segs = model.predict(events=df)
        else:
            preds, segs = model.predict(events=df)
        dt = time.time() - t0
        g = group_preds_by_timeline(preds, segs)
        out = np.stack([np.asarray(g[tl]) for _, tl in rows if tl in g])
        return dt, out
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    args = ap.parse_args()

    paths = sorted(Path("./val").glob("*.jpg"))[: args.n_images]
    imgs = [cv2.imread(str(p)) for p in paths]
    tmp_root = get_tmp_root()

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    vj = model.data.video_feature.image.model.model
    print(f"model dtype: {next(vj.parameters()).dtype}")

    print(f"\n{len(imgs)} images per condition\n")
    print(f"{'condition':22s} {'sec/image':>10s} {'speedup':>8s} {'r vs ref':>10s}")
    print("-" * 54)

    t_ref, ref = run(model, imgs, tmp_root, "ref")
    per_ref = t_ref / len(imgs)
    print(f"{'fp32, 64 frames (ref)':22s} {per_ref:10.2f} {1.0:8.2f}x {1.0:10.4f}")
    free()

    # bf16 autocast raises "unsupported ScalarType BFloat16": the result is
    # handed to numpy by the caching layer, which has no bf16. fp16 does convert.
    for name, dt_ in [("fp16 autocast, 64f", torch.float16),
                      ("bf16 autocast, 64f", torch.bfloat16)]:
        try:
            t, out = run(model, imgs, tmp_root, name[:4], autocast=dt_)
            r = float(np.corrcoef(ref.ravel(), out.ravel())[0, 1])
            print(f"{name:22s} {t/len(imgs):10.2f} {per_ref/(t/len(imgs)):8.2f}x {r:10.4f}")
        except Exception as e:
            print(f"{name:22s} {'failed':>10s}   {str(e)[:34]}")
        free()

    for nf in [32, 16, 8]:
        lvl = set_num_frames(model, nf)
        try:
            t, out = run(model, imgs, tmp_root, f"nf{nf}")
        except Exception as e:
            print(f"{'fp32, ' + str(nf) + ' frames':22s} failed: {str(e)[:40]}")
            continue
        r = (float(np.corrcoef(ref.ravel(), out.ravel())[0, 1])
             if out.shape == ref.shape else float("nan"))
        print(f"{'fp32, ' + str(nf) + ' frames':22s} {t/len(imgs):10.2f} "
              f"{per_ref/(t/len(imgs)):8.2f}x {r:10.4f}   (set at {lvl})")
        free()

    print("\nAt 13 passes per image, sec/image x 13 x N is the run cost.")
    print(f"  101,698 general images at the reference rate: "
          f"{per_ref*13*101698/3600:,.0f} GPU-hours")


if __name__ == "__main__":
    main()

"""Infer TRIBE only on the new in-the-wild Mia crops inside target/mia.zip.

Does not re-run the original 175. Writes target_preds/mia_in_the_wild.npz
and appends those rows onto target_preds/mia.npz if it exists.
"""

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from chunk_utils import ensure_fused_zip, resolve_zip_member, save_npz
from infer_fairface_bulk import (
    get_tmp_root, group_preds_by_timeline, make_multi_row_df, write_static_clip,
)
from infer_target_face import decode_image_from_zip
from abliteration import TribeModel, free
from measure_identity_signal import build_masks

ROIS = ["FACE(OFA+FFA)", "OFA", "FFA", "V1", "AUD", "MOTOR"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--out", type=Path, default=Path("./target_preds/mia_in_the_wild.npz"))
    ap.add_argument("--merge-into", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--cache-folder", type=Path, default=Path("/home/research/e85_cache"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    zpath = ensure_fused_zip(args.zip)
    with zipfile.ZipFile(zpath) as zf:
        members = [
            m for m in zf.namelist()
            if "in_the_wild" in m
            and Path(m).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and not Path(m).name.startswith("._")
        ]
        imgs = []
        for m in members:
            try:
                imgs.append((m, decode_image_from_zip(zf, m)))
            except Exception as e:
                print(f"skip {m}: {e}")
    print(f"{len(imgs)} in-the-wild images")
    if not imgs:
        raise SystemExit("no in_the_wild members in zip")

    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    tmp_root = get_tmp_root()
    names, vecs = [], []

    for s in range(0, len(imgs), args.batch):
        chunk = imgs[s:s + args.batch]
        td = Path(tempfile.mkdtemp(prefix="miawild_", dir=tmp_root))
        try:
            rows = [(td / f"w{s+i}.mp4", f"w{s+i}") for i in range(len(chunk))]
            for (_name, im), (clip, _) in zip(chunk, rows):
                write_static_clip(im, clip, duration=1.0, fps=2)
            df = make_multi_row_df(rows, duration=1.0)
            if args.no_fp16:
                preds, segs = model.predict(events=df)
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    preds, segs = model.predict(events=df)
            grouped = group_preds_by_timeline(preds, segs)
            for (name, _), (_, tl) in zip(chunk, rows):
                if tl not in grouped:
                    print(f"  missing pred {name}")
                    continue
                v = np.asarray(grouped[tl])
                v = v.mean(0) if v.ndim > 1 else v
                names.append(name)
                vecs.append(v.astype(np.float32))
        finally:
            shutil.rmtree(td, ignore_errors=True)
        print(f"  ...{min(s + args.batch, len(imgs))}/{len(imgs)}", flush=True)
        free()

    P = np.stack(vecs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.out, preds=P, filenames=np.array(names))
    print(f"saved {P.shape} -> {args.out}")

    if args.merge_into.exists():
        old = np.load(args.merge_into, allow_pickle=True)
        old_names = [str(x) for x in old["filenames"]]
        old_P = old["preds"]
        keep = [i for i, n in enumerate(old_names) if "in_the_wild" not in n]
        merged_names = [old_names[i] for i in keep] + names
        merged_P = np.concatenate([old_P[keep], P], axis=0)
        save_npz(args.merge_into, preds=merged_P, filenames=np.array(merged_names),
                 failed=old["failed"] if "failed" in old.files else np.array([]))
        print(f"merged into {args.merge_into}: {len(keep)} old + {len(names)} new = {len(merged_names)}")

    masks = build_masks()
    print(f"\n{'ROI':>14s} {'n':>4s} {'mean':>10s}")
    print("-" * 32)
    for r in ROIS:
        if r not in masks:
            continue
        m = P[:, masks[r]].mean(1)
        print(f"{r:>14s} {len(m):4d} {m.mean():+10.5f}")


if __name__ == "__main__":
    main()

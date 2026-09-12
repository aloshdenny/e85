"""Apply the Mia-suppress readout residual and report collateral + maps.

Captures encoder bottlenecks once, then compares frozen W0 vs W0+U@V:

  * top Facenet lookalikes (should NOT spike, and should not drop like Mia)
  * random general faces (anchor: stay put)
  * studio Mia + in-the-wild holdout (should drop)

Writes suppressed vertex preds for every Mia image so viz_suppress.py can
build the 3-way interactive map.

Usage:
  python scripts/apply_suppress_eval.py \
      --lookalikes-zip target/lookalikes.zip \
      --readout abliterated/mia_suppress_readout.npz
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from chunk_utils import ensure_fused_zip, save_npz
from infer_fairface_bulk import get_tmp_root
from infer_target_face import decode_image_from_zip
from measure_identity_signal import build_masks
from mia_suppress_readout import (
    IMAGE_EXTS, ROIS, capture_bottlenecks, find_fmri_encoder, free,
    load_mia_from_zip,
)
from abliteration import OUT_DIR


def load_split_from_zip(zpath: Path, prefix: str):
    items = []
    with zipfile.ZipFile(ensure_fused_zip(zpath)) as zf:
        names = [
            m for m in zf.namelist()
            if m.startswith(prefix)
            and Path(m).suffix.lower() in IMAGE_EXTS
            and not Path(m).name.startswith("._")
        ]
        for n in sorted(names):
            try:
                items.append((n, decode_image_from_zip(zf, n)))
            except Exception as e:
                print(f"skip {n}: {e}")
    return items


def report(label, base, ft, masks):
    print(f"\n{label}  n={base.shape[0]}")
    print(f"{'ROI':>14s} {'baseline':>10s} {'suppressed':>11s} {'delta':>10s}")
    print("-" * 50)
    for r in ROIS:
        bm = base[:, masks[r]].mean()
        fm = ft[:, masks[r]].mean()
        print(f"{r:>14s} {bm:+10.5f} {fm:+11.5f} {fm - bm:+10.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mia-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--lookalikes-zip", type=Path, default=Path("./target/lookalikes.zip"))
    ap.add_argument("--readout", type=Path, default=OUT_DIR / "mia_suppress_readout.npz")
    ap.add_argument("--cache-folder", type=Path,
                    default=Path("/home/research/.cache/huggingface"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-preds", type=Path, default=Path("./target_preds/mia_suppressed.npz"))
    ap.add_argument("--out-means", type=Path, default=Path("./target_preds/suppress_maps.npz"))
    ap.add_argument("--out-collateral", type=Path,
                    default=Path("./target_preds/lookalike_collateral.npz"))
    args = ap.parse_args()

    studio, wild = load_mia_from_zip(args.mia_zip)
    looks = load_split_from_zip(args.lookalikes_zip, "lookalike/")
    rand = load_split_from_zip(args.lookalikes_zip, "random/")
    print(f"Mia studio {len(studio)}  wild {len(wild)}  "
          f"lookalikes {len(looks)}  random {len(rand)}")
    if len(looks) < 5 or len(rand) < 5:
        raise SystemExit("lookalikes.zip missing lookalike/ or random/ splits")

    d = np.load(args.readout)
    U_np, V_np = d["U"].astype(np.float32), d["V"].astype(np.float32)
    print(f"residual rank={int(d['rank'])}  U{U_np.shape} V{V_np.shape}")

    print("Loading TribeModel...")
    model, fe = find_fmri_encoder(args.cache_folder)
    device = fe.device
    W0 = fe.predictor.weights[0].detach().clone()
    b0 = fe.predictor.bias[0].detach().clone()
    W0_t, b0_t = W0.to(device), b0.to(device)
    U = torch.tensor(U_np, device=device)
    V = torch.tensor(V_np, device=device)
    Wsup = W0_t + U @ V

    groups = {
        "studio": studio,
        "wild": wild,
        "lookalike": looks,
        "random": rand,
    }
    tmp_root = get_tmp_root()
    X = {}
    names = {}
    for k, items in groups.items():
        print(f"\nBottlenecks: {k} ({len(items)})")
        X[k], names[k] = capture_bottlenecks(model, fe, items, args.batch, tmp_root)
    free()

    def apply(Xin, W):
        with torch.no_grad():
            Xt = torch.tensor(Xin, device=device, dtype=torch.float32)
            return (Xt @ W + b0_t).cpu().numpy()

    base, ft = {}, {}
    for k in groups:
        base[k] = apply(X[k], W0_t)
        ft[k] = apply(X[k], Wsup)

    masks = build_masks()
    face_m = masks["FACE(OFA+FFA)"]
    report("STUDIO MIA (trained on)", base["studio"], ft["studio"], masks)
    report("IN-THE-WILD HOLDOUT", base["wild"], ft["wild"], masks)
    report("FACENET LOOKALIKES (not her)", base["lookalike"], ft["lookalike"], masks)
    report("RANDOM GENERAL FACES", base["random"], ft["random"], masks)

    def face_mean(arr):
        return float(arr[:, face_m].mean())

    mia_drop = face_mean(base["wild"]) - face_mean(ft["wild"])
    look_drop = face_mean(base["lookalike"]) - face_mean(ft["lookalike"])
    rand_drop = face_mean(base["random"]) - face_mean(ft["random"])
    print("\n" + "=" * 60)
    print("COLLATERAL")
    print(f"  wild Mia FACE drop:      {mia_drop:+.5f}")
    print(f"  lookalike FACE drop:     {look_drop:+.5f}")
    print(f"  random general FACE drop:{rand_drop:+.5f}")
    if abs(look_drop) < 0.4 * abs(mia_drop):
        print("  VERDICT: suppression is identity-selective "
              "(lookalikes moved << Mia).")
    elif look_drop > 0.8 * mia_drop:
        print("  VERDICT: suppression is NOT selective — lookalikes dropped "
              "almost as much as Mia.")
    else:
        print("  VERDICT: partial collateral — lookalikes moved some, less than Mia.")
    print("=" * 60)

    # Full Mia suppressed preds in original zip order: studio then wild,
    # matching how load_mia_from_zip walks the zip (studio first, then wild
    # is NOT zip order). Re-stack in zip member order for viz.
    mia_items = studio + wild
    name_to_ft = {}
    for split in ("studio", "wild"):
        for n, row in zip(names[split], ft[split]):
            name_to_ft[n] = row
    ordered_names = [n for n, _ in mia_items]
    missing = [n for n in ordered_names if n not in name_to_ft]
    if missing:
        raise SystemExit(f"missing suppressed preds for {len(missing)} images")
    P = np.stack([name_to_ft[n] for n in ordered_names]).astype(np.float32)
    args.out_preds.parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.out_preds, preds=P, filenames=np.array(ordered_names))
    print(f"\nSaved suppressed Mia preds -> {args.out_preds}  {P.shape}")

    np.savez(
        args.out_means,
        target_baseline=np.concatenate([base["studio"], base["wild"]]).mean(0),
        target_suppressed=P.mean(0),
        general_baseline=base["random"].mean(0),
        general_suppressed=ft["random"].mean(0),
        lookalike_baseline=base["lookalike"].mean(0),
        lookalike_suppressed=ft["lookalike"].mean(0),
    )
    print(f"Saved maps -> {args.out_means}")

    np.savez(
        args.out_collateral,
        lookalike_base=base["lookalike"], lookalike_ft=ft["lookalike"],
        lookalike_names=np.array(names["lookalike"]),
        random_base=base["random"], random_ft=ft["random"],
        random_names=np.array(names["random"]),
        wild_base=base["wild"], wild_ft=ft["wild"],
        studio_base=base["studio"], studio_ft=ft["studio"],
    )
    print(f"Saved collateral arrays -> {args.out_collateral}")


if __name__ == "__main__":
    main()

"""
check_face_roi_selectivity.py

Answers one question: for the target-person-vs-general-population contrast,
is the effect concentrated in face-selective ROIs (OFA, FFA, STS, ATL, TP,
+ optionally PREC/MPFC/PCC for familiarity), or is it diffuse across the
other ~20 non-face ROIs / the rest of cortex too?

Uses the same Destrieux (aparc.a2009s) exact-label masks as layer_analysis.py
so results are directly comparable to what you'd actually ablate.

Inputs (per-image .npz from the merged FairFace+FFHQ layout):
  --general-dir   folder of general-population .npz (default: ./fairface + ffhq preds)
                  each file: preds shape (20484,)  OR legacy (N, 20484)
  --target-npz    target .npz file or directory (default: ./target_preds)

Usage:
  python scripts/diagnostics/check_face_roi_selectivity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from chunk_utils import discover_npz, load_npz, npz_exists, preds_as_image_vectors

GENERAL_DIR = Path("./fairface + ffhq preds")
TARGET_NPZ = Path("./target_preds")

FACE_ROIS = {
    "OFA":  ["G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "Pole_occipital"],
    "FFA":  ["G_oc-temp_lat-fusifor"],
    "STS":  ["S_temporal_sup", "S_temporal_inf"],
    "ATL":  ["G_temporal_inf", "G_oc-temp_med-Parahip"],
    "TP":   ["Pole_temporal"],
    # Familiarity/autobiographical nodes -- toggle with --no-familiarity.
    "PREC": ["G_precuneus", "S_subparietal"],
    "MPFC": ["G_and_S_frontomargin", "G_and_S_transv_frontopol", "G_subcallosal"],
    "PCC":  ["G_cingul-Post-dorsal", "G_cingul-Post-ventral", "S_cingul-Marginalis"],
}

NON_FACE_ROIS = {
    "V1":   ["S_calcarine"],
    "V2":   ["G_cuneus", "S_parieto_occipital", "G_and_S_occipital_inf"],
    "V4":   ["G_occipital_middle", "G_occipital_sup", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal"],
    "MT":   ["G_oc-temp_lat-fusifor", "S_oc-temp_lat"],
    "LO":   ["G_oc-temp_med-Lingual", "S_oc-temp_med_and_Lingual"],
    "A1":   ["G_temp_sup-G_T_transv"],
    "STG":  ["G_temp_sup-Lateral", "G_temp_sup-Plan_tempo", "G_temp_sup-Plan_polar", "S_temporal_sup"],
    "PPA":  ["G_oc-temp_med-Parahip", "S_collat_transv_ant", "S_collat_transv_post"],
    "MTG":  ["G_temporal_middle", "S_temporal_inf"],
    "ITG":  ["G_temporal_inf", "S_oc-temp_med_and_Lingual"],
    "PT":   ["Lat_Fis-post"],
    "S1":   ["G_postcentral", "S_postcentral"],
    "SPL":  ["G_parietal_sup", "S_intrapariet_and_P_trans"],
    "IPL":  ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    "TPJ":  ["G_and_S_subcentral", "S_interm_prim-Jensen"],
    "MPC":  ["S_subparietal", "G_cingul-Post-ventral"],
    "M1":   ["G_precentral", "S_precentral-inf-part", "S_precentral-sup-part"],
    "PMC":  ["G_and_S_paracentral", "G_front_sup"],
    "DLPFC":["G_front_middle", "S_front_middle", "S_front_sup"],
    "IFG":  ["G_front_inf-Opercular", "G_front_inf-Triangul", "G_front_inf-Orbital", "S_front_inf"],
    "OFC":  ["G_orbital", "G_rectus", "S_orbital_lateral", "S_orbital_med-olfact", "S_orbital-H_Shaped", "S_suborbital"],
    "ACC":  ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant", "G_and_S_cingul-Mid-Post"],
    "FPC":  ["G_and_S_frontomargin", "G_and_S_transv_frontopol"],
    "INS":  ["G_Ins_lg_and_S_cent_ins", "G_insular_short", "S_circular_insula_ant", "S_circular_insula_inf", "S_circular_insula_sup"],
}


def build_masks(roi_dict, atlas_names, lh_labels, rh_labels):
    masks = {}
    for key, exact in roi_dict.items():
        idxs = [i for i, n in enumerate(atlas_names) if n in exact]
        if not idxs:
            continue
        mask = np.concatenate([np.isin(lh_labels, idxs), np.isin(rh_labels, idxs)])
        if mask.sum() > 0:
            masks[key] = mask
    return masks


def load_general_mean(general_dir: Path) -> tuple[np.ndarray, int]:
    """
    Running mean over all general-population images.

    Streams one npz at a time so ~100k single-image files do not need to be
    stacked in RAM (~8GB). Equal weight per image (no demographic buckets).
    """
    npz_files = discover_npz(general_dir)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {general_dir}")

    first = preds_as_image_vectors(load_npz(npz_files[0])["preds"])
    print(
        f"  Detected npz format: "
        f"{'single-image (1D preds)' if first.shape[0] == 1 and len(npz_files) > 1 else 'per-file preds'} "
        f"-- pooling {len(npz_files)} file(s) with equal weight per image"
    )

    acc = np.zeros(20484, dtype=np.float64)
    n_images = 0
    for f in npz_files:
        vecs = preds_as_image_vectors(load_npz(f)["preds"])
        acc += vecs.sum(axis=0)
        n_images += vecs.shape[0]

    mean = (acc / n_images).astype(np.float32)
    assert mean.shape == (20484,), (
        f"general_mean has shape {mean.shape}, expected (20484,) -- "
        f"1D preds were likely mean'd across vertices instead of images."
    )
    return mean, n_images


def load_target_mean(target_npz: Path) -> tuple[np.ndarray, int]:
    target_path = Path(target_npz)
    if target_path.is_dir():
        target_files = discover_npz(target_path, recursive=True)
    elif npz_exists(target_path):
        target_files = [target_path]
    else:
        target_files = [
            f for f in discover_npz(Path("."), recursive=True)
            if str(target_npz) in str(f)
        ]

    if not target_files:
        raise FileNotFoundError(f"No target .npz files found matching {target_npz}")

    acc = np.zeros(20484, dtype=np.float64)
    n_images = 0
    for f in target_files:
        vecs = preds_as_image_vectors(load_npz(f)["preds"])
        acc += vecs.sum(axis=0)
        n_images += vecs.shape[0]

    mean = (acc / n_images).astype(np.float32)
    assert mean.shape == (20484,)
    return mean, n_images


def main():
    parser = argparse.ArgumentParser(
        description="Face ROI selectivity for target vs FairFace+FFHQ general population."
    )
    parser.add_argument(
        "--general-dir", default=GENERAL_DIR, type=Path,
        help="Folder of general-population per-image .npz "
             "(default: ./fairface + ffhq preds)",
    )
    parser.add_argument(
        "--target-npz", default=TARGET_NPZ, type=Path,
        help="Target person .npz file or directory (default: ./target_preds)",
    )
    parser.add_argument(
        "--no-familiarity", action="store_true",
        help="Drop PREC/MPFC/PCC from the face ROI set -- use this "
             "if your target is generic face detection, not a "
             "specific known person.",
    )
    parser.add_argument(
        "--top-pct", type=float, default=5.0,
        help="Percentile threshold (of |contrast|) used for the "
             "concentration check.",
    )
    args = parser.parse_args()

    print("Loading Destrieux atlas (fsaverage5)...")
    from nilearn import datasets as nl_datasets
    destrieux = nl_datasets.fetch_atlas_surf_destrieux()
    lh_labels = np.array(destrieux["map_left"])
    rh_labels = np.array(destrieux["map_right"])
    atlas_names = [n.decode() if isinstance(n, bytes) else n for n in destrieux["labels"]]

    face_dict = dict(FACE_ROIS)
    if args.no_familiarity:
        for k in ["PREC", "MPFC", "PCC"]:
            face_dict.pop(k, None)

    face_masks = build_masks(face_dict, atlas_names, lh_labels, rh_labels)
    non_face_masks = build_masks(NON_FACE_ROIS, atlas_names, lh_labels, rh_labels)

    print(f"Face ROIs retained: {list(face_masks.keys())}")
    print(f"Non-face ROIs retained: {list(non_face_masks.keys())}")

    print(f"\nLoading general population from {args.general_dir} ...")
    general_mean, n_general = load_general_mean(args.general_dir)
    print(f"  {n_general} images, general_mean shape={general_mean.shape}")

    print(f"Loading target person from {args.target_npz} ...")
    target_mean, n_target = load_target_mean(args.target_npz)
    print(f"  {n_target} target image(s), target_mean shape={target_mean.shape}")

    contrast = target_mean - general_mean  # (20484,)

    all_masks = {**face_masks, **non_face_masks}
    is_face = {k: (k in face_masks) for k in all_masks}

    rows = []
    for key, mask in all_masks.items():
        val = float(contrast[mask].mean())
        absval = float(np.abs(contrast[mask]).mean())
        rows.append((key, is_face[key], val, absval, int(mask.sum())))

    rows.sort(key=lambda r: abs(r[2]), reverse=True)

    print("\n" + "=" * 70)
    print(f"{'ROI':<8} {'face?':<6} {'mean contrast':>14} {'mean |contrast|':>16} {'n_verts':>8}")
    print("-" * 70)
    for key, face, val, absval, n in rows:
        marker = "FACE" if face else ""
        print(f"{key:<8} {marker:<6} {val:>+14.5f} {absval:>16.5f} {n:>8d}")

    face_vals = [r[2] for r in rows if r[1]]
    non_face_vals = [r[2] for r in rows if not r[1]]

    print("\n" + "=" * 70)
    print("MARGIN CHECK")
    print("=" * 70)
    print(f"Face ROI contrast values:     min={min(face_vals):+.5f}  max={max(face_vals):+.5f}  mean={np.mean(face_vals):+.5f}")
    print(f"Non-face ROI contrast values: min={min(non_face_vals):+.5f}  max={max(non_face_vals):+.5f}  mean={np.mean(non_face_vals):+.5f}")
    strict_margin = min(face_vals) - max(non_face_vals)
    mean_margin = np.mean(face_vals) - np.mean(non_face_vals)
    print(f"\nStrict margin (weakest face ROI - strongest non-face ROI): {strict_margin:+.5f}")
    print(f"Mean margin   (avg face ROI - avg non-face ROI):            {mean_margin:+.5f}")
    if strict_margin > 0:
        print(">> Clean separation: every face ROI exceeds every non-face ROI.")
    elif mean_margin > 0:
        print(">> Face ROIs are stronger on average, but overlap with some non-face ROIs.")
    else:
        print(">> No clear separation -- effect looks diffuse, not face-specific.")

    all_31_mask = np.zeros_like(contrast, dtype=bool)
    for m in all_masks.values():
        all_31_mask |= m
    rest_of_cortex_mask = ~all_31_mask

    total_energy = float((contrast ** 2).sum())
    face_energy = float(sum((contrast[m] ** 2).sum() for m in face_masks.values())) if face_masks else 0.0
    non_face_energy = float(sum((contrast[m] ** 2).sum() for m in non_face_masks.values())) if non_face_masks else 0.0
    rest_energy = float((contrast[rest_of_cortex_mask] ** 2).sum())

    print("\n" + "=" * 70)
    print("ENERGY CONCENTRATION (sum of squared contrast values)")
    print("=" * 70)
    print(f"Face ROIs:          {face_energy/total_energy*100:5.1f}%  ({sum(m.sum() for m in face_masks.values())} verts)")
    print(f"Non-face ROIs (30): {non_face_energy/total_energy*100:5.1f}%  ({sum(m.sum() for m in non_face_masks.values())} verts)")
    print(f"Rest of cortex:     {rest_energy/total_energy*100:5.1f}%  ({rest_of_cortex_mask.sum()} verts)")

    n_face_verts = sum(m.sum() for m in face_masks.values())
    n_total_verts = len(contrast)
    expected_pct_if_uniform = n_face_verts / n_total_verts * 100
    actual_pct = face_energy / total_energy * 100
    print(f"\nIf effect were uniform across cortex, face ROIs "
          f"({n_face_verts} of {n_total_verts} verts) would hold "
          f"{expected_pct_if_uniform:.1f}% of energy.")
    print(f"Actual: {actual_pct:.1f}%  "
          f"({'concentrated' if actual_pct > expected_pct_if_uniform * 1.5 else 'roughly uniform / diffuse'})")

    thresh = np.percentile(np.abs(contrast), 100 - args.top_pct)
    top_mask = np.abs(contrast) >= thresh
    top_in_face = float(
        (top_mask & all_31_mask & np.logical_or.reduce(list(face_masks.values()))).sum()
    ) if face_masks else 0.0
    print(f"\nOf the top {args.top_pct}% strongest-contrast vertices "
          f"({top_mask.sum()} verts), {top_in_face:.0f} "
          f"({top_in_face/max(top_mask.sum(),1)*100:.1f}%) fall inside face ROIs.")


if __name__ == "__main__":
    main()

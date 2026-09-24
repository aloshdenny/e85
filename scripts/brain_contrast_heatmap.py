"""
brain_contrast_heatmap.py

The "White Christmas" question, answered in brain space rather than pixel
space: where on the CORTEX does the target photo set differ from the general
population? Zero new inference -- every .npz already holds the full
20484-vertex prediction vector, we've only ever compared them through a
single mask-averaged scalar (validation.py's `y`). This computes and renders
the FULL signed contrast (target_mean - general_mean) across every vertex,
with the face-processing circuit's ROI boundaries drawn as an outline on top
-- so you can see directly whether the strongest contrast lands inside
genuine face-selective cortex or spills outside it (the same question
check_face_roi_selectivity.py answered as a printed table, now as a picture).

Reuses the mesh-loading/IDW-upsampling pattern from viz.py. Diverging
colormap (RdBu_r, centered at 0) rather than viz.py's abs-value hot-on-sulc
style, since a signed contrast needs to show BOTH directions: red where the
target runs higher than general population, blue where general population
runs higher.

Defaults to the merged FairFace+FFHQ per-image preds layout.

Output names (outline mode only changes the black ROI outline + filename):
  default              -> ofa_ffa.png        (OFA+FFA outline)
  --include-secondary  -> all_visual.png     (full face circuit outline)
  --entire             -> entire_brain.png   (no ROI outline; full cortex)

Usage:
  python scripts/brain_contrast_heatmap.py
  python scripts/brain_contrast_heatmap.py --include-secondary
  python scripts/brain_contrast_heatmap.py --entire
"""

import os, sys, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import numpy as np
from pathlib import Path
from nilearn import datasets, surface
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_utils import discover_npz, load_npz, preds_as_image_vectors

N_LH5 = 10242  # fsaverage5 vertices per hemisphere

GENERAL_DIR = Path("./fairface + ffhq preds")
TARGET_DIR = Path("./target_preds")

# ── Same face-circuit ROI definitions used throughout abliteration.py ──────
# (Kept here standalone rather than importing, so this script has no
# dependency on the abliteration pipeline being importable / its OUT_DIR
# existing -- purely a reader of .npz files.)

PRIMARY_FACE_ROIS = {
    "OFA": ["G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "Pole_occipital"],
    "FFA": ["G_oc-temp_lat-fusifor"],
}
SECONDARY_FACE_ROIS = {
    "STS":  ["S_temporal_sup", "S_temporal_inf"],
    "ATL":  ["G_temporal_inf", "G_oc-temp_med-Parahip"],
    "TP":   ["Pole_temporal"],
    "PREC": ["G_precuneus", "S_subparietal"],
    "MPFC": ["G_and_S_frontomargin", "G_and_S_transv_frontopol", "G_subcallosal"],
    "PCC":  ["G_cingul-Post-dorsal", "G_cingul-Post-ventral", "S_cingul-Marginalis"],
}


def load_group_mean(preds_dir: Path, recursive=False):
    """
    Running mean over all .npz files in a directory.

    Handles multi-image category npz (preds: (N,20484)) and single-image
    npz (preds: (20484,)). Streams one file at a time so ~100k FairFace+FFHQ
    per-image files do not need to be stacked in RAM.
    """
    files = discover_npz(preds_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {preds_dir}")

    first = preds_as_image_vectors(load_npz(files[0])["preds"])
    print(
        f"  Detected format: "
        f"{'single-image (1D preds)' if first.shape[0] == 1 else 'multi-image (2D preds)'} "
        f"-- {len(files)} file(s)"
    )

    acc = np.zeros(20484, dtype=np.float64)
    n_images = 0
    for f in files:
        vecs = preds_as_image_vectors(load_npz(f)["preds"])
        acc += vecs.sum(axis=0)
        n_images += vecs.shape[0]

    mean = (acc / n_images).astype(np.float32)
    assert mean.shape == (20484,), (
        f"group mean has shape {mean.shape}, expected (20484,) -- "
        f"1D preds were likely averaged across vertices instead of images."
    )
    return mean, len(files), n_images


def build_face_circuit_masks(atlas_names, lh_labels, rh_labels, include_secondary):
    roi_dict = dict(PRIMARY_FACE_ROIS)
    if include_secondary:
        roi_dict.update(SECONDARY_FACE_ROIS)
    masks = {}
    for key, exact in roi_dict.items():
        idxs = [i for i, n in enumerate(atlas_names) if n in exact]
        if not idxs:
            continue
        mask = np.concatenate([np.isin(lh_labels, idxs), np.isin(rh_labels, idxs)])
        if mask.sum() > 0:
            masks[key] = mask
    return masks


def build_idw(src_sphere, dst_sphere, k=8):
    tree = cKDTree(src_sphere)
    dist, idx = tree.query(dst_sphere, k=k)
    w = 1.0 / (dist ** 2 + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    return idx, w


def upsample(data_1d, lh_idx, lh_w, rh_idx, rh_w, n_lh5=N_LH5):
    lh5 = data_1d[:n_lh5]
    rh5 = data_1d[n_lh5:]
    lh_full = (lh5[lh_idx] * lh_w).sum(axis=1)
    rh_full = (rh5[rh_idx] * rh_w).sum(axis=1)
    return np.concatenate([lh_full, rh_full])


def upsample_mask(mask_1d, lh_idx, rh_idx, n_lh5=N_LH5):
    """Boolean masks upsample via nearest-source-vertex (majority of the k
    IDW neighbors), not weighted averaging -- keeps ROI boundaries crisp
    rather than fuzzy."""
    lh5 = mask_1d[:n_lh5]
    rh5 = mask_1d[n_lh5:]
    lh_full = lh5[lh_idx][:, 0]  # nearest neighbor is idx[:,0] since query() sorts by distance
    rh_full = rh5[rh_idx][:, 0]
    return np.concatenate([lh_full, rh_full])


def render_hemisphere(ax, coords, faces, activation, roi_mask, vmax, title):
    vy = coords[:, 1]
    vz = coords[:, 2]

    face_act = activation[faces].mean(axis=1)
    face_roi = roi_mask[faces].mean(axis=1) > 0.5  # majority of face's 3 verts in an ROI

    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    face_colors = cmap(norm(face_act))

    depth = coords[faces, 0].mean(axis=1)
    order = np.argsort(depth)

    verts2d = np.stack([vy[faces], vz[faces]], axis=2)[order]
    fc = face_colors[order]

    ax.set_facecolor("#0d0d0d")
    ax.set_aspect("equal")
    ax.axis("off")
    pc = PolyCollection(verts2d, facecolors=fc, edgecolors="none", linewidths=0)
    ax.add_collection(pc)

    # ROI boundary outline: thin black edge around every ROI face (cheap,
    # reads as a boundary given how densely tessellated the mesh is).
    roi_verts2d = verts2d[face_roi[order]]
    if len(roi_verts2d) > 0:
        pc_outline = PolyCollection(roi_verts2d, facecolors="none",
                                    edgecolors="black", linewidths=0.4, alpha=0.6)
        ax.add_collection(pc_outline)

    ax.set_xlim(vy.min() - 5, vy.max() + 5)
    ax.set_ylim(vz.min() - 5, vz.max() + 5)
    ax.set_title(title, color="white", fontsize=11, pad=6)
    return norm, cmap


def main():
    parser = argparse.ArgumentParser(
        description="Brain-surface contrast heatmap: target vs FairFace+FFHQ general population."
    )
    parser.add_argument(
        "--general-preds-dir", default=GENERAL_DIR, type=Path,
        help="General-population per-image .npz folder "
             '(default: "./fairface + ffhq preds")',
    )
    parser.add_argument(
        "--target-preds-dir", default=TARGET_DIR, type=Path,
        help="Target person .npz file or directory (default: ./target_preds)",
    )
    parser.add_argument("--out-dir", default=Path("./brain_contrast"), type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--include-secondary", action="store_true",
                      help="Outline the full face circuit (OFA/FFA + STS/ATL/TP/PREC/MPFC/PCC). "
                           "Saves as all_visual.png.")
    mode.add_argument("--entire", action="store_true",
                      help="No ROI outline -- full-cortex contrast map. "
                           "Saves as entire_brain.png.")
    parser.add_argument("--vmax-percentile", type=float, default=99.0,
                        help="Colorscale is symmetric around 0, clipped at this "
                             "percentile of |contrast| so a few extreme outlier "
                             "vertices don't wash out the rest of the map.")
    args = parser.parse_args()

    if args.entire:
        outline_mode = "entire"
        out_name = "entire_brain.png"
    elif args.include_secondary:
        outline_mode = "all_visual"
        out_name = "all_visual.png"
    else:
        outline_mode = "ofa_ffa"
        out_name = "ofa_ffa.png"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading general population from {args.general_preds_dir} ...")
    general_mean, n_gen_files, n_gen_images = load_group_mean(args.general_preds_dir)
    print(f"  {n_gen_files} files, {n_gen_images} images, mean shape={general_mean.shape}")

    print(f"Loading target from {args.target_preds_dir} ...")
    target_mean, n_tgt_files, n_tgt_images = load_group_mean(
        args.target_preds_dir, recursive=True
    )
    print(f"  {n_tgt_files} files, {n_tgt_images} images, mean shape={target_mean.shape}")

    contrast = target_mean - general_mean  # (20484,) signed, native fsaverage5

    print(f"\nContrast stats: min={contrast.min():+.5f} max={contrast.max():+.5f} "
          f"mean={contrast.mean():+.5f}")

    print("\nLoading Destrieux atlas + fsaverage meshes...")
    destrieux = datasets.fetch_atlas_surf_destrieux()
    lh_labels = np.array(destrieux["map_left"])
    rh_labels = np.array(destrieux["map_right"])
    atlas_names = [n.decode() if isinstance(n, bytes) else n for n in destrieux["labels"]]

    if outline_mode == "entire":
        face_masks = {}
        combined_roi_mask = np.zeros(20484, dtype=bool)
        print("Outline mode: entire brain (no ROI outline)")
    else:
        face_masks = build_face_circuit_masks(
            atlas_names, lh_labels, rh_labels,
            include_secondary=(outline_mode == "all_visual"),
        )
        print(f"Outline mode: {outline_mode} -> {list(face_masks.keys())}")
        combined_roi_mask = np.zeros(20484, dtype=bool)
        for m in face_masks.values():
            combined_roi_mask |= m

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    lhF_coords, lhF_faces = surface.load_surf_mesh(fsaverage["pial_left"])
    rhF_coords, rhF_faces = surface.load_surf_mesh(fsaverage["pial_right"])

    lh5_sphere, _ = surface.load_surf_mesh(fsaverage5["sphere_left"])
    rh5_sphere, _ = surface.load_surf_mesh(fsaverage5["sphere_right"])
    lhF_sphere, _ = surface.load_surf_mesh(fsaverage["sphere_left"])
    rhF_sphere, _ = surface.load_surf_mesh(fsaverage["sphere_right"])

    print("Building IDW interpolants (fsaverage5 -> full fsaverage)...")
    lhF_idx, lhF_w = build_idw(lh5_sphere, lhF_sphere)
    rhF_idx, rhF_w = build_idw(rh5_sphere, rhF_sphere)

    contrast_full = upsample(contrast, lhF_idx, lhF_w, rhF_idx, rhF_w)
    roi_mask_full = upsample_mask(combined_roi_mask, lhF_idx, rhF_idx)

    n_lh_full = lhF_coords.shape[0]
    lh_contrast = contrast_full[:n_lh_full]
    rh_contrast = contrast_full[n_lh_full:]
    lh_roi = roi_mask_full[:n_lh_full]
    rh_roi = roi_mask_full[n_lh_full:]

    vmax = float(np.percentile(np.abs(contrast_full), args.vmax_percentile))
    print(f"Colorscale range: +/-{vmax:.5f} ({args.vmax_percentile}th percentile of |contrast|)")

    fig = plt.figure(figsize=(16, 8), facecolor="#0d0d0d")
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.02)

    ax_lh = fig.add_subplot(gs[0, 0])
    ax_rh = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    norm, cmap = render_hemisphere(ax_lh, lhF_coords, lhF_faces, lh_contrast, lh_roi,
                                   vmax, "Left Hemisphere")
    render_hemisphere(ax_rh, rhF_coords, rhF_faces, rh_contrast, rh_roi,
                      vmax, "Right Hemisphere")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("target \u2212 general population", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    if outline_mode == "entire":
        outline_note = "entire cortex (no ROI outline)"
    else:
        outline_note = f"black outline = {'+'.join(face_masks.keys())}"

    fig.suptitle(
        f"Brain-surface contrast: target ({n_tgt_images} images) vs "
        f"FairFace+FFHQ general ({n_gen_images} images)  |  {outline_note}",
        color="white", fontsize=12, y=0.98,
    )

    out_path = args.out_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")

    np.save(args.out_dir / "contrast_native_fsaverage5.npy", contrast)
    print(f"Raw signed contrast vector (native fsaverage5, 20484,) -> "
          f"{args.out_dir / 'contrast_native_fsaverage5.npy'}")

    if outline_mode != "entire":
        # How much of the contrast lands inside vs outside the outlined circuit
        inside_mean = float(np.abs(contrast[combined_roi_mask]).mean()) if combined_roi_mask.sum() else 0.0
        outside_mean = float(np.abs(contrast[~combined_roi_mask]).mean())
        print(f"\nMean |contrast| inside outline:  {inside_mean:.5f} "
              f"({combined_roi_mask.sum()} verts)")
        print(f"Mean |contrast| outside outline: {outside_mean:.5f} "
              f"({(~combined_roi_mask).sum()} verts)")
        print(f"Ratio (inside/outside): {inside_mean/max(outside_mean,1e-9):.2f}x")
    else:
        print(f"\nMean |contrast| over entire cortex: "
              f"{float(np.abs(contrast).mean()):.5f} ({len(contrast)} verts)")


if __name__ == "__main__":
    main()

"""
plot_visual_cortex_wireframe.py

Dark-theme, zoomed, bihemispheric wireframe/point-cloud render of the same 4
selected predictions (baseline, A, B, C), styled after a reference image the
user provided: black background, translucent mesh wireframe of the cortical
surface, both occipital poles visible from behind, tight crop on just the
visual-cortex patch, ROI rendered as a dense coloured point cloud on top of
the wireframe.

WHY A CUSTOM RENDERER: nilearn's plot_surf_stat_map draws one shaded,
solid-lit hemisphere per call and has no wireframe/point-cloud mode -- it
cannot produce this look directly. This builds the same wireframe (mesh
edges as thin translucent lines) and ROI (masked vertices as a scatter of
points colours by activation) by hand with matplotlib's own 3D toolkit, using
the SAME data, mask, and shared colour scale as the flat-lateral version.

VIEWPOINT: pial_left and pial_right are co-registered in one shared
coordinate frame (checked directly: left hemisphere x in [-68.8, 1.2], right
in [-0.1, 69.8], both share the same y/z ranges), so both can be drawn in one
3D scene without manual alignment. Rather than guess a posterior azimuth/
elevation, the camera direction is computed FROM THE DATA: the vector from
the whole-brain centroid to the ROI (face mask) centroid, so the camera
always ends up facing the visual cortex regardless of any assumption about
which axis is anterior-posterior. Same fixed view + same zoom (cropped to the
ROI's own bounding box, padded) + same colour scale for all 4, exactly like
the lateral version -- only the mask determines what's shown, never the
individual image's own extent, so the four are still honestly comparable.

Usage:
  python scripts/plot_visual_cortex_wireframe.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from nilearn import datasets as nl_datasets

OUT_DIR = Path("/Users/aoxo/vscode/e85/abliterated")
SEL_NPZ = OUT_DIR / "visual_cortex_snapshot_selection.npz"
SEL_JSON = OUT_DIR / "visual_cortex_snapshot_selection.json"


def load_hemisphere(path):
    import nibabel as nib
    try:
        coords, faces = nib.freesurfer.read_geometry(path)
    except ValueError:
        g = nib.load(path)
        coords, faces = g.darrays[0].data, g.darrays[1].data
    return coords.astype(np.float64), faces


def mesh_edges(coords, faces, stride=1):
    """Unique undirected edges as line segments, subsampled by `stride` faces
    for a lighter wireframe (10,242-vertex meshes have ~30k edges/hemisphere;
    drawing all of them is legible but slow to iterate on, stride keeps the
    grid look while cutting draw time)."""
    f = faces[::stride]
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    segs = np.stack([coords[edges[:, 0]], coords[edges[:, 1]]], axis=1)
    return segs


def main():
    sel = json.load(open(SEL_JSON))
    d = np.load(SEL_NPZ)
    face_mask = d["face_mask"]  # (20484,) bool, left-then-right

    fsavg5 = nl_datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    coords_l, faces_l = load_hemisphere(fsavg5["pial_left"])
    coords_r, faces_r = load_hemisphere(fsavg5["pial_right"])
    coords_full = np.concatenate([coords_l, coords_r], axis=0)  # (20484,3)

    edges_l = mesh_edges(coords_l, faces_l, stride=2)
    edges_r = mesh_edges(coords_r, faces_r, stride=2)

    roi_idx = np.where(face_mask)[0]
    roi_coords = coords_full[roi_idx]
    roi_centroid = roi_coords.mean(axis=0)
    brain_centroid = coords_full.mean(axis=0)

    # camera direction: from the whole-brain centroid toward the ROI centroid,
    # so the view always faces the visual cortex regardless of axis convention
    view_dir = roi_centroid - brain_centroid
    view_dir /= np.linalg.norm(view_dir)
    azim = float(np.degrees(np.arctan2(view_dir[1], view_dir[0])))
    elev = float(np.degrees(np.arcsin(np.clip(view_dir[2], -1, 1))))
    print(f"ROI centroid {roi_centroid.round(1)}, brain centroid {brain_centroid.round(1)}")
    print(f"computed view: elev={elev:.1f} azim={azim:.1f}")

    # crop tight on the ROI's own bounding box, padded, shared across all 4.
    # First attempt used pad=0.6x the ROI extent and left too much of the
    # whole head visible around the patch compared to the reference image,
    # which fills most of the frame with just the two occipital lobes --
    # tightened to 0.22x.
    pad = 0.22 * max(roi_coords.max(axis=0) - roi_coords.min(axis=0))
    lo = roi_coords.min(axis=0) - pad
    hi = roi_coords.max(axis=0) + pad

    panels = ["baseline", "A", "B", "C"]
    zone_label = {p: sel[p].get("zone", "") for p in panels}
    titles = {
        "baseline": f"Baseline (flat)   mean={sel['baseline']['mean']:+.3f}",
        "A": f"Face A   mean={sel['A']['mean']:+.3f}  spike={sel['A']['spike']:.3f}\n{zone_label['A']}",
        "B": f"Face B   mean={sel['B']['mean']:+.3f}  spike={sel['B']['spike']:.3f}\n{zone_label['B']}",
        "C": f"Face C   mean={sel['C']['mean']:+.3f}  spike={sel['C']['spike']:.3f}\n{zone_label['C']}",
    }

    preds = {p: d[f"pred_{p}"].astype(np.float64) for p in panels}
    all_masked_vals = np.concatenate([preds[p][face_mask] for p in panels])
    vmin, vmax = np.percentile(all_masked_vals, [1, 99])
    print(f"shared colour scale: [{vmin:.4f}, {vmax:.4f}]")

    for p in panels:
        fig = plt.figure(figsize=(9, 8), facecolor="black")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        for segs in (edges_l, edges_r):
            lc = Line3DCollection(segs, colors=(0.55, 0.6, 0.7, 0.35), linewidths=0.35)
            ax.add_collection3d(lc)

        vals = preds[p][roi_idx]
        order = np.argsort(vals)  # brightest drawn last, on top
        sc = ax.scatter(
            roi_coords[order, 0], roi_coords[order, 1], roi_coords[order, 2],
            c=vals[order], cmap="hot", vmin=vmin, vmax=vmax,
            s=26, depthshade=False, edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cbar.outline.set_edgecolor("white")

        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
        ax.set_axis_off()
        ax.set_title(titles[p], color="white", fontsize=12, pad=10)

        out_png = OUT_DIR / f"visual_cortex_wireframe_{p}.png"
        fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()

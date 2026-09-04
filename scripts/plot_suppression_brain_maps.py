"""
plot_suppression_brain_maps.py

Brain maps comparing the three suppression conditions run for each person:
OFA/FFA-only, vATL-only, and the combined OFA/FFA+vATL edit.

WHAT IS PLOTTED. Every *_suppress*.npz from atl_suppress_readout.py /
mia_suppress_readout_v2.py stores the trained low-rank residual as
U (2048, rank) and V (rank, 20484) -- the edit is `W = W0 + U @ V`, added
directly to the frozen readout's output columns (one column per fsaverage5
vertex). So `residual = U @ V` has one column per vertex, and that column's
L2 norm is exactly "how much this vertex's predicted response is allowed to
move" under the trained edit -- independent of any specific input image.
That is what a brain map of the edit itself (as opposed to a brain map of
one person's predicted response) should show, and it needs nothing besides
the npz already saved: no pods, no re-run of inference.

For the --mask-to-target runs this column norm is EXACTLY zero outside the
targeted ROI by construction (the mask multiplies the residual before it's
added), so the map is a direct visual proof of the architectural guarantee,
not just a summary statistic.

Usage:
  python scripts/plot_suppression_brain_maps.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets as nl_datasets
from nilearn import plotting as nl_plotting

import sys
sys.path.append(str(Path(__file__).parent))
from measure_identity_signal import build_masks

OUT_DIR = Path("/Users/aoxo/vscode/e85/abliterated/brain_maps")
ABL_DIR = Path("/Users/aoxo/vscode/e85/abliterated")

PEOPLE = {
    "Mia": {
        "OFA/FFA only":      ABL_DIR / "mia_suppress_readout_v2_face_top_lam15.npz",
        "vATL only":         ABL_DIR / "atl_mia_masked.npz",
        "OFA/FFA + vATL":    ABL_DIR / "both_suppress_mia.npz",
    },
    "Sins": {
        "OFA/FFA only":      ABL_DIR / "sins_suppress_readout_v2_face_top_lam15.npz",
        "vATL only":         ABL_DIR / "atl_sins_masked.npz",
        "OFA/FFA + vATL":    ABL_DIR / "both_suppress_sins.npz",
    },
    "Michael Jackson": {
        "OFA/FFA only":      ABL_DIR / "mj_suppress_readout_v2_face_top_lam15.npz",
        "vATL only":         ABL_DIR / "atl_mj_masked.npz",
        "OFA/FFA + vATL":    ABL_DIR / "both_suppress_mj.npz",
    },
}
CONDITIONS = ["OFA/FFA only", "vATL only", "OFA/FFA + vATL"]


def edit_magnitude(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    U, V = d["U"], d["V"]
    residual = U @ V  # (2048, 20484)
    return np.linalg.norm(residual, axis=0)  # (20484,) one value per vertex


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fsavg5 = nl_datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    masks = build_masks()
    n_left = 10242  # fsaverage5: 10242 verts/hemisphere, concat lh then rh

    # shared color scale across every person/condition so panels are honestly
    # comparable, not each auto-scaled to its own max
    all_mags = []
    for person, conds in PEOPLE.items():
        for cond, path in conds.items():
            if path.exists():
                all_mags.append(edit_magnitude(path))
    vmax = float(np.percentile(np.concatenate(all_mags), 99.5))
    print(f"shared color scale: [0, {vmax:.4f}]")

    # The "free" (non --mask-to-target) runs anchor non-target vertices
    # toward zero with a soft loss penalty, not an architectural zero, so
    # they carry a small nonzero floor (~1e-3) across the WHOLE surface
    # instead of being exactly zero outside the target. Left at nilearn's
    # default near-zero threshold, that floor paints as "near-black hot"
    # over the entire hemisphere instead of falling through to the sulcal
    # background -- visually indistinguishable from a real edit. The true
    # suppressed-region signal is two to three orders of magnitude above
    # that floor (see the checked percentiles), so a fixed threshold well
    # above the floor and far below the signal makes every panel -- masked
    # or free -- fall back to plain cortex outside the region that was
    # actually targeted.
    THRESH = 0.02

    for person, conds in PEOPLE.items():
        missing = [c for c, p in conds.items() if not p.exists()]
        if missing:
            print(f"[skip incomplete] {person}: missing {missing}")
            continue

        fig, axes = plt.subplots(
            2, 3, figsize=(15, 8), subplot_kw={"projection": "3d"}
        )
        fig.suptitle(f"{person} — edit magnitude by suppression target", fontsize=15)

        for col, cond in enumerate(CONDITIONS):
            mag = edit_magnitude(conds[cond])
            mag_l, mag_r = mag[:n_left], mag[n_left:]

            nl_plotting.plot_surf_stat_map(
                fsavg5["infl_left"], mag_l, hemi="left", view="lateral",
                bg_map=fsavg5["sulc_left"], colorbar=False,
                vmax=vmax, threshold=THRESH, cmap="hot",
                axes=axes[0, col], figure=fig,
            )
            axes[0, col].set_title(cond, fontsize=12)

            nl_plotting.plot_surf_stat_map(
                fsavg5["infl_right"], mag_r, hemi="right", view="lateral",
                bg_map=fsavg5["sulc_right"], colorbar=(col == 2),
                vmax=vmax, threshold=THRESH, cmap="hot",
                axes=axes[1, col], figure=fig,
            )

        axes[0, 0].set_ylabel("left hemi", fontsize=11)
        axes[1, 0].set_ylabel("right hemi", fontsize=11)

        out_png = OUT_DIR / f"{person.lower().replace(' ', '_')}_brain_maps.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()

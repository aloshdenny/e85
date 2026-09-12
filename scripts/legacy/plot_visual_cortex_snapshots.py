"""
plot_visual_cortex_snapshots.py (v2)

Renders the 4 selected predictions (baseline, A, B, C -- from
select_visual_cortex_snapshots.py) as FOUR SEPARATE left-hemisphere PNG files,
restricted to the OFA+FFA mask, on a shared color scale so the panels are
honestly comparable (the scale is fixed across all 4 files, not
auto-ranged per image, so a flat baseline looks genuinely flat rather than
being auto-stretched to look interesting).

Native fsaverage5 resolution is used directly (10,242 vertices/hemisphere,
TRIBE's real output space) rather than upsampling to a finer mesh -- nilearn's
inflated fsaverage5 surface geometry already renders smoothly, and upsampling
the DATA would only interpolate, not add real information.

Usage:
  python scripts/plot_visual_cortex_snapshots.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets as nl_datasets
from nilearn import plotting as nl_plotting

OUT_DIR = Path("/Users/aoxo/vscode/e85/abliterated")
SEL_NPZ = OUT_DIR / "visual_cortex_snapshot_selection.npz"
SEL_JSON = OUT_DIR / "visual_cortex_snapshot_selection.json"


def main():
    sel = json.load(open(SEL_JSON))
    d = np.load(SEL_NPZ)
    face_mask = d["face_mask"]
    face_mask_lh = face_mask[:10242]

    fsavg5 = nl_datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    bg = fsavg5.get("sulc_left")

    panels = ["baseline", "A", "B", "C"]
    zone_label = {
        "baseline": "",
        "A": sel["A"].get("zone", ""),
        "B": sel["B"].get("zone", ""),
        "C": sel["C"].get("zone", ""),
    }
    titles = {
        "baseline": f"Baseline (flat)\n{sel['baseline']['file']}, mean={sel['baseline']['mean']:+.3f}",
        "A": f"Face A\n{sel['A']['file']}, mean={sel['A']['mean']:+.3f}, spike={sel['A']['spike']:.3f}\n{zone_label['A']}",
        "B": f"Face B\n{sel['B']['file']}, mean={sel['B']['mean']:+.3f}, spike={sel['B']['spike']:.3f}\n{zone_label['B']}",
        "C": f"Face C\n{sel['C']['file']}, mean={sel['C']['mean']:+.3f}, spike={sel['C']['spike']:.3f}\n{zone_label['C']}",
    }

    raw_lh = {p: d[f"pred_{p}"][:10242] for p in panels}

    # Shared colour scale across all 4, computed only over the masked
    # (OFA+FFA) vertices, so a bland baseline can't be auto-stretched to look
    # spiky and a real spike can't be flattened by an unrelated outlier.
    all_masked_vals = np.concatenate([raw_lh[p][face_mask_lh] for p in panels])
    vmin, vmax = np.percentile(all_masked_vals, [1, 99])
    print(f"shared colour scale: [{vmin:.4f}, {vmax:.4f}]")

    masked = {p: np.where(face_mask_lh, raw_lh[p], np.nan) for p in panels}

    saved = []
    for p in panels:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        nl_plotting.plot_surf_stat_map(
            fsavg5["infl_left"], masked[p],
            hemi="left", view="lateral", bg_map=bg,
            colorbar=True, cmap="hot", vmin=vmin, vmax=vmax,
            title=titles[p], axes=ax, figure=fig, bg_on_data=True,
        )
        out_png = OUT_DIR / f"visual_cortex_snapshot_{p}.png"
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_png)
        print(f"Saved -> {out_png}")

    print("\nAll 4 saved with the SAME left-hemisphere view, SAME OFA+FFA mask, "
          f"SAME colour scale [{vmin:.4f}, {vmax:.4f}].")
    return saved


if __name__ == "__main__":
    main()

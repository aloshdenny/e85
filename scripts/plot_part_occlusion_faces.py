"""
plot_part_occlusion_faces.py

Part-occlusion face diagrams for the OFA and FFA sections of the blog.

WHAT THIS SHOWS. Each of the seven face parts was occluded one at a time and
the change in that ROI's predicted response was measured. A part whose
occlusion moves the ROI a lot is a part that ROI cares about. Plotting those
seven numbers ON a face makes the "does this region have sharp opinions about
facial features" question visual, which a table of seven floats does not.

The range/mean ratio printed per panel is the blog's differentiation metric:
    (max - min) / mean(|values|)
A higher ratio = the parts score more differently from each other = sharper,
more feature-selective opinions. A flat ratio near 1 = the region responds to
every part about equally.

COLOUR. One shared symmetric diverging scale per figure, NOT per panel, so
before/after are honestly comparable -- including the sign flip (baseline
occlusion deltas are mostly negative, post-finetune mostly positive). Values
are also printed as text so no precision is lost to the colour mapping.

Usage:
  python scripts/plot_part_occlusion_faces.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "abliterated" / "part_occlusion"

PARTS = ["forehead", "eyebrows", "eyes", "nose", "lips", "cheeks", "jawline"]

# ---------------------------------------------------------------- measured data
# Verified from the occlusion runs (baseline = frozen readout, finetuned =
# rank-16 scale-matched NOD residual). FFA reproduces the blog's recalculated
# ratios exactly; OFA's stored values give 1.055 / 2.746 rather than the
# recalculated 5.63 / 4.98 -- swap OFA's finetuned dicts here if the re-run
# supersedes them.
DATA = {
    ("OFA", "general", "before"): dict(
        cheeks=-0.00013, eyebrows=-0.00401, eyes=-0.00302, forehead=-0.00169,
        jawline=-0.00229, lips=0.00004, nose=-0.00247),
    ("OFA", "general", "after"): dict(
        cheeks=0.00209, eyebrows=0.00192, eyes=0.00093, forehead=0.00138,
        jawline=0.00129, lips=0.00090, nose=0.00070),
    ("OFA", "target", "before"): dict(
        cheeks=0.00227, eyebrows=-0.00168, eyes=-0.00163, forehead=-0.00092,
        jawline=-0.00123, lips=-0.00075, nose=-0.00182),
    ("OFA", "target", "after"): dict(
        cheeks=0.00206, eyebrows=0.00024, eyes=-0.00014, forehead=0.00184,
        jawline=0.00070, lips=0.00024, nose=0.00040),
    ("FFA", "general", "before"): dict(
        cheeks=-0.00241, eyebrows=-0.00317, eyes=-0.00220, forehead=-0.00082,
        jawline=-0.00189, lips=-0.00044, nose=-0.00157),
    ("FFA", "general", "after"): dict(
        cheeks=-0.00144, eyebrows=0.00075, eyes=0.00002, forehead=0.00043,
        jawline=0.00009, lips=0.00107, nose=0.00053),
    ("FFA", "target", "before"): dict(
        cheeks=-0.00135, eyebrows=-0.00176, eyes=-0.00217, forehead=-0.00052,
        jawline=-0.00164, lips=-0.00169, nose=-0.00169),
    ("FFA", "target", "after"): dict(
        cheeks=-0.00027, eyebrows=-0.00013, eyes=-0.00018, forehead=0.00047,
        jawline=-0.00077, lips=0.00011, nose=0.00017),
}


def range_mean_ratio(vals):
    v = np.array([vals[p] for p in PARTS], dtype=float)
    return (v.max() - v.min()) / max(np.abs(v).mean(), 1e-12)


def draw_face(ax, vals, cmap, norm):
    """Schematic frontal face with the seven occlusion regions shaded."""
    ax.set_xlim(-1.52, 1.52)
    ax.set_ylim(-1.58, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")

    def c(part):
        return cmap(norm(vals[part]))

    # face outline
    ax.add_patch(Ellipse((0, -0.05), 1.72, 2.36, facecolor="#f2ece6",
                         edgecolor="#4a4a4a", lw=1.6, zorder=1))
    # jawline: lower rim band, drawn as a thick arc-ish wedge
    th = np.linspace(np.pi * 1.04, np.pi * 1.96, 120)
    jx, jy = 0.845 * np.cos(th), -0.05 + 1.16 * np.sin(th)
    ax.add_patch(Polygon(np.column_stack([jx, jy]), closed=False,
                         fill=False, edgecolor=c("jawline"), lw=13,
                         capstyle="round", zorder=2))
    # forehead band
    ax.add_patch(Ellipse((0, 0.74), 1.30, 0.66, facecolor=c("forehead"),
                         edgecolor="none", alpha=0.95, zorder=3))
    # cheeks
    for sx in (-1, 1):
        ax.add_patch(Ellipse((sx * 0.52, -0.30), 0.50, 0.60,
                             facecolor=c("cheeks"), edgecolor="none",
                             alpha=0.95, zorder=3))
    # eyebrows
    for sx in (-1, 1):
        ax.add_patch(FancyBboxPatch((sx * 0.30 - 0.22, 0.32), 0.44, 0.10,
                                    boxstyle="round,pad=0.02",
                                    facecolor=c("eyebrows"), edgecolor="none",
                                    zorder=4))
    # eyes
    for sx in (-1, 1):
        ax.add_patch(Ellipse((sx * 0.31, 0.14), 0.36, 0.20,
                             facecolor=c("eyes"), edgecolor="#4a4a4a",
                             lw=0.7, zorder=4))
    # nose
    ax.add_patch(Polygon([(0, 0.10), (-0.16, -0.46), (0.16, -0.46)],
                         closed=True, facecolor=c("nose"),
                         edgecolor="#4a4a4a", lw=0.7, zorder=4))
    # lips
    ax.add_patch(Ellipse((0, -0.72), 0.58, 0.22, facecolor=c("lips"),
                         edgecolor="#4a4a4a", lw=0.7, zorder=4))

    # value labels
    lbl = {
        "forehead": (0, 1.24), "eyebrows": (-1.19, 0.52), "eyes": (1.21, 0.20),
        "nose": (1.17, -0.52), "lips": (0, -1.10), "cheeks": (-1.21, -0.40),
        "jawline": (0, -1.44),
    }
    for p, (lx, ly) in lbl.items():
        ax.text(lx, ly, f"{p}\n{vals[p]:+.5f}", ha="center", va="center",
                fontsize=6.6, color="#222", linespacing=1.25, zorder=6)


def figure_for(roi, out_name):
    rows = ["general", "target"]
    cols = ["before", "after"]
    allv = np.concatenate([[DATA[(roi, r, c)][p] for p in PARTS]
                           for r in rows for c in cols])
    lim = float(np.abs(allv).max())
    norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.8))
    fig.suptitle(f"{roi} — which face parts the region actually cares about",
                 fontsize=14, y=0.975)

    row_label = {"general": "general faces", "target": "our targets' faces"}
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            vals = DATA[(roi, r, c)]
            ax = axes[i, j]
            draw_face(ax, vals, cmap, norm)
            ratio = range_mean_ratio(vals)
            state = "baseline" if c == "before" else "after fine-tune"
            ax.set_title(f"{row_label[r]} · {state}\nspread ratio = {ratio:.2f}",
                         fontsize=10.5, pad=6)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.045, pad=0.05, aspect=42)
    cbar.set_label("change in predicted response when this part is hidden",
                   fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")
    for r in rows:
        for c in cols:
            print(f"   {roi:3s} {r:8s} {c:6s} ratio={range_mean_ratio(DATA[(roi,r,c)]):.3f}")


def main():
    figure_for("OFA", "ofa_part_occlusion.png")
    figure_for("FFA", "ffa_part_occlusion.png")


if __name__ == "__main__":
    main()

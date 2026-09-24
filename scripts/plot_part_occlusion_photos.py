"""
plot_part_occlusion_photos.py

Same figure as plot_part_occlusion_faces.py, but drawn on REAL photographs
instead of a schematic face.

WHY THIS IS THE FAITHFUL VERSION. `part_masks()` below is copied verbatim from
the experiment's own occlusion code (scripts/legacy/part_occlusion_map.py), so
the regions shaded here are exactly the regions that were blurred to produce
the numbers -- convex hulls of the landmark groups for eyes/eyebrows/nose/lips,
a dilated polyline for the jaw, and the two derived regions (a band above the
brow line for forehead, eye-corner-to-jaw wedges for cheeks) that have no
landmarks of their own. A hand-drawn cartoon can only approximate that.

LANDMARKS. The original used dlib's 68-point predictor. dlib isn't installed
locally and the .dat lives on the server, so this uses insightface buffalo_l's
`1k3d68` model, which outputs the same standard 68-point iBUG layout (jaw
0-16, brows 17-26, nose 27-35, eyes 36-47, mouth 48-67) that IDX indexes into.

PRESENTATION. Only the post-fine-tune panels are drawn, and nothing is
painted on top of the face -- earlier versions shaded the regions in place,
which buried the photograph under seven translucent polygons. Instead each
part gets a callout outside the head, coloured by its value, with an arrow
pointing at the structure it refers to. The face stays fully visible; the
colour scale is shared across both panels of a figure so the two are
comparable.

Usage:
  python scripts/plot_part_occlusion_photos.py
"""
import zipfile
import warnings
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "abliterated" / "part_occlusion"

# Faces chosen by measured frontality + forehead headroom (see scan in history):
# the forehead band reaches 0.42 x face-height above the brow line, so a tight
# crop without headroom would clip the very region under discussion.
GENERAL_SRC = (ROOT / "fairface + ffhq" / "fairface + ffhq.zip", "99829.png")
TARGET_SRC = (ROOT / "target" / "mia.zip", "mia/mia_00271.jpg")

PARTS = ["forehead", "eyebrows", "eyes", "nose", "lips", "cheeks", "jawline"]
# draw order: big derived regions first, fine features last, so overlaps
# resolve in favour of the smaller, more specific part
DRAW_ORDER = ["forehead", "cheeks", "jawline", "nose", "eyes", "eyebrows", "lips"]

IDX = {
    "jaw": list(range(0, 17)),
    "eyebrows": list(range(17, 27)),
    "nose": list(range(27, 36)),
    "eyes": list(range(36, 48)),
    "lips": list(range(48, 68)),
}

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


def part_masks(shape, pts, split_forehead=False):
    """VERBATIM from scripts/legacy/part_occlusion_map.py -- do not 'improve'.
    Binary masks per named part. Filled convex hulls for compact parts; the
    jaw is a dilated polyline, since the hull of its landmarks would swallow the
    entire lower face. Forehead and cheeks are derived, having no landmarks."""
    h, w = shape
    face_w = float(np.ptp(pts[:, 0])) or 1.0
    face_h = float(np.ptp(pts[:, 1])) or 1.0
    out = {}

    for name in ["eyes", "eyebrows", "nose", "lips"]:
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(pts[IDX[name]].astype(np.int32)), 255)
        m = cv2.dilate(m, np.ones((max(3, int(0.03 * face_w)),) * 2, np.uint8))
        out[name] = m

    m = np.zeros((h, w), np.uint8)
    cv2.polylines(m, [pts[IDX["jaw"]].astype(np.int32)], False, 255,
                  thickness=max(3, int(0.09 * face_w)))
    out["jawline"] = m

    brow = pts[IDX["eyebrows"]]
    top = brow[:, 1].min()
    x0, x1 = int(brow[:, 0].min()), int(brow[:, 0].max())

    def band_rect(lo_frac, hi_frac):
        m = np.zeros((h, w), np.uint8)
        yb = int(top - lo_frac * face_h)
        ya = int(top - hi_frac * face_h)
        cv2.rectangle(m, (max(0, x0), max(0, ya)), (min(w - 1, x1), max(0, yb)), 255, -1)
        return m

    if split_forehead:
        out["forehead_low"] = band_rect(0.05, 0.17)
        out["forehead_mid"] = band_rect(0.17, 0.29)
        out["forehead_high"] = band_rect(0.29, 0.42)
    else:
        out["forehead"] = band_rect(0.05, 0.42)

    m = np.zeros((h, w), np.uint8)
    for eye_out, jaw_i in [(36, 3), (45, 13)]:
        p_eye, p_jaw, p_nose = pts[eye_out], pts[jaw_i], pts[33]
        poly = np.array([p_eye, p_jaw, p_nose, [p_nose[0], p_eye[1]]], np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(poly), 255)
    out["cheeks"] = m
    return out


def range_mean_ratio(vals):
    v = np.array([vals[p] for p in PARTS], dtype=float)
    return (v.max() - v.min()) / max(np.abs(v).mean(), 1e-12)


def load_face(src, app, upscale=3):
    zpath, member = src
    with zipfile.ZipFile(zpath) as z:
        img = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not decode {member}")
    faces = app.get(img)
    if not faces:
        raise SystemExit(f"no face detected in {member}")
    pts = np.array(faces[0].landmark_3d_68)[:, :2]
    # upscale so the mask outlines and overlaid text render crisply; landmarks
    # scale with the image, masks are rebuilt at the final resolution
    img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    pts = pts * upscale
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb, pts


def label_style(val, cmap, norm):
    """Box colour from the value, text colour from that box's luminance."""
    rgba = cmap(norm(val))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return rgba, ("black" if lum > 0.55 else "white")


def arrow_anchors(masks, pts):
    """Where each arrow should POINT -- the structure itself, on the face."""
    ys, xs = np.nonzero(masks["forehead"])
    forehead = (xs.mean(), ys.mean()) if len(xs) else (pts[:, 0].mean(),
                                                       pts[:, 1].min())
    return {
        "forehead": forehead,
        "eyebrows": tuple(pts[19]),
        "eyes":     tuple(pts[42:48].mean(axis=0)),
        "cheeks":   (np.mean([pts[36][0], pts[3][0], pts[33][0]]),
                     np.mean([pts[36][1], pts[3][1], pts[33][1]])),
        "nose":     tuple(pts[33]),
        "lips":     tuple(pts[48:68].mean(axis=0)),
        "jawline":  tuple(pts[5]),
    }


# Callout placement, in fractions of image width/height. Split left/right and
# ordered top-to-bottom to match each part's own vertical position, so no two
# arrows cross.
CALLOUTS = {
    "forehead": (-0.04, 0.02, "right"),
    "eyebrows": (-0.04, 0.27, "right"),
    "cheeks":   (-0.04, 0.58, "right"),
    "jawline":  (-0.04, 0.89, "right"),
    "eyes":     (1.04, 0.09, "left"),
    "nose":     (1.04, 0.43, "left"),
    "lips":     (1.04, 0.78, "left"),
}


def figure_for(roi, faces, out_name):
    rows = ["general", "target"]
    allv = np.concatenate([[DATA[(roi, r, "after")][p] for p in PARTS]
                           for r in rows])
    lim = float(np.abs(allv).max())
    norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")

    # Margins reserve room for the callouts; the figure is then sized to that
    # exact aspect, because imshow axes hold a fixed data aspect and any
    # mismatch shows up as a band of dead whitespace above and below.
    MX, MY = 0.47, 0.17
    panel_aspect = (1 + 2 * MX) / (1 + 2 * MY)
    # Solve for the width that makes the axes boxes exactly the aspect the
    # image data needs, given the margins below -- otherwise the leftover
    # slack appears as a gap between the two panels.
    L, R, TOP, BOT, WS = 0.015, 0.985, 0.86, 0.17, 0.04
    frac_h = TOP - BOT
    frac_w = (R - L) / (1 + WS / 2)
    fig_h = 6.05
    fig_w = 2 * panel_aspect * frac_h * fig_h / frac_w
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT, wspace=WS)
    fig.suptitle(f"{roi} — which face parts the region cares about, "
                 f"after fine-tuning", fontsize=15, y=0.985)
    row_label = {"general": "general faces", "target": "our targets' faces"}

    for ax, r in zip(axes, rows):
        rgb, pts = faces[r]
        H, W = rgb.shape[:2]
        masks = part_masks((H, W), pts)
        anchors = arrow_anchors(masks, pts)
        vals = DATA[(roi, r, "after")]

        ax.imshow(rgb)
        ax.set_xlim(-MX * W, (1 + MX) * W)
        ax.set_ylim((1 + MY) * H, -MY * H)
        ax.axis("off")

        for part, (fx, fy, ha) in CALLOUTS.items():
            box_rgba, txt_col = label_style(vals[part], cmap, norm)
            ax.annotate(
                f"{part}\n{vals[part]:+.5f}",
                xy=anchors[part], xycoords="data",
                xytext=(fx * W, fy * H), textcoords="data",
                ha=ha, va="center", fontsize=9.2, color=txt_col,
                fontweight="bold", linespacing=1.3, zorder=5,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=box_rgba,
                          edgecolor="#333", linewidth=0.8),
                arrowprops=dict(arrowstyle="-|>", color="#222", linewidth=1.5,
                                shrinkA=3, shrinkB=3,
                                connectionstyle="arc3,rad=0.0"),
            )

        ax.set_title(f"{row_label[r]}\nspread ratio = "
                     f"{range_mean_ratio(vals):.2f}", fontsize=11.5, pad=8)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.055, pad=0.06, aspect=55)
    cbar.set_label("change in predicted response when this part is blurred",
                   fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}  (spread: "
          + ", ".join(f"{r}={range_mean_ratio(DATA[(roi, r, 'after')]):.2f}"
                      for r in rows) + ")")


def main():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = {"general": load_face(GENERAL_SRC, app),
             "target": load_face(TARGET_SRC, app)}
    for r, (rgb, pts) in faces.items():
        print(f"  {r}: image {rgb.shape[1]}x{rgb.shape[0]}, 68 landmarks ok")
    figure_for("OFA", faces, "ofa_part_occlusion_arrows.png")
    figure_for("FFA", faces, "ffa_part_occlusion_arrows.png")


if __name__ == "__main__":
    main()

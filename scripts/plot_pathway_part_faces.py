"""
plot_pathway_part_faces.py

One figure covering general faces vs Johnny Sins vs Mia Khalifa, showing which
facial features drive each stage of the face pathway: OFA -> FFA -> vATL.

INPUT. The .npy row arrays from part_occlusion_map_v2.py (one per arm). Each
row is (is_target, "part:<name>"|"ctrl:N", area_fraction, {roi: delta}).
Effects are area-adjusted exactly the way part_occlusion_report.py does it --
the random area-matched controls are regressed on area within the same arm and
subtracted -- because a bigger blurred region perturbs the image more whatever
it covers, so a raw part effect partly just measures the part's size.

COLOUR IS PER ROW, NOT GLOBAL. vATL's effects are roughly an order of
magnitude smaller than OFA's, so a single shared scale would render the whole
vATL row as flat nothing. Each region therefore gets its own symmetric scale,
and the figure answers "which parts matter most WITHIN this region" -- not
"which region responds hardest", which the colours deliberately cannot say.

Usage:
  python scripts/plot_pathway_part_faces.py
"""
import zipfile
import warnings
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OCC = ROOT / "abliterated" / "part_occlusion"
OUT_DIR = OCC

PARTS = ["forehead", "eyebrows", "eyes", "nose", "lips", "cheeks", "jawline"]
DRAW_ORDER = ["forehead", "cheeks", "jawline", "nose", "eyes", "eyebrows", "lips"]
REGIONS = [("OFA", "OFA\ndetects facial parts"),
           ("FFA", "FFA\nidentifies the face"),
           ("ATL_TP", "vATL\nlinks it to a person")]

# (column label, npy file, is_target flag to select, face image source)
COLUMNS = [
    ("general faces", "part_occlusion_general.npy", False,
     (ROOT / "fairface + ffhq" / "fairface + ffhq.zip", "99829.png")),
    ("Johnny Sins", "part_occlusion_sins.npy", True,
     (ROOT / "target" / "sins.zip", "sins/sins_00075.jpg")),
    ("Mia Khalifa", "part_occlusion_mia.npy", True,
     (ROOT / "target" / "mia.zip", "mia/mia_00271.jpg")),
]

IDX = {"jaw": list(range(0, 17)), "eyebrows": list(range(17, 27)),
       "nose": list(range(27, 36)), "eyes": list(range(36, 48)),
       "lips": list(range(48, 68))}


def part_masks(shape, pts):
    """VERBATIM from the experiment's own part_occlusion_map.py, so the shaded
    regions are the regions that were actually blurred."""
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
    mm = np.zeros((h, w), np.uint8)
    yb, ya = int(top - 0.05 * face_h), int(top - 0.42 * face_h)
    cv2.rectangle(mm, (max(0, x0), max(0, ya)), (min(w - 1, x1), max(0, yb)), 255, -1)
    out["forehead"] = mm
    m = np.zeros((h, w), np.uint8)
    for eye_out, jaw_i in [(36, 3), (45, 13)]:
        p_eye, p_jaw, p_nose = pts[eye_out], pts[jaw_i], pts[33]
        poly = np.array([p_eye, p_jaw, p_nose, [p_nose[0], p_eye[1]]], np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(poly), 255)
    out["cheeks"] = m
    return out


def adjusted(rows, want_target, with_sem=False):
    """Per-part effect, area-adjusted against the arm's own random controls.

    with_sem also returns the standard error, so the figure can distinguish a
    part that genuinely drives a region from one that merely has a large mean
    over 40 noisy images. This project has been burned repeatedly by numbers
    that looked convincing until someone asked for the spread."""
    sel = (lambda t: bool(t)) if want_target else (lambda t: not bool(t))
    ctrl = [(a, d) for t, k, a, d in rows if str(k).startswith("ctrl") and sel(t)]
    rois = list(ctrl[0][1].keys())
    out = {}
    for roi in rois:
        A = np.array([[a, 1.0] for a, _ in ctrl])
        y = np.array([d[roi] for _, d in ctrl])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        for p in PARTS:
            v = [(a, d[roi]) for t, k, a, d in rows
                 if str(k) == f"part:{p}" and sel(t)]
            if v:
                arr = np.array([x - (coef[0] * a + coef[1]) for a, x in v])
                if with_sem:
                    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                    out.setdefault(p, {})[roi] = (float(arr.mean()), sem)
                else:
                    out.setdefault(p, {})[roi] = float(arr.mean())
    return out


def load_face(src, app, upscale=3):
    zpath, member = src
    with zipfile.ZipFile(zpath) as z:
        img = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_COLOR)
    faces = app.get(img)
    if not faces:
        raise SystemExit(f"no face detected in {member}")
    f0 = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pts = np.array(f0.landmark_3d_68, dtype=np.float32)[:, :2] * upscale
    img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), pts


def shade(rgb, masks, vals, cmap, norm, sig=None):
    """Significant parts (|mean| > 2 sem) are painted at full strength; the
    rest are left faint, so the eye is not drawn to noise."""
    out = rgb.astype(np.float32) / 255.0
    for part in DRAW_ORDER:
        m = masks[part] > 0
        if not m.any() or part not in vals:
            continue
        alpha = 0.78 if (sig is None or sig.get(part)) else 0.22
        col = np.array(cmap(norm(vals[part]))[:3], dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * col
        edges = cv2.dilate(masks[part], np.ones((3, 3), np.uint8)) - masks[part]
        out[edges > 0] = 0.10
    return np.clip(out, 0, 1)


def main():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    effects, faces = {}, {}
    for label, fname, is_t, src in COLUMNS:
        path = OCC / fname
        if not path.exists():
            raise SystemExit(f"missing {path} -- run the Modal occlusion job first")
        effects[label] = adjusted(np.load(path, allow_pickle=True), is_t,
                                  with_sem=True)
        faces[label] = load_face(src, app)
        print(f"  {label}: {fname} ok")

    cmap = plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 11.6))
    fig.subplots_adjust(hspace=0.12, wspace=0.02, top=0.885,
                        bottom=0.02, left=0.06, right=0.90)
    fig.suptitle("Which facial features drive each stage of the face pathway",
                 fontsize=16, y=0.972)
    fig.text(0.5, 0.935, "solid = the region genuinely responds to that part "
             "(|mean| > 2\u00d7s.e.m.);  faded = within noise.  "
             "Colour scale is per row, so rows show which part matters most "
             "WITHIN a region, not which region responds hardest.",
             ha="center", fontsize=8.6, color="#444")

    for i, (roi, roi_label) in enumerate(REGIONS):
        # one scale per REGION -- see module docstring
        allv = [effects[c[0]][p][roi][0] for c in COLUMNS for p in PARTS]
        lim = max(abs(np.array(allv)).max(), 1e-9)
        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        for j, (label, _, _, _) in enumerate(COLUMNS):
            rgb, pts = faces[label]
            masks = part_masks(rgb.shape[:2], pts)
            vals = {p: effects[label][p][roi][0] for p in PARTS}
            sig = {p: abs(effects[label][p][roi][0]) > 2 * effects[label][p][roi][1]
                   for p in PARTS}
            ax = axes[i, j]
            ax.imshow(shade(rgb, masks, vals, cmap, norm, sig))
            strongest = max(PARTS, key=lambda p: abs(vals[p]) if sig[p] else -1)
            if sig[strongest]:
                ax.text(0.5, -0.045, strongest, transform=ax.transAxes,
                        ha="center", va="top", fontsize=11.5,
                        fontweight="bold", color="#8c2d16")
            ax.axis("off")
            if i == 0:
                ax.set_title(label, fontsize=13, pad=8)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[i, :].tolist(), fraction=0.020, pad=0.012)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("weaker → stronger", fontsize=8)
        axes[i, 0].text(-0.10, 0.5, roi_label, transform=axes[i, 0].transAxes,
                        ha="center", va="center", rotation=90, fontsize=12,
                        fontweight="bold", linespacing=1.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "pathway_part_faces.png"
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

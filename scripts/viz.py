"""
viz.py

Render brain-surface maps for the target person vs the merged FairFace+FFHQ
general population, plus an interactive HTML toggle viewer.

Inputs (flat merged layout):
  fairface + ffhq preds/{n}.npz  -- per-image preds (20484,) + filename
  target_preds/*.npz               -- multi-image or per-image

Outputs:
  fairface_study/group_maps_sagittal/general.png
  target_study/target_map_sagittal.png
  interactive_study/combined_interactive.html
"""

import os
import warnings
import logging
import base64
import json
import shutil

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
from pathlib import Path
from nilearn import datasets, surface
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chunk_utils import discover_npz, load_npz, preds_as_image_vectors

# ── Paths ──────────────────────────────────────────────────────────────────────

PRED_ROOT        = Path("./fairface + ffhq preds")
TARGET_PRED_ROOT = Path("./target_preds")
FAIRFACE_OUT     = Path("./fairface_study")
TARGET_OUT       = Path("./target_study")
INTERACTIVE_OUT  = Path("./interactive_study")

for d in (FAIRFACE_OUT, TARGET_OUT, INTERACTIVE_OUT):
    d.mkdir(parents=True, exist_ok=True)

N_LH5 = 10242  # fsaverage5 vertices per hemisphere (native prediction resolution)


def load_group_mean(preds_dir: Path, recursive=False, label="group"):
    """Streaming mean over all npzs. Handles 1D per-image and 2D multi-image."""
    files = discover_npz(preds_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {preds_dir}")

    first = preds_as_image_vectors(load_npz(files[0])["preds"])
    print(
        f"  {label}: "
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
    print(f"  {label}: {n_images} images -> mean shape {mean.shape}")
    return mean, len(files), n_images


print(f"Loading general population from {PRED_ROOT} ...")
GENERAL_MEAN, n_gen_files, n_gen_images = load_group_mean(PRED_ROOT, label="general")

print(f"Loading target from {TARGET_PRED_ROOT} ...")
TARGET_MEAN, n_tgt_files, n_tgt_images = load_group_mean(
    TARGET_PRED_ROOT, recursive=True, label="target"
)

ALL_NATIVE = {
    "general": GENERAL_MEAN,
    "__target__": TARGET_MEAN,
}

# ── Masks (native fsaverage5) ────────────────────────────────────────────────

def make_mask(data, z_thresh=1.0):
    mu, sd = data.mean(), data.std()
    return data > (mu + z_thresh * sd)

ff_mask_dir = FAIRFACE_OUT / "masks"
ff_mask_dir.mkdir(exist_ok=True)
np.save(ff_mask_dir / "general.npy", make_mask(GENERAL_MEAN))
print(f"General mask saved to {ff_mask_dir}")

tgt_mask_dir = TARGET_OUT / "masks"
tgt_mask_dir.mkdir(exist_ok=True)
target_mask = make_mask(TARGET_MEAN)
np.save(tgt_mask_dir / "target.npy", target_mask)
print(f"Target mask saved to {tgt_mask_dir} "
      f"({target_mask.sum()} vertices, LH={target_mask[:N_LH5].sum()}, RH={target_mask[N_LH5:].sum()})")

# ── Meshes ────────────────────────────────────────────────────────────────────

print("Fetching surface meshes (fsaverage, fsaverage5)...")
fsaverage  = datasets.fetch_surf_fsaverage(mesh="fsaverage")
fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

lhF_coords, lhF_faces = surface.load_surf_mesh(fsaverage["pial_left"])
rhF_coords, rhF_faces = surface.load_surf_mesh(fsaverage["pial_right"])
N_LH_FULL = lhF_coords.shape[0]
lhF_sulc = normalize(surface.load_surf_data(fsaverage["sulc_left"]))
rhF_sulc = normalize(surface.load_surf_data(fsaverage["sulc_right"]))


def build_idw(src_sphere, dst_sphere, k=8):
    tree = cKDTree(src_sphere)
    dist, idx = tree.query(dst_sphere, k=k)
    w = 1.0 / (dist ** 2 + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    return idx, w

lh5_sphere, _ = surface.load_surf_mesh(fsaverage5["sphere_left"])
rh5_sphere, _ = surface.load_surf_mesh(fsaverage5["sphere_right"])
lhF_sphere, _ = surface.load_surf_mesh(fsaverage["sphere_left"])
rhF_sphere, _ = surface.load_surf_mesh(fsaverage["sphere_right"])

print("Building IDW interpolants (fsaverage5 -> fsaverage full-res, for PNGs)...")
lhF_idx, lhF_w = build_idw(lh5_sphere, lhF_sphere)
rhF_idx, rhF_w = build_idw(rh5_sphere, rhF_sphere)

def upsample(data_1d, lh_idx, lh_w, rh_idx, rh_w):
    lh5 = data_1d[:N_LH5]
    rh5 = data_1d[N_LH5:]
    lh_full = (lh5[lh_idx] * lh_w).sum(axis=1)
    rh_full = (rh5[rh_idx] * rh_w).sum(axis=1)
    return np.concatenate([lh_full, rh_full])

print("Upsampling maps to full fsaverage...")
MAPS_FULLRES = {k: upsample(v, lhF_idx, lhF_w, rhF_idx, rhF_w) for k, v in ALL_NATIVE.items()}

# ── Colormap + blending ───────────────────────────────────────────────────────

hot = plt.get_cmap("hot")

def blend_activation_onto_sulc(sulc_norm, activation, threshold_pct=85):
    thresh = np.nanpercentile(np.abs(activation), threshold_pct)
    vmax   = np.nanpercentile(np.abs(activation), 99)
    r_base = (200 + sulc_norm * 55).astype(float)
    g_base = (200 + sulc_norm * 55).astype(float)
    b_base = (200 + sulc_norm * 55).astype(float)
    colors = []
    for idx in range(len(sulc_norm)):
        val = activation[idx]
        if np.isnan(val) or abs(val) < thresh:
            colors.append(f"rgb({int(r_base[idx])},{int(g_base[idx])},{int(b_base[idx])})")
        else:
            t = float(np.clip((abs(val) - thresh) / (vmax - thresh + 1e-9), 0, 1))
            rc, gc, bc, _ = hot(t)
            colors.append(f"rgb({int(rc*255)},{int(gc*255)},{int(bc*255)})")
    return colors

lighting = dict(ambient=0.75, diffuse=0.7, specular=0.05, roughness=0.8, fresnel=0.1)
lightposition = dict(x=100, y=200, z=300)

lh_camera = dict(eye=dict(x=-2.4, y=0.0, z=0.05), up=dict(x=0, y=0, z=1),
                  projection=dict(type="orthographic"))
rh_camera = dict(eye=dict(x=2.4, y=0.0, z=0.05), up=dict(x=0, y=0, z=1),
                  projection=dict(type="orthographic"))

def scene_kwargs(camera, domain_x):
    return dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                bgcolor="#0d0d0d", aspectmode="data", camera=camera,
                domain=dict(x=domain_x, y=[0, 1]))

def render_png(cname, data_full, out_path):
    lh_vc = blend_activation_onto_sulc(lhF_sulc, data_full[:N_LH_FULL])
    rh_vc = blend_activation_onto_sulc(rhF_sulc, data_full[N_LH_FULL:])

    snap = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                          horizontal_spacing=0.0)
    snap.add_trace(go.Mesh3d(x=lhF_coords[:, 0], y=lhF_coords[:, 1], z=lhF_coords[:, 2],
                              i=lhF_faces[:, 0], j=lhF_faces[:, 1], k=lhF_faces[:, 2],
                              vertexcolor=lh_vc, lighting=lighting, lightposition=lightposition,
                              hoverinfo="skip", showscale=False), row=1, col=1)
    snap.add_trace(go.Mesh3d(x=rhF_coords[:, 0], y=rhF_coords[:, 1], z=rhF_coords[:, 2],
                              i=rhF_faces[:, 0], j=rhF_faces[:, 1], k=rhF_faces[:, 2],
                              vertexcolor=rh_vc, lighting=lighting, lightposition=lightposition,
                              hoverinfo="skip", showscale=False), row=1, col=2)
    snap.update_layout(scene=scene_kwargs(lh_camera, [0.0, 0.5]), scene2=scene_kwargs(rh_camera, [0.5, 1.0]),
                        paper_bgcolor="black", margin=dict(l=0, r=0, t=0, b=0),
                        width=3200, height=1600, showlegend=False)
    snap.add_annotation(text=f"<b>{cname}</b>", x=0.985, y=0.02, xref="paper", yref="paper",
                         xanchor="right", yanchor="bottom", showarrow=False,
                         font=dict(size=44, color="white"))
    snap.write_image(str(out_path), scale=4, engine="kaleido")
    print("  Saved:", out_path)

print("Rendering general + target PNGs (full fsaverage resolution)...")
ff_png_dir = FAIRFACE_OUT / "group_maps_sagittal"
ff_png_dir.mkdir(exist_ok=True)
render_png(
    f"general ({n_gen_images} imgs)",
    MAPS_FULLRES["general"],
    ff_png_dir / "general.png",
)
render_png(
    f"target ({n_tgt_images} imgs)",
    MAPS_FULLRES["__target__"],
    TARGET_OUT / "target_map_sagittal.png",
)

# ── Interactive viewer: target vs general toggle ─────────────────────────────

general_vec = MAPS_FULLRES["general"].astype(np.float32)
target_vec = MAPS_FULLRES["__target__"].astype(np.float32)

gmin = float(min(general_vec.min(), target_vec.min()))
gmax = float(max(general_vec.max(), target_vec.max()))
scale = (gmax - gmin) / 255.0 if gmax > gmin else 1.0

def quantize(v):
    return np.clip(np.round((v - gmin) / scale), 0, 255).astype(np.uint8)

general_q = quantize(general_vec)
target_q = quantize(target_vec)

def b64_bytes(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")

def b64_float32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")

def b64_int32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.int32).tobytes()).decode("ascii")

embedded = {
    "n_lh": int(N_LH_FULL),
    "n_rh": int(rhF_coords.shape[0]),
    "gmin": gmin,
    "scale": scale,
    "n_general_images": int(n_gen_images),
    "n_target_images": int(n_tgt_images),
    "general_b64": b64_bytes(general_q),
    "target_b64": b64_bytes(target_q),
    "lh_sulc_b64": b64_float32(lhF_sulc),
    "rh_sulc_b64": b64_float32(rhF_sulc),
    "lh_coords_b64": b64_float32(lhF_coords),
    "rh_coords_b64": b64_float32(rhF_coords),
    "lh_faces_b64": b64_int32(lhF_faces),
    "rh_faces_b64": b64_int32(rhF_faces),
}

embedded_json = json.dumps(embedded)
print(f"Total embedded JSON size: {len(embedded_json) / 1e6:.1f} MB")

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TRIBE v2 — Target vs FairFace+FFHQ</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { margin:0; background:#0d0d0d; color:#eee; font-family: -apple-system, Helvetica, Arial, sans-serif; }
  #controls { padding: 14px 20px; background:#151515; border-bottom:1px solid #333; }
  #plot { width:100vw; height:86vh; }
  #toggle label { margin-right:18px; font-size:14px; cursor:pointer; }
  h2 { margin: 0 0 10px 0; font-size:16px; font-weight:600; color:#fff; }
  #meta { font-size:12px; color:#888; margin-top:6px; }
</style>
</head>
<body>

<div id="controls">
  <h2>TRIBE v2 — Target vs FairFace+FFHQ General Population</h2>
  <div id="toggle">
    <label><input type="radio" name="mode" value="target" checked> Target</label>
    <label><input type="radio" name="mode" value="general"> General population</label>
  </div>
  <div id="meta"></div>
</div>

<div id="plot"></div>

<script id="embedded-data" type="application/json">__EMBEDDED_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('embedded-data').textContent);

function b64ToUint8(b64) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}
function b64ToFloat32(b64) { return new Float32Array(b64ToUint8(b64).buffer); }
function b64ToInt32(b64)   { return new Int32Array(b64ToUint8(b64).buffer); }

const generalQ   = b64ToUint8(DATA.general_b64);
const targetQ    = b64ToUint8(DATA.target_b64);
const lhSulc     = b64ToFloat32(DATA.lh_sulc_b64);
const rhSulc     = b64ToFloat32(DATA.rh_sulc_b64);
const lhCoords   = b64ToFloat32(DATA.lh_coords_b64);
const rhCoords   = b64ToFloat32(DATA.rh_coords_b64);
const lhFaces    = b64ToInt32(DATA.lh_faces_b64);
const rhFaces    = b64ToInt32(DATA.rh_faces_b64);

const V = targetQ.length;
const nLH = DATA.n_lh, nRH = DATA.n_rh;
const GMIN = DATA.gmin, SCALE = DATA.scale;

document.getElementById('meta').textContent =
  `target: ${DATA.n_target_images} images  |  general: ${DATA.n_general_images} images`;

function dequantize(q) {
  const out = new Float32Array(V);
  for (let v = 0; v < V; v++) out[v] = GMIN + q[v] * SCALE;
  return out;
}

function hotColor(t) {
  const r = Math.min(1, Math.max(0, 3 * t));
  const g = Math.min(1, Math.max(0, 3 * t - 1));
  const b = Math.min(1, Math.max(0, 3 * t - 2));
  return [r, g, b];
}

function percentile(sortedArr, p) {
  const idx = Math.min(sortedArr.length - 1, Math.max(0, Math.floor(p / 100 * sortedArr.length)));
  return sortedArr[idx];
}

function blendOntoSulc(sulc, activation, thresholdPct = 85) {
  const absVals = Array.from(activation, Math.abs).sort((a, b) => a - b);
  const thresh = percentile(absVals, thresholdPct);
  const vmax = percentile(absVals, 99);
  const n = sulc.length;
  const colors = new Array(n);
  for (let i = 0; i < n; i++) {
    const base = 200 + sulc[i] * 55;
    const val = Math.abs(activation[i]);
    if (val < thresh) {
      const b = Math.round(base);
      colors[i] = `rgb(${b},${b},${b})`;
    } else {
      const t = Math.min(1, Math.max(0, (val - thresh) / (vmax - thresh + 1e-9)));
      const [rc, gc, bc] = hotColor(t);
      colors[i] = `rgb(${Math.round(rc * 255)},${Math.round(gc * 255)},${Math.round(bc * 255)})`;
    }
  }
  return colors;
}

function applyActivation(fullVec) {
  const lhAct = fullVec.subarray(0, nLH);
  const rhAct = fullVec.subarray(nLH, nLH + nRH);
  const lhColors = blendOntoSulc(lhSulc, lhAct);
  const rhColors = blendOntoSulc(rhSulc, rhAct);
  Plotly.restyle('plot', { vertexcolor: [lhColors, rhColors] }, [0, 1]);
}

const lighting = { ambient: 0.75, diffuse: 0.7, specular: 0.05, roughness: 0.8, fresnel: 0.1 };
const lightposition = { x: 100, y: 200, z: 300 };

function toXYZ(flat) {
  const n = flat.length / 3;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = flat[3*i]; y[i] = flat[3*i+1]; z[i] = flat[3*i+2]; }
  return [x, y, z];
}
function toIJK(flat) {
  const n = flat.length / 3;
  const i = new Int32Array(n), j = new Int32Array(n), k = new Int32Array(n);
  for (let t = 0; t < n; t++) { i[t] = flat[3*t]; j[t] = flat[3*t+1]; k[t] = flat[3*t+2]; }
  return [i, j, k];
}

const [lhX, lhY, lhZ] = toXYZ(lhCoords);
const [rhX, rhY, rhZ] = toXYZ(rhCoords);
const [lhI, lhJ, lhK] = toIJK(lhFaces);
const [rhI, rhJ, rhK] = toIJK(rhFaces);

const initTarget = dequantize(targetQ);
const initLhColors = blendOntoSulc(lhSulc, initTarget.subarray(0, nLH));
const initRhColors = blendOntoSulc(rhSulc, initTarget.subarray(nLH, nLH + nRH));

const traceLH = {
  type: 'mesh3d', x: lhX, y: lhY, z: lhZ, i: lhI, j: lhJ, k: lhK,
  vertexcolor: initLhColors, lighting, lightposition, showscale: false, hoverinfo: 'skip',
  scene: 'scene',
};
const traceRH = {
  type: 'mesh3d', x: rhX, y: rhY, z: rhZ, i: rhI, j: rhJ, k: rhK,
  vertexcolor: initRhColors, lighting, lightposition, showscale: false, hoverinfo: 'skip',
  scene: 'scene2',
};

function sceneKwargs(eyeX, domainX) {
  return {
    xaxis: { visible: false }, yaxis: { visible: false }, zaxis: { visible: false },
    bgcolor: '#0d0d0d', aspectmode: 'data',
    camera: { eye: { x: eyeX, y: 0.0, z: 0.05 }, up: { x: 0, y: 0, z: 1 },
              projection: { type: 'orthographic' } },
    domain: { x: domainX, y: [0, 1] },
  };
}

const layout = {
  paper_bgcolor: '#0d0d0d', plot_bgcolor: '#0d0d0d', font: { color: 'white' },
  margin: { l: 0, r: 0, t: 10, b: 10 },
  scene: sceneKwargs(-2.4, [0.0, 0.5]),
  scene2: sceneKwargs(2.4, [0.5, 1.0]),
};

Plotly.newPlot('plot', [traceLH, traceRH], layout, { responsive: true });

function scheduleUpdate() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  applyActivation(dequantize(mode === 'target' ? targetQ : generalQ));
}

document.querySelectorAll('input[name=mode]').forEach(el =>
  el.addEventListener('change', scheduleUpdate)
);
</script>
</body>
</html>
"""

html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", embedded_json)

out_html = INTERACTIVE_OUT / "combined_interactive.html"
out_html.write_text(html, encoding="utf-8")
print(f"\nSaved interactive viewer -> {out_html}  ({out_html.stat().st_size / 1e6:.1f} MB)")

# Convenience copies
for dest in (FAIRFACE_OUT / "combined_interactive.html", TARGET_OUT / "combined_interactive.html"):
    shutil.copy2(out_html, dest)

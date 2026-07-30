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

from chunk_utils import discover_npz, load_npz

# ── Paths ──────────────────────────────────────────────────────────────────────
# Predictions (input):
#   fairface_preds/{category}.npz  -- 126 age_gender_race buckets
#     (or {category}_chunk_NNN.npz if >100MB)
#   target_preds/*.npz             -- one or more files for the target person
# Outputs:
#   fairface_study/  -- fairface masks + full-fsaverage PNGs
#   target_study/    -- target mask + full-fsaverage PNG
#   interactive_study/combined_interactive.html -- the toggle+slider viewer
#   (a copy of the html is also dropped into fairface_study/ and target_study/
#    for convenience)

PRED_ROOT        = Path("./fairface_preds")
TARGET_PRED_ROOT = Path("./target_preds")
FAIRFACE_OUT     = Path("./fairface_study")
TARGET_OUT       = Path("./target_study")
INTERACTIVE_OUT  = Path("./interactive_study")

for d in (FAIRFACE_OUT, TARGET_OUT, INTERACTIVE_OUT):
    d.mkdir(parents=True, exist_ok=True)

N_LH5 = 10242  # fsaverage5 vertices per hemisphere (native prediction resolution)

# ── Ordered axes for the slider grid ─────────────────────────────────────────
# Age and gender have a natural order. Race does not -- this order is an
# arbitrary (alphabetical) fixed axis purely so a continuous slider has
# something to crossfade along. Reorder this list if you want a different
# crossfade sequence (e.g. by measured similarity in activation space).

AGE_LEVELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more_than_70"]
GENDER_LEVELS = ["female", "male"]
RACE_LEVELS = ["black", "east_asian", "indian", "latino_hispanic",
               "middle_eastern", "southeast_asian", "white"]

def category_key(age, gender, race):
    # matches the on-disk category naming, e.g. "20-29_female_east_asian"
    # note: FairFace's raw label is "more than 70" -> zip/category name used
    # "_" joins, so it became "more than 70_..." in earlier steps. Handle both.
    age_norm = age.replace("more_than_70", "more than 70")
    return f"{age_norm}_{gender}_{race}"

# ── Discover + load native (fsaverage5) means ────────────────────────────────

def discover_categories():
    return sorted(p.stem for p in discover_npz(PRED_ROOT))

CATEGORIES = discover_categories()
print(f"Discovered {len(CATEGORIES)} fairface categories")

def load_category_mean(path):
    d = load_npz(path)
    preds = d["preds"]  # (N_images, 20484)
    return preds.mean(axis=0)

print("Loading fairface category means (native fsaverage5)...")
FAIRFACE_MEANS = {cat: load_category_mean(PRED_ROOT / f"{cat}.npz") for cat in CATEGORIES}

print("Loading target mean (native fsaverage5)...")
target_files = discover_npz(TARGET_PRED_ROOT, recursive=True)
if not target_files:
    raise FileNotFoundError(f"No .npz files found in {TARGET_PRED_ROOT}")
target_preds_list = [load_npz(p)["preds"] for p in target_files]
TARGET_MEAN = np.concatenate(target_preds_list, axis=0).mean(axis=0)
print(f"  target: {len(target_files)} file(s), "
      f"{sum(p.shape[0] for p in target_preds_list)} images total")

MAPS = dict(FAIRFACE_MEANS)  # category -> native fsaverage5 vector
ALL_NATIVE = dict(MAPS)
ALL_NATIVE["__target__"] = TARGET_MEAN

# ── Masks (native fsaverage5, unchanged logic) ───────────────────────────────

def make_mask(data, z_thresh=1.0):
    mu, sd = data.mean(), data.std()
    return data > (mu + z_thresh * sd)

ff_mask_dir = FAIRFACE_OUT / "masks"
ff_mask_dir.mkdir(exist_ok=True)
for name, data in FAIRFACE_MEANS.items():
    mask = make_mask(data)
    np.save(ff_mask_dir / f"{name.lower().replace(' ', '_')}.npy", mask)
print(f"Fairface masks saved to {ff_mask_dir}")

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

# Full-res mesh
lhF_coords, lhF_faces = surface.load_surf_mesh(fsaverage["pial_left"])
rhF_coords, rhF_faces = surface.load_surf_mesh(fsaverage["pial_right"])
N_LH_FULL = lhF_coords.shape[0]
lhF_sulc = normalize(surface.load_surf_data(fsaverage["sulc_left"]))
rhF_sulc = normalize(surface.load_surf_data(fsaverage["sulc_right"]))

# ── IDW upsampling helpers (fsaverage5 native -> target mesh) ────────────────

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

print("Upsampling all maps to full fsaverage (for PNGs)...")
MAPS_FULLRES = {cat: upsample(v, lhF_idx, lhF_w, rhF_idx, rhF_w) for cat, v in ALL_NATIVE.items()}

print("Using full fsaverage maps for the interactive viewer...")
MAPS_INTERACTIVE = MAPS_FULLRES

# ── Colormap + blending (Python side, used only for the static PNGs) ────────

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

print("Rendering fairface PNGs (full fsaverage resolution)...")
ff_png_dir = FAIRFACE_OUT / "group_maps_sagittal"
ff_png_dir.mkdir(exist_ok=True)
for cname in CATEGORIES:
    render_png(cname, MAPS_FULLRES[cname], ff_png_dir / f"{cname.lower().replace(' ', '_')}.png")

print("Rendering target PNG (full fsaverage resolution)...")
render_png("target", MAPS_FULLRES["__target__"], TARGET_OUT / "target_map_sagittal.png")

# ── Build the interactive viewer's embedded data ─────────────────────────────
# Grid layout: (age, gender, race) -> fsaverage6 vector. Missing combinations
# are filled with NaN and will simply not appear as slider hits, since every
# FairFace bucket was confirmed present earlier.

n_age, n_gender, n_race = len(AGE_LEVELS), len(GENDER_LEVELS), len(RACE_LEVELS)
V = MAPS_INTERACTIVE[CATEGORIES[0]].shape[0]

grid = np.full((n_age, n_gender, n_race, V), np.nan, dtype=np.float32)
missing = []
for ai, age in enumerate(AGE_LEVELS):
    for gi, gender in enumerate(GENDER_LEVELS):
        for ri, race in enumerate(RACE_LEVELS):
            key = category_key(age, gender, race)
            if key in MAPS_INTERACTIVE:
                grid[ai, gi, ri] = MAPS_INTERACTIVE[key]
            else:
                missing.append(key)
if missing:
    print(f"WARNING: {len(missing)} grid cells missing (no matching category found): {missing}")

target_vec = MAPS_INTERACTIVE["__target__"].astype(np.float32)

# ── Quantize to uint8 with a single global scale/offset (kept consistent so
#    interpolated + target values decode on the same footing) ───────────────

finite_vals = grid[~np.isnan(grid)]
gmin = float(min(finite_vals.min(), target_vec.min()))
gmax = float(max(finite_vals.max(), target_vec.max()))
scale = (gmax - gmin) / 255.0 if gmax > gmin else 1.0

def quantize(v):
    q = np.clip(np.round((np.nan_to_num(v, nan=gmin) - gmin) / scale), 0, 255).astype(np.uint8)
    return q

grid_q = quantize(grid)             # (n_age, n_gender, n_race, V) uint8
target_q = quantize(target_vec)     # (V,) uint8

def b64_bytes(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")

def b64_float32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")

def b64_int32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.int32).tobytes()).decode("ascii")

print(f"Quantized grid: {grid_q.nbytes / 1e6:.1f} MB raw uint8 "
      f"({grid_q.nbytes * 4 / 3 / 1e6:.1f} MB as base64 text, approx)")

embedded = {
    "age_levels": AGE_LEVELS,
    "gender_levels": GENDER_LEVELS,
    "race_levels": RACE_LEVELS,
    "n_lh": int(N_LH_FULL),
    "n_rh": int(rhF_coords.shape[0]),
    "gmin": gmin,
    "scale": scale,
    "grid_shape": list(grid_q.shape),
    "grid_b64": b64_bytes(grid_q),
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

# ── HTML + JS ─────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TRIBE v2 — Target vs FairFace Population, Sagittal View</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { margin:0; background:#0d0d0d; color:#eee; font-family: -apple-system, Helvetica, Arial, sans-serif; }
  #controls { padding: 14px 20px; background:#151515; border-bottom:1px solid #333; }
  #plot { width:100vw; height:82vh; }
  .row { display:flex; align-items:center; gap:14px; margin:8px 0; }
  .row label { width:110px; font-size:13px; color:#aaa; }
  .row input[type=range] { flex:1; }
  .val { width:220px; font-size:12px; color:#ddd; text-align:right; }
  #toggle { margin-bottom:6px; }
  #toggle label { margin-right:18px; font-size:14px; cursor:pointer; }
  #sliders.disabled { opacity:0.35; pointer-events:none; }
  h2 { margin: 0 0 10px 0; font-size:16px; font-weight:600; color:#fff; }
</style>
</head>
<body>

<div id="controls">
  <h2>TRIBE v2 — Target vs FairFace Population</h2>
  <div id="toggle">
    <label><input type="radio" name="mode" value="target" checked> Target</label>
    <label><input type="radio" name="mode" value="fairface"> FairFace population</label>
  </div>
  <div id="sliders" class="disabled">
    <div class="row">
      <label>Age</label>
      <input type="range" id="ageSlider" min="0" max="__AGE_MAX__" step="0.01" value="__AGE_MAX_HALF__">
      <div class="val" id="ageVal"></div>
    </div>
    <div class="row">
      <label>Gender</label>
      <input type="range" id="genderSlider" min="0" max="__GENDER_MAX__" step="0.01" value="0.5">
      <div class="val" id="genderVal"></div>
    </div>
    <div class="row">
      <label>Race</label>
      <input type="range" id="raceSlider" min="0" max="__RACE_MAX__" step="0.01" value="__RACE_MAX_HALF__">
      <div class="val" id="raceVal"></div>
    </div>
  </div>
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

const gridFlat   = b64ToUint8(DATA.grid_b64);           // (nAge*nGender*nRace*V)
const targetQ    = b64ToUint8(DATA.target_b64);         // (V)
const lhSulc     = b64ToFloat32(DATA.lh_sulc_b64);
const rhSulc     = b64ToFloat32(DATA.rh_sulc_b64);
const lhCoords   = b64ToFloat32(DATA.lh_coords_b64);
const rhCoords   = b64ToFloat32(DATA.rh_coords_b64);
const lhFaces    = b64ToInt32(DATA.lh_faces_b64);
const rhFaces    = b64ToInt32(DATA.rh_faces_b64);

const [nAge, nGender, nRace, V] = DATA.grid_shape;
const nLH = DATA.n_lh, nRH = DATA.n_rh;
const GMIN = DATA.gmin, SCALE = DATA.scale;

function gridIndex(ai, gi, ri) { return ((ai * nGender + gi) * nRace + ri) * V; }

// Dequantize + blend one grid cell into an accumulator (weighted add)
function accumulate(acc, ai, gi, ri, weight) {
  if (weight <= 0) return;
  const base = gridIndex(ai, gi, ri);
  for (let v = 0; v < V; v++) {
    acc[v] += weight * (GMIN + gridFlat[base + v] * SCALE);
  }
}

// Trilinear interpolation across the (age, gender, race) grid for continuous
// slider positions in [0, nAxis-1].
function interpolate(ageFrac, genderFrac, raceFrac) {
  const a0 = Math.min(Math.floor(ageFrac), nAge - 1);
  const a1 = Math.min(a0 + 1, nAge - 1);
  const wa = ageFrac - a0;

  const g0 = Math.min(Math.floor(genderFrac), nGender - 1);
  const g1 = Math.min(g0 + 1, nGender - 1);
  const wg = genderFrac - g0;

  const r0 = Math.min(Math.floor(raceFrac), nRace - 1);
  const r1 = Math.min(r0 + 1, nRace - 1);
  const wr = raceFrac - r0;

  const acc = new Float32Array(V);
  const corners = [
    [a0, g0, r0, (1 - wa) * (1 - wg) * (1 - wr)],
    [a1, g0, r0, wa       * (1 - wg) * (1 - wr)],
    [a0, g1, r0, (1 - wa) * wg       * (1 - wr)],
    [a1, g1, r0, wa       * wg       * (1 - wr)],
    [a0, g0, r1, (1 - wa) * (1 - wg) * wr],
    [a1, g0, r1, wa       * (1 - wg) * wr],
    [a0, g1, r1, (1 - wa) * wg       * wr],
    [a1, g1, r1, wa       * wg       * wr],
  ];
  for (const [ai, gi, ri, w] of corners) accumulate(acc, ai, gi, ri, w);
  return acc;
}

function dequantizeTarget() {
  const out = new Float32Array(V);
  for (let v = 0; v < V; v++) out[v] = GMIN + targetQ[v] * SCALE;
  return out;
}

// hot colormap approximation (matplotlib 'hot'): black -> red -> yellow -> white
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

// ── Initial figure ────────────────────────────────────────────────────────

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

const initTarget = dequantizeTarget();
const initLhColors = blendOntoSulc(lhSulc, initTarget.subarray(0, nLH));
const initRhColors = blendOntoSulc(rhSulc, initTarget.subarray(nLH, nLH + nRH));

const traceLH = {
  type: 'mesh3d', x: lhX, y: lhY, z: lhZ, i: lhI, j: lhJ, k: lhK,
  vertexcolor: initLhColors, lighting, lightposition, showscale: false, hoverinfo: 'skip',
};
const traceRH = {
  type: 'mesh3d', x: rhX, y: rhY, z: rhZ, i: rhI, j: rhJ, k: rhK,
  vertexcolor: initRhColors, lighting, lightposition, showscale: false, hoverinfo: 'skip',
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
  annotations: [],
  grid: { rows: 1, columns: 2 },
};

// mesh3d traces need to be routed to scene/scene2 explicitly
traceLH.scene = 'scene';
traceRH.scene = 'scene2';

Plotly.newPlot('plot', [traceLH, traceRH], layout, { responsive: true });

// ── Controls wiring ──────────────────────────────────────────────────────

const ageSlider = document.getElementById('ageSlider');
const genderSlider = document.getElementById('genderSlider');
const raceSlider = document.getElementById('raceSlider');
const ageVal = document.getElementById('ageVal');
const genderVal = document.getElementById('genderVal');
const raceVal = document.getElementById('raceVal');
const slidersDiv = document.getElementById('sliders');

function labelFor(levels, frac) {
  const i0 = Math.min(Math.floor(frac), levels.length - 1);
  const i1 = Math.min(i0 + 1, levels.length - 1);
  const w = frac - i0;
  if (i0 === i1 || w < 0.02) return levels[i0];
  if (w > 0.98) return levels[i1];
  return `${levels[i0]} → ${levels[i1]} (${Math.round(w * 100)}%)`;
}

let pending = null;
function scheduleUpdate() {
  if (pending) cancelAnimationFrame(pending);
  pending = requestAnimationFrame(() => {
    pending = null;
    const mode = document.querySelector('input[name=mode]:checked').value;
    if (mode === 'target') {
      applyActivation(dequantizeTarget());
    } else {
      const a = parseFloat(ageSlider.value);
      const g = parseFloat(genderSlider.value);
      const r = parseFloat(raceSlider.value);
      ageVal.textContent = labelFor(DATA.age_levels, a);
      genderVal.textContent = labelFor(DATA.gender_levels, g);
      raceVal.textContent = labelFor(DATA.race_levels, r);
      applyActivation(interpolate(a, g, r));
    }
  });
}

[ageSlider, genderSlider, raceSlider].forEach(el => el.addEventListener('input', scheduleUpdate));
document.querySelectorAll('input[name=mode]').forEach(el => el.addEventListener('change', () => {
  const mode = document.querySelector('input[name=mode]:checked').value;
  slidersDiv.classList.toggle('disabled', mode !== 'fairface');
  scheduleUpdate();
}));

scheduleUpdate();
</script>
</body>
</html>
"""

html = (HTML_TEMPLATE
        .replace("__EMBEDDED_JSON__", embedded_json)
        .replace("__AGE_MAX__", str(n_age - 1))
        .replace("__AGE_MAX_HALF__", str((n_age - 1) / 2))
        .replace("__GENDER_MAX__", str(n_gender - 1))
        .replace("__RACE_MAX__", str(n_race - 1))
        .replace("__RACE_MAX_HALF__", str((n_race - 1) / 2)))

out_html = INTERACTIVE_OUT / "combined_interactive.html"
out_html.write_text(html, encoding="utf-8")
print(f"\nSaved interactive viewer -> {out_html}  ({out_html.stat().st_size / 1e6:.1f} MB)")
"""3-way interactive cortical map: target / suppressed / general.

Reads native fsaverage5 means from target_preds/suppress_maps.npz (written by
apply_suppress_eval.py) and writes interactive_study/suppress_toggle.html.

Does not re-stream the FairFace pred dump.

Usage:
  python scripts/viz_suppress.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

MAPS = Path("./target_preds/suppress_maps.npz")
OUT = Path("./interactive_study/suppress_toggle.html")
N_LH5 = 10242


def b64_bytes(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def b64_float32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def b64_int32(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.int32).tobytes()).decode("ascii")


def quantize(v, gmin, scale):
    return np.clip(np.round((v - gmin) / scale), 0, 255).astype(np.uint8)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TRIBE v2 — Mia suppress readout</title>
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
  <h2>TRIBE v2 — face-cortex before / after identity suppress</h2>
  <div id="toggle">
    <label><input type="radio" name="mode" value="target" checked> Target (baseline)</label>
    <label><input type="radio" name="mode" value="suppressed"> Target suppressed</label>
    <label><input type="radio" name="mode" value="general"> General faces</label>
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

const maps = {
  target: b64ToUint8(DATA.target_b64),
  suppressed: b64ToUint8(DATA.suppressed_b64),
  general: b64ToUint8(DATA.general_b64),
};
const lhSulc = b64ToFloat32(DATA.lh_sulc_b64);
const rhSulc = b64ToFloat32(DATA.rh_sulc_b64);
const lhCoords = b64ToFloat32(DATA.lh_coords_b64);
const rhCoords = b64ToFloat32(DATA.rh_coords_b64);
const lhFaces = b64ToInt32(DATA.lh_faces_b64);
const rhFaces = b64ToInt32(DATA.rh_faces_b64);
const V = maps.target.length;
const nLH = DATA.n_lh, nRH = DATA.n_rh;
const GMIN = DATA.gmin, SCALE = DATA.scale;

document.getElementById('meta').textContent = DATA.meta || '';

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
  Plotly.restyle('plot', { vertexcolor: [blendOntoSulc(lhSulc, lhAct), blendOntoSulc(rhSulc, rhAct)] }, [0, 1]);
}
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
const lighting = { ambient: 0.75, diffuse: 0.7, specular: 0.05, roughness: 0.8, fresnel: 0.1 };
const lightposition = { x: 100, y: 200, z: 300 };
const [lhX, lhY, lhZ] = toXYZ(lhCoords);
const [rhX, rhY, rhZ] = toXYZ(rhCoords);
const [lhI, lhJ, lhK] = toIJK(lhFaces);
const [rhI, rhJ, rhK] = toIJK(rhFaces);
const init = dequantize(maps.target);
const traceLH = {
  type: 'mesh3d', x: lhX, y: lhY, z: lhZ, i: lhI, j: lhJ, k: lhK,
  vertexcolor: blendOntoSulc(lhSulc, init.subarray(0, nLH)),
  lighting, lightposition, showscale: false, hoverinfo: 'skip', scene: 'scene',
};
const traceRH = {
  type: 'mesh3d', x: rhX, y: rhY, z: rhZ, i: rhI, j: rhJ, k: rhK,
  vertexcolor: blendOntoSulc(rhSulc, init.subarray(nLH, nLH + nRH)),
  lighting, lightposition, showscale: false, hoverinfo: 'skip', scene: 'scene2',
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
Plotly.newPlot('plot', [traceLH, traceRH], {
  paper_bgcolor: '#0d0d0d', plot_bgcolor: '#0d0d0d', font: { color: 'white' },
  margin: { l: 0, r: 0, t: 10, b: 10 },
  scene: sceneKwargs(-2.4, [0.0, 0.5]),
  scene2: sceneKwargs(2.4, [0.5, 1.0]),
}, { responsive: true });
document.querySelectorAll('input[name=mode]').forEach(el =>
  el.addEventListener('change', () => {
    const mode = document.querySelector('input[name=mode]:checked').value;
    applyActivation(dequantize(maps[mode]));
  })
);
</script>
</body>
</html>
"""


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def main():
    from nilearn import datasets, surface
    from scipy.spatial import cKDTree

    d = np.load(MAPS)
    native = {
        "target": d["target_baseline"].astype(np.float32),
        "suppressed": d["target_suppressed"].astype(np.float32),
        "general": d["general_baseline"].astype(np.float32),
    }

    print("Fetching fsaverage meshes...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    lhF_coords, lhF_faces = surface.load_surf_mesh(fsaverage["pial_left"])
    rhF_coords, rhF_faces = surface.load_surf_mesh(fsaverage["pial_right"])
    lhF_sulc = normalize(surface.load_surf_data(fsaverage["sulc_left"]))
    rhF_sulc = normalize(surface.load_surf_data(fsaverage["sulc_right"]))
    N_LH_FULL = lhF_coords.shape[0]

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
    lhF_idx, lhF_w = build_idw(lh5_sphere, lhF_sphere)
    rhF_idx, rhF_w = build_idw(rh5_sphere, rhF_sphere)

    def upsample(data_1d):
        lh5, rh5 = data_1d[:N_LH5], data_1d[N_LH5:]
        lh = (lh5[lhF_idx] * lhF_w).sum(axis=1)
        rh = (rh5[rhF_idx] * rhF_w).sum(axis=1)
        return np.concatenate([lh, rh]).astype(np.float32)

    full = {k: upsample(v) for k, v in native.items()}
    gmin = float(min(v.min() for v in full.values()))
    gmax = float(max(v.max() for v in full.values()))
    scale = (gmax - gmin) / 255.0 if gmax > gmin else 1.0

    embedded = {
        "n_lh": int(N_LH_FULL),
        "n_rh": int(rhF_coords.shape[0]),
        "gmin": gmin,
        "scale": scale,
        "meta": "target = Mia baseline  |  suppressed = rank-16 readout residual  |  general = random FairFace faces",
        "target_b64": b64_bytes(quantize(full["target"], gmin, scale)),
        "suppressed_b64": b64_bytes(quantize(full["suppressed"], gmin, scale)),
        "general_b64": b64_bytes(quantize(full["general"], gmin, scale)),
        "lh_sulc_b64": b64_float32(lhF_sulc),
        "rh_sulc_b64": b64_float32(rhF_sulc),
        "lh_coords_b64": b64_float32(lhF_coords),
        "rh_coords_b64": b64_float32(rhF_coords),
        "lh_faces_b64": b64_int32(lhF_faces),
        "rh_faces_b64": b64_int32(rhF_faces),
    }
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", json.dumps(embedded))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

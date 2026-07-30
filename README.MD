# TRIBE v2 Face Identity Abliteration Pipeline

A fully data-driven pipeline for surgeon-like **face identity suppression** inside the TRIBE v2 brain model. Same model and abliteration goal as [janice-stfu](https://github.com/aloshdenny/janice-stfu), retargeted from video concept categories to a **specific person's face**, with FairFace demographic controls, face-selective cortical ROI masks (OFA/FFA), multi-layer PCA weight surgery, and automated artifact chunking for GitHub.

---

## 🚀 Setup & Installation

Follow these steps to set up the project locally using your existing `tribev2` Conda environment:

```bash
# 1. Clone the repository
git clone https://github.com/aloshdenny/e85.git
cd e85

# 2. Activate your tribev2 conda environment
conda activate tribev2

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 📂 Data Layout

The pipeline uses two data sources: a **FairFace demographic baseline** (general population) and a **target person** whose face identity is to be abliterated. Paths are auto-discovered where noted — category names come from FairFace zip / npz stems.

```
fairface/               # 126 zip files, one per age × gender × race bucket
  ├── 20-29_female_white.zip
  ├── 30-39_male_east_asian.zip
  └── ...

fairface_preds/         # npz predictions from infer_fairface_bulk.py
  ├── 20-29_female_white.npz
  ├── 20-29_female_white_chunk_000.npz   # if >~95MB (GitHub 100MB limit)
  └── ...

target/                 # target person raw images (one or more *.zip)
  └── **.zip            # discovered recursively (e.g. mia.zip)

target_preds/           # npz predictions from infer_target_face.py
  └── **.npz            # discovered recursively (e.g. mia.npz)

val/                    # holdout images for validation (target + general faces)
  ├── mia1_face.jpg
  ├── euro1_face.jpg
  └── ...

abliterated/            # outputs of abliteration.py
  ├── vjepa2_hf_checkpoint/   # HF-format sharded safetensors for validation redirect
  ├── directions_L{idx}.npy
  ├── surgery_log.json
  ├── raw_activations_norm/
  └── masks/
      ├── face_mask.npy
      └── layer_profile.npy
```

> **Target Specification**: All `*.zip` under `./target/` and all `*.npz` under `./target_preds/` are discovered recursively. FairFace buckets are one zip / logical npz per demographic category (126 buckets).

---

## ⚙️ Abliteration Pipeline

The dependency chain is linear:

```mermaid
flowchart TD
    A["0. (Optional) Crop Faces<br/>diagnostics/crop_faces.py"] --> B["1. Baseline Inference<br/>infer_fairface_bulk.py"]
    B --> C["2. Target Face Inference<br/>infer_target_face.py"]
    C --> D["3. ROI Selectivity Diagnostic<br/>check_face_roi_selectivity.py"]
    D --> E["4. Identity Abliteration<br/>abliteration.py"]
    E --> F["5. Validation<br/>validation.py"]
    F --> G["6. Visualization<br/>viz.py"]
```

### 0. (Optional) Crop Faces
Detects the highest-confidence face per image with MTCNN and crops in place under `./target/` (configurable `SRC_DIR` in the script).
```bash
python scripts/diagnostics/crop_faces.py
```
Zip cropped faces under `./target/` before running target inference.

### 1. Baseline Inference
Bulk TribeV2 inference over all FairFace category zips in `./fairface/`. Batched for GPU throughput. Writes per-category `.npz` (auto-chunked if over GitHub’s 100MB limit) to `./fairface_preds/`.
```bash
python scripts/infer_fairface_bulk.py
```

### 2. Target Face Inference
TribeV2 inference over all target zips found recursively in `./target/`. Outputs to `./target_preds/`.
```bash
python scripts/infer_target_face.py
```

### 3. ROI Selectivity Diagnostic
Checks whether the target-vs-general contrast is concentrated in face-selective ROIs (OFA, FFA, STS, ATL, TP, …) or diffuse across cortex. Run before surgery to confirm a clean, face-localised signal.
```bash
python scripts/diagnostics/check_face_roi_selectivity.py
```

### 4. Identity Abliteration (Surgery)
Builds a cortical face mask (OFA+FFA by default; add `--include-secondary` for TP+ATL), collects per-layer vjepa2 activations from static repeated frames, computes PCA directions, applies surgical weight suppression, and exports an HF-format checkpoint for validation.
```bash
python scripts/abliteration.py \
  --general-preds-dir ./fairface_preds --general-zips-dir ./fairface \
  --target-preds-npz ./target_preds/mia.npz --target-zip ./target/mia.zip \
  --tolerance -1.0 --n_components 3 --n_layers 5

# Include secondary face ROIs (TP + ATL)
python scripts/abliteration.py \
  --general-preds-dir ./fairface_preds --general-zips-dir ./fairface \
  --target-preds-npz ./target_preds/mia.npz --target-zip ./target/mia.zip \
  --tolerance -1.0 --n_components 3 --n_layers 5 --include-secondary
```

### 5. Validation
Redirects TribeV2’s image extractor to the local HF checkpoint, clears cache pitfalls with distinct pass-2 events, and compares mask-averaged brain response before vs. after surgery on `./val/` holdouts (target prefix vs. general faces).
```bash
python scripts/validation.py --val-dir ./val --target-prefix mia \
  --hf-checkpoint-dir ./abliterated/vjepa2_hf_checkpoint
```

### 6. Visualization
Interactive brain-scan HTML (`interactive_study/combined_interactive.html`) with FairFace slider and target-vs-baseline toggle, plus per-category PNG maps under `fairface_study/` / `target_study/`.
```bash
python scripts/viz.py
```

---

## 💾 Checkpoint & NPZ Chunking

GitHub enforces a **100MB per-file** limit and a **~2GB pack size** per push.

### FairFace prediction NPZs
Large category `.npz` files are automatically split on save into byte chunks (`{stem}_chunk_000.npz`, …) via `scripts/chunk_utils.py`. Consumers (`viz.py`, `abliteration.py`, diagnostics) call `load_npz()` / `discover_npz()`, which re-fuse chunks to a temp file, load, and delete the temp.

### Abliterated HF checkpoint
Surgery exports `abliterated/vjepa2_hf_checkpoint/` as sharded `.safetensors` (default `--max-shard-size 20MB`) so individual shards stay under the file limit. Push large trees in separate size-capped commits (subfolder-isolated batches under ~1.5 GiB) to stay under the pack limit.

### Legacy `.pt` chunk helpers
`chunk_utils.save_chunked` / `load_chunked` still support `<25MB` binary splits of PyTorch state dicts (same pattern as janice-stfu) when needed.

---

## 🧠 Methodology & Concepts

### 1. Cortical Face ROIs (Destrieux Atlas)
Surgery targets vertices in **OFA** and **FFA** — the primary face-selective regions with clean target-vs-general contrast. Secondary ROIs (TP, ATL) are optional via `--include-secondary`.

| ROI | Label(s) | Role |
|-----|----------|------|
| **OFA** | `G_and_S_occipital_inf`, `S_oc_middle_and_Lunatus`, `Pole_occipital` | Early face detection |
| **FFA** | `G_oc-temp_lat-fusifor` | Face identity recognition |
| **TP** *(secondary)* | `Pole_temporal` | Person familiarity / semantic memory |
| **ATL** *(secondary)* | `G_temporal_inf`, `G_oc-temp_med-Parahip` | Person identity (broader) |

> **Note**: Subcortical reward nodes (nucleus accumbens, amygdala, hippocampus) and high-ranking photometric confounds (e.g. V4/MT on static images) are not used as ablation targets.

### 2. Leave-One-Group-Out (LOGO) Contrast
Instead of janice-stfu’s video-category LOSO, the identity contrast is against the FairFace demographic baseline:
- **LOGO contrast** = `target_mean − mean(all FairFace category means, equal-weight per bucket)`

Equal weight per age×gender×race bucket prevents large demographic groups from dominating the general-population estimate.

### 3. PCA Direction-Finding (Weighted)
Activations are collected for target and sampled general images at each selected encoder layer (static frame repeated to vjepa2’s clip length — no fake mp4 path). PCA directions use the **y-weighted covariance** of centered activations, where `y = preds[:, mask].mean()` (mask-averaged brain response from the precomputed npzs). Directions track the face-identity brain signal rather than arbitrary activation variance.

### 4. Multi-Layer Surgery
`--n_layers N` selects the top-N encoder layers ranked by correlation between per-layer L2-norm and `y`. Directions and surgery are per-layer:
- Directions: `abliterated/directions_L{layer_idx}.npy`
- Surgery log: `abliterated/surgery_log.json`
- Deployable weights: `abliterated/vjepa2_hf_checkpoint/`

### 5. Tolerance Scale
Continuous scale mapping surgeon intervention outcomes (same as janice-stfu):
```
  -1       -0.5       0       +0.5       +1
   |--------|---------|---------|--------|
   Repulsion Aversion  Neutral  Acceptance Attraction
```
- **Tolerance < 0**: Suppress / repel the concept (face identity).
- **Tolerance = 0**: Neutral (no effect).
- **Tolerance > 0**: Accept / tolerate the concept.
> **Note**: Positive tolerance represents the absence of aversion (no pushback), not a preference for the concept.

---

## 🛠️ Troubleshooting

### Deadlock on Cache Access
If multi-process file locks deadlock, remove the inflight database in your cache directory:
```bash
find ./cache -name "inflight.db" -delete
```

### No Zip / NPZ Files Found
Place target archives under `./target/` (`.zip`) and predictions under `./target_preds/` (`.npz`). Discovery is recursive. For chunked FairFace preds, keep the full `{stem}_chunk_NNN.npz` set — `load_npz(stem.npz)` recombines them automatically.

### Validation Shows Zero “Encoding video” Bars on Pass 2
Pass 2 must use distinct event filepaths **and** timeline IDs (the validation script hardlinks clips for this). Identical events hit the neuralset cache regardless of the abliterated weights.

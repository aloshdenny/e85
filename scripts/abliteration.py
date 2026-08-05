"""
abliteration.py

Abliterate a specific person's face identity from the vjepa2 encoder used by
TribeV2, targeting face-selective cortical ROIs specifically (not the full
31-ROI table from layer_analysis.py / abliteration.py, which was built for
content-category work like porn/food addiction).

Based on check_face_roi_selectivity.py's diagnostic on Mia's data:
  - OFA (+0.025) and FFA (+0.014) show clean, individually-dominant contrast.
  - STS/ATL/TP/PREC/MPFC/PCC were weak/near-zero for this target -- excluded
    by default. Pass --include-secondary to add TP+ATL back in (small but
    positive margin) if you want broader identity coverage.
  - V4/MT ranked surprisingly high (#2, #4) -- likely a photometric confound
    (lighting/background/compression), not real identity signal, since MT is
    motion-selective and your inputs are static. Not used as ablation targets
    regardless, but worth a manual sanity check on your image sets before
    trusting this contrast further.

KEY ARCHITECTURAL DIFFERENCE FROM abliteration.py:
  abliteration.py collects activations by decoding real .mp4 video files
  through iter_clips_from_video() (torchvision VideoReader), because its
  target/baseline data were literal video categories.
  Here, activations are collected by feeding vjepa2_module a STATIC REPEATED
  FRAME directly (no fake .mp4, no TribeModel.predict(), no FmriExtractor
  offset concerns) -- since we're hooking the vjepa2 encoder directly rather
  than running the full brain-prediction pipeline. This is the same encoder
  (model.data.video_feature.image.model.model) but a much cheaper path to it.

  y (brain response used to weight PCA direction-finding) is NOT recomputed
  here -- it's read directly from the already-saved fairface_preds/*.npz and
  target_preds/*.npz files (preds[:, mask].mean() per image), since those
  already contain the full per-image predicted vertex vector.

DATA LAYOUT ASSUMPTIONS (adjust via CLI flags if yours differs):
  --general-preds-dir   folder of category .npz (from infer_face_bulk.py),
                         each with 'preds' (N,20484) and 'filenames' (N,)
  --general-zips-dir    folder of the matching category .zip files (same
                         stem as each .npz) -- needed to re-read raw images
                         for activation collection (npz only stored brain
                         predictions, not vjepa2 activations)
  --target-preds-npz    target person's .npz (same format)
  --target-zip          zip file containing the target person's raw images,
                         with filenames matching target-preds-npz's 'filenames'

Usage:
  python scripts/abliteration.py \
      --general-preds-dir ./fairface_preds --general-zips-dir ./fairface \
      --target-preds-npz ./target_preds/mia.npz --target-zip ./target/mia.zip \
      --tolerance -1.0 --n_components 3 --n_layers 5
"""

import os, gc, sys, json, warnings, logging, argparse, zipfile, random
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.update({"PYTHONWARNINGS": "ignore", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})

import numpy as np
import cv2
import torch
from torchvision import transforms
from tribev2.demo_utils import TribeModel
from chunk_utils import discover_npz, load_npz

# ── ROI definitions (Destrieux exact labels) ─────────────────────────────────

PRIMARY_FACE_ROIS = {
    "OFA": ["G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "Pole_occipital"],
    "FFA": ["G_oc-temp_lat-fusifor"],
}
SECONDARY_FACE_ROIS = {
    "TP":  ["Pole_temporal"],
    "ATL": ["G_temporal_inf", "G_oc-temp_med-Parahip"],
}

OUT_DIR = Path("./abliterated")
OUT_DIR.mkdir(exist_ok=True)
MASK_DIR = OUT_DIR / "masks"
MASK_DIR.mkdir(exist_ok=True)

CLIP_FRAMES = 16    # vjepa2's native expected temporal length
IMG_SIZE    = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Mask construction ─────────────────────────────────────────────────────────

def build_face_mask(include_secondary: bool):
    from nilearn import datasets as nl_datasets
    print("Loading Destrieux atlas...")
    destrieux = nl_datasets.fetch_atlas_surf_destrieux()
    lh_labels = np.array(destrieux["map_left"])
    rh_labels = np.array(destrieux["map_right"])
    atlas_names = [n.decode() if isinstance(n, bytes) else n for n in destrieux["labels"]]

    roi_dict = dict(PRIMARY_FACE_ROIS)
    if include_secondary:
        roi_dict.update(SECONDARY_FACE_ROIS)

    mask = np.zeros(20484, dtype=bool)
    for key, exact in roi_dict.items():
        idxs = [i for i, n in enumerate(atlas_names) if n in exact]
        if not idxs:
            print(f"  [WARN] {key}: no matching labels found")
            continue
        roi_mask = np.concatenate([np.isin(lh_labels, idxs), np.isin(rh_labels, idxs)])
        print(f"  {key}: {roi_mask.sum()} vertices")
        mask |= roi_mask

    print(f"Combined face mask: {mask.sum()} vertices "
          f"({'OFA+FFA+TP+ATL' if include_secondary else 'OFA+FFA only'})")
    return mask


# ── Image -> vjepa2 input tensor (static repeated frame) ─────────────────────

# Fixed reference values for photometric normalization. Chosen close to the
# general population's own typical range (not an extreme target) so images
# aren't pushed unnaturally far from realistic appearance -- just brought
# onto a common scale so contrast/luminance stop being a usable signal for
# telling target apart from general.
PHOTOMETRIC_REFERENCE_MEAN = 100.0
PHOTOMETRIC_REFERENCE_STD = 50.0
ENABLE_PHOTOMETRIC_NORMALIZATION = True  # set via --disable-photometric-norm


def normalize_photometrics(img_bgr, target_mean=PHOTOMETRIC_REFERENCE_MEAN,
                           target_std=PHOTOMETRIC_REFERENCE_STD):
    """
    Matches an image's overall luminance mean/std to a fixed reference,
    applied uniformly across all 3 channels (derived from grayscale stats,
    so hue/color balance is preserved while brightness and contrast are
    normalized). Confirmed via check_direction_confound.py: target images
    ran ~30% higher contrast than the general population on average, and
    that contrast difference leaked substantially into the extracted
    "identity" direction (corr=+0.38). This strips that axis out at the
    input level for BOTH target and general images, rather than trying to
    out-math it in activation space.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur_mean, cur_std = float(gray.mean()), float(gray.std())
    if cur_std < 1e-6:
        cur_std = 1e-6
    scale = target_std / cur_std
    img_f = img_bgr.astype(np.float32)
    normalized = (img_f - cur_mean) * scale + target_mean
    return np.clip(normalized, 0, 255).astype(np.uint8)


def image_to_vjepa_input(img_bgr):
    """
    img_bgr: raw decoded image (H,W,3) BGR uint8, e.g. from cv2.imdecode.
    Returns a (CLIP_FRAMES, 3, IMG_SIZE, IMG_SIZE) tensor, ImageNet-normalized,
    same frame repeated -- since we're hooking vjepa2 directly (not going
    through TribeModel.predict()), there's no offset/duration constraint to
    satisfy, just vjepa2's own expected clip length.
    """
    if ENABLE_PHOTOMETRIC_NORMALIZATION:
        img_bgr = normalize_photometrics(img_bgr)
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
    img_t = normalize_fn(img_t)
    return img_t.unsqueeze(0).repeat(CLIP_FRAMES, 1, 1, 1)  # (T,3,H,W)


def decode_image_from_zip(zf: zipfile.ZipFile, name: str):
    data = zf.read(name)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode {name}")
    return img


# ── Sample general-population images matched to their npz preds ────────────

def sample_general_images(general_preds_dir: Path, general_zips_dir: Path,
                           mask: np.ndarray, total_sample: int, seed: int = 0):
    """
    Returns list of (img_bgr, y) tuples, y = preds[:, mask].mean() for that image,
    sampled across category buckets so no single demographic bucket dominates.
    """
    rng = random.Random(seed)
    npz_files = discover_npz(general_preds_dir)
    if not npz_files:
        raise FileNotFoundError(f"No npz files in {general_preds_dir}")

    per_cat_budget = max(1, total_sample // len(npz_files))
    print(f"Sampling ~{per_cat_budget} images per category ({len(npz_files)} categories)...")

    samples = []
    for npz_path in npz_files:
        zip_path = general_zips_dir / f"{npz_path.stem}.zip"
        if not zip_path.exists():
            print(f"  [WARN] no matching zip for {npz_path.stem}, skipping")
            continue
        data = load_npz(npz_path)
        preds = data["preds"]
        filenames = data["filenames"]
        n = len(filenames)
        if n == 0:
            continue
        idxs = rng.sample(range(n), min(per_cat_budget, n))
        with zipfile.ZipFile(zip_path, "r") as zf:
            for i in idxs:
                try:
                    img = decode_image_from_zip(zf, str(filenames[i]))
                    y = float(preds[i][mask].mean())
                    samples.append((img, y))
                except Exception as e:
                    print(f"  [WARN] {npz_path.stem}/{filenames[i]}: {e}")

    print(f"  Collected {len(samples)} general-population samples")
    return samples


def load_target_images(target_preds_npz: Path, target_zip: Path, mask: np.ndarray):
    data = load_npz(target_preds_npz)
    preds = data["preds"]
    filenames = data["filenames"]
    samples = []
    with zipfile.ZipFile(target_zip, "r") as zf:
        for i, name in enumerate(filenames):
            try:
                img = decode_image_from_zip(zf, str(name))
                y = float(preds[i][mask].mean())
                samples.append((img, y))
            except Exception as e:
                print(f"  [WARN] target/{name}: {e}")
    print(f"  Loaded {len(samples)} target images")
    return samples


# ── Layer profiling: find which vjepa2 layers correlate with the face mask ──

def profile_layers(vjepa2_module, encoder_blocks, n_layers, target_samples, general_samples,
                    max_profile_n=400):
    """
    Quick per-layer correlation between hidden-state L2 norm and y (mask-averaged
    brain response), across a combined target+general sample. Mirrors
    layer_analysis.py's approach but computed fresh from face-image data.
    """
    print("\nProfiling layers (combined target+general sample)...")
    combined = target_samples + general_samples
    if len(combined) > max_profile_n:
        combined = random.sample(combined, max_profile_n)

    layer_acts = {}
    def make_hook(idx):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_acts[idx] = hidden.mean(dim=1).mean(dim=0).detach().float().cpu().numpy()
        return hook

    handles = [encoder_blocks[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]

    layer_norms = [[] for _ in range(n_layers)]
    ys = []
    try:
        for i, (img, y) in enumerate(combined):
            clip = image_to_vjepa_input(img).unsqueeze(0).to(DEVICE)
            layer_acts.clear()
            with torch.no_grad():
                vjepa2_module(pixel_values_videos=clip)
            for li in range(n_layers):
                layer_norms[li].append(float(np.linalg.norm(layer_acts[li])))
            ys.append(y)
            del clip
            if i % 50 == 49:
                free()
    finally:
        for h in handles:
            h.remove()

    ys = np.array(ys)
    profile = np.zeros(n_layers)
    for li in range(n_layers):
        x = np.array(layer_norms[li])
        if x.std() > 1e-9 and ys.std() > 1e-9:
            profile[li] = abs(float(np.corrcoef(x, ys)[0, 1]))

    print("Layer profile (|r| between layer-norm and mask-y):")
    for li in np.argsort(profile)[::-1][:10]:
        print(f"  L{li:2d}  |r|={profile[li]:.4f}")

    np.save(MASK_DIR / "layer_profile.npy", profile)
    return profile


def pick_top_layers(profile, n_layers_wanted):
    k = min(n_layers_wanted, len(profile))
    top = np.argsort(profile)[::-1][:k].tolist()
    top.sort()
    return top


# ── Activation collection at a specific layer ────────────────────────────────

def collect_layer_activations(vjepa2_module, encoder_blocks, layer_idx,
                               target_samples, general_samples,
                               cache_dir: Path = None):
    """
    Returns X, y (order: target_samples first, then general_samples -- caller
    knows n_target = len(target_samples) and can split accordingly).

    If cache_dir is given, raw X/y are saved to disk after collection (and
    loaded from there on a rerun instead of re-collecting) -- collection is
    the expensive part; iterating on direction-finding math shouldn't require
    re-running it every time.
    """
    if cache_dir is not None:
        x_path = cache_dir / f"raw_X_L{layer_idx}.npy"
        y_path = cache_dir / f"raw_y_L{layer_idx}.npy"
        if x_path.exists() and y_path.exists():
            print(f"  [L{layer_idx}] loading cached raw activations from {cache_dir}")
            return np.load(x_path), np.load(y_path)

    hook_buffer = [None]
    def hook_fn(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_buffer[0] = hidden.mean(dim=1).squeeze(0).detach().cpu().float().numpy()

    handle = encoder_blocks[layer_idx].register_forward_hook(hook_fn)
    X, y_list = [], []
    try:
        for img, y in (target_samples + general_samples):
            clip = image_to_vjepa_input(img).unsqueeze(0).to(DEVICE)
            hook_buffer[0] = None
            with torch.no_grad():
                vjepa2_module(pixel_values_videos=clip)
            if hook_buffer[0] is not None:
                X.append(hook_buffer[0])
                y_list.append(y)
            del clip
    finally:
        handle.remove()
    free()
    X_arr = np.stack(X)
    y_arr = np.array(y_list, dtype=np.float32)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / f"raw_X_L{layer_idx}.npy", X_arr)
        np.save(cache_dir / f"raw_y_L{layer_idx}.npy", y_arr)

    return X_arr, y_arr


# ── Direction finding ─────────────────────────────────────────────────────────

def find_directions(X, y, n_components, label):
    """Original weighted-PCA approach. Kept for use as SECONDARY components
    (on the residual after removing the contrastive direction below) -- on
    its own this was validated to produce a non-selective direction (general
    population dropped as much or more than the target person), since it
    optimizes for 'what makes this ROI fire strongly' rather than 'what is
    specific to the target identity'."""
    y_range = y.max() - y.min()
    if y_range < 1e-9:
        weights = np.ones(len(y)) / len(y)
    else:
        weights = (y - y.min()) / (y_range + 1e-9)
        weights /= weights.sum()
    X_mean = (X * weights[:, None]).sum(axis=0, keepdims=True)
    X_c = (X - X_mean) * np.sqrt(weights[:, None])
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    print(f"  [{label}] singular values: {S[:5].round(4)}")
    dirs = Vt[:n_components].copy()
    for i in range(n_components):
        proj = X @ dirs[i]
        if float(np.corrcoef(proj, y)[0, 1]) < 0:
            dirs[i] *= -1
            print(f"  [{label}] flipped direction {i}")
    return dirs


def find_directions_contrastive(X, y, n_target, n_components, label):
    """
    Primary direction = normalize(mean(X_target) - mean(X_general)) -- a true
    difference-of-means contrast, which is what actually isolates 'target
    identity' rather than 'strong generic activation of this ROI'. This is
    the fix for the confound validation.py exposed: weighted-PCA alone let
    general-population images with strong OFA/FFA response dominate the
    direction just as much as the target person did.

    If n_components > 1, additional directions come from weighted-PCA
    (find_directions, same as before) run on the RESIDUAL after projecting
    out the primary direction -- these can still capture target-relevant
    variance beyond the raw mean shift, but they no longer carry the entire
    burden of separating target from general on their own.
    """
    X_target = X[:n_target]
    X_general = X[n_target:]

    primary = X_target.mean(axis=0) - X_general.mean(axis=0)
    norm = np.linalg.norm(primary)
    if norm < 1e-9:
        raise ValueError(f"[{label}] target and general means are identical -- "
                         f"no contrastive signal at this layer.")
    primary = primary / norm

    proj = X @ primary
    print(f"  [{label}] contrastive direction: target proj mean={proj[:n_target].mean():.4f}, "
          f"general proj mean={proj[n_target:].mean():.4f}, "
          f"separation={proj[:n_target].mean() - proj[n_target:].mean():+.4f}")

    dirs = [primary]

    if n_components > 1:
        # Remove primary component from every sample, then find secondary
        # directions in what's left via the original weighted-PCA approach.
        residual = X - np.outer(proj, primary)
        secondary = find_directions(residual, y, n_components - 1, f"{label}-residual")
        dirs.extend(secondary)

    return np.stack(dirs)


# ── Surgery (reused verbatim from abliteration.py) ───────────────────────────

def apply_surgery(encoder_blocks, all_dirs_by_layer, tolerance):
    for layer_idx, dirs in all_dirs_by_layer.items():
        dirs_t = torch.tensor(dirs, dtype=torch.float32).to(DEVICE)
        ortho = []
        for d in dirs_t:
            for q in ortho:
                d = d - (d @ q) * q
            n = d.norm()
            if n > 1e-6:
                ortho.append(d / n)
        if not ortho:
            print(f"  [WARN] No valid directions for layer {layer_idx}, skipping")
            continue
        ortho = torch.stack(ortho)
        block = encoder_blocks[layer_idx]
        print(f"\n  Surgery on encoder.layer[{layer_idx}] -- "
              f"{len(ortho)} direction(s), tolerance={tolerance}")
        for attr_path in ["attention.value", "attention.proj"]:
            mod = block
            for part in attr_path.split("."):
                mod = getattr(mod, part)
            W = mod.weight.data.clone()
            for q in ortho:
                W += tolerance * (W @ q).unsqueeze(-1) * q
            mod.weight.data = W
            print(f"    {attr_path}: {tuple(W.shape)} updated")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-preds-dir", required=True, type=Path)
    parser.add_argument("--general-zips-dir", required=True, type=Path)
    parser.add_argument("--target-preds-npz", required=True, type=Path)
    parser.add_argument("--target-zip", required=True, type=Path)
    parser.add_argument("--cache-folder", default="./cache", type=Path)
    parser.add_argument("--tolerance", type=float, default=-1.0,
                        help="-1=full suppression, 0=neutral, +1=amplify")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--include-secondary", action="store_true",
                        help="Include TP+ATL in the face mask (small positive margin "
                             "per the diagnostic, excluded by default).")
    parser.add_argument("--general-sample-size", type=int, default=2000,
                        help="Total general-population images to sample for activation "
                             "collection (spread across category buckets).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-weighted-pca-only", action="store_true",
                        help="Use the original weighted-PCA-only direction finding "
                             "instead of the contrastive (difference-of-means) primary "
                             "direction. NOT recommended -- validated to produce a "
                             "non-selective ablation (general population dropped as much "
                             "or more than the target person). Kept as an option for "
                             "comparison, not as the default path.")
    parser.add_argument("--source-model-name", default="facebook/vjepa2-vitg-fpc64-256",
                        help="Original HF model id, used to save the (unmodified) video "
                             "processor config alongside your abliterated weights.")
    parser.add_argument("--max-shard-size", default="20MB",
                        help="Shards the saved HF checkpoint below this size (git-friendly, "
                             "e.g. GitHub's ~25MB soft limit). AutoModel.from_pretrained() "
                             "reassembles sharded checkpoints transparently on load -- this "
                             "does not require any change to validation.py's loading logic.")
    parser.add_argument("--disable-photometric-norm", action="store_true",
                        help="Skip luminance/contrast normalization. NOT recommended -- "
                             "check_direction_confound.py found the target image set ran "
                             "~30% higher contrast than the general population, and that "
                             "leaked substantially into the extracted direction. Kept as "
                             "an option for direct before/after comparison.")
    args = parser.parse_args()

    random.seed(args.seed)

    global ENABLE_PHOTOMETRIC_NORMALIZATION
    ENABLE_PHOTOMETRIC_NORMALIZATION = not args.disable_photometric_norm
    print(f"Photometric normalization: {'ENABLED' if ENABLE_PHOTOMETRIC_NORMALIZATION else 'DISABLED'}")

    print("="*60)
    print("PHASE 0 -- Face mask construction")
    print("="*60)
    mask = build_face_mask(args.include_secondary)
    np.save(MASK_DIR / "face_mask.npy", mask)

    print("\nLoading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    vjepa2_module = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    n_layers = len(encoder_blocks)
    vjepa2_module.eval()
    vjepa2_module.to(DEVICE)
    print(f"Encoder layers: {n_layers}")

    print("\n" + "="*60)
    print("PHASE 1 -- Sample images + collect y from stored preds")
    print("="*60)
    target_samples = load_target_images(args.target_preds_npz, args.target_zip, mask)
    general_samples = sample_general_images(
        args.general_preds_dir, args.general_zips_dir, mask,
        total_sample=args.general_sample_size, seed=args.seed,
    )

    print(f"\nTarget y: min={min(y for _,y in target_samples):.4f} "
          f"max={max(y for _,y in target_samples):.4f} "
          f"mean={np.mean([y for _,y in target_samples]):.4f}")
    print(f"General y: min={min(y for _,y in general_samples):.4f} "
          f"max={max(y for _,y in general_samples):.4f} "
          f"mean={np.mean([y for _,y in general_samples]):.4f}")

    print("\n" + "="*60)
    print("PHASE 2 -- Layer profiling")
    print("="*60)
    profile = profile_layers(vjepa2_module, encoder_blocks, n_layers,
                             target_samples, general_samples)
    target_layers = pick_top_layers(profile, args.n_layers)
    print(f"\nSelected layers: {target_layers}")

    print("\n" + "="*60)
    print("PHASE 3 -- Per-layer activation collection + direction finding")
    print("="*60)
    dirs_by_layer = {}
    n_target = len(target_samples)
    activation_cache_dir = OUT_DIR / f"raw_activations_{'norm' if ENABLE_PHOTOMETRIC_NORMALIZATION else 'unnorm'}"

    for layer_idx in target_layers:
        print(f"\nCollecting activations at L{layer_idx}...")
        X, y = collect_layer_activations(vjepa2_module, encoder_blocks, layer_idx,
                                         target_samples, general_samples,
                                         cache_dir=activation_cache_dir)
        print(f"  X.shape={X.shape} y range=[{y.min():.4f},{y.max():.4f}]")
        if args.use_weighted_pca_only:
            dirs = find_directions(X, y, args.n_components, f"L{layer_idx}")
        else:
            dirs = find_directions_contrastive(X, y, n_target, args.n_components, f"L{layer_idx}")
        dirs_by_layer[layer_idx] = dirs
        np.save(OUT_DIR / f"directions_L{layer_idx}.npy", dirs)
        del X, y
        free()

    print("\n" + "="*60)
    print("PHASE 4 -- Surgery")
    print("="*60)
    apply_surgery(encoder_blocks, dirs_by_layer, args.tolerance)

    # HF-format checkpoint (config.json + weights) -- REQUIRED for validation.
    # TribeModel's video_feature.image extractor reconstructs a fresh vjepa2
    # model internally from model_name on every predict() call (confirmed via
    # smoke_test_model_identity.py + smoke_test_model_name_redirect.py) -- a
    # raw state_dict alone can't be loaded via that path, since the extractor
    # calls AutoModel.from_pretrained(model_name), not load_state_dict().
    hf_checkpoint_dir = OUT_DIR / "vjepa2_hf_checkpoint"
    vjepa2_module.save_pretrained(hf_checkpoint_dir, max_shard_size=args.max_shard_size)
    print(f"\n  HF-format checkpoint (sharded, max_shard_size={args.max_shard_size}) "
          f"for validation.py's model_name redirect -> {hf_checkpoint_dir}")
    shard_files = sorted(hf_checkpoint_dir.glob("model*.safetensors"))
    print(f"  {len(shard_files)} shard file(s):")
    for f in shard_files:
        print(f"    {f.name}: {f.stat().st_size/1e6:.2f} MB")

    print(f"  Saving (unmodified) video processor config alongside it...")
    from transformers import AutoVideoProcessor
    processor = AutoVideoProcessor.from_pretrained(args.source_model_name)
    processor.save_pretrained(hf_checkpoint_dir)
    print(f"  Processor config saved -> {hf_checkpoint_dir} "
          f"(this directory is now validation-ready on its own)")

    surgery_log = {
        "mask_rois": "OFA+FFA+TP+ATL" if args.include_secondary else "OFA+FFA",
        "n_mask_verts": int(mask.sum()),
        "tolerance": args.tolerance,
        "n_components": args.n_components,
        "n_layers": args.n_layers,
        "layers_operated": target_layers,
        "n_target_images": len(target_samples),
        "n_general_images": len(general_samples),
    }
    with open(OUT_DIR / "surgery_log.json", "w") as f:
        json.dump(surgery_log, f, indent=2)
    print(f"  Log -> {OUT_DIR / 'surgery_log.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
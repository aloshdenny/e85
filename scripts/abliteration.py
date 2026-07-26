"""
abliteration.py

Abliterate a specific person's face identity from the vjepa2 encoder used by
TribeV2, targeting face-selective cortical ROIs specifically (not the full
31-ROI table from layer_analysis.py / abliteration.py, which was built for
content-category work like porn/food addiction).

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

DATA LAYOUT ASSUMPTIONS:
  --general-preds-dir   folder of category .npz (from infer_fairface_bulk.py),
                         default: ./fairface_preds
  --general-zips-dir    folder of matching category .zip files, default: ./fairface
  --target-preds-npz    target person's .npz file or directory, default: ./target_preds
  --target-zip          zip file or directory containing target raw images, default: ./target

Usage:
  python scripts/abliteration.py --tolerance -1.0 --n_components 3 --n_layers 6
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

try:
    sys.path.append(str(Path(__file__).parent))
    import chunk_utils
    HAVE_CHUNK_UTILS = True
except ImportError:
    HAVE_CHUNK_UTILS = False
    print("[WARN] chunk_utils not found -- will save with plain torch.save instead.")

# ── Paths ────────────────────────────────────────────────────────────────────

GENERAL_PREDS_DIR = Path("./fairface_preds")   # npz predictions per demographic bucket
GENERAL_ZIPS_DIR  = Path("./fairface")          # raw image zips per demographic bucket
TARGET_PREDS_DIR  = Path("./target_preds")      # target person npz predictions (/**/.npz)
TARGET_DIR        = Path("./target")            # target person image zips (/**/.zip)
CACHE_DIR         = Path("./cache")
OUT_DIR           = Path("./abliterated_face")
MASK_DIR          = OUT_DIR / "masks"

OUT_DIR.mkdir(exist_ok=True)
MASK_DIR.mkdir(exist_ok=True)

# ── ROI definitions (Destrieux exact labels) ─────────────────────────────────

PRIMARY_FACE_ROIS = {
    "OFA": ["G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "Pole_occipital"],
    "FFA": ["G_oc-temp_lat-fusifor"],
}
SECONDARY_FACE_ROIS = {
    "TP":  ["Pole_temporal"],
    "ATL": ["G_temporal_inf", "G_oc-temp_med-Parahip"],
}

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

def image_to_vjepa_input(img_bgr):
    """
    img_bgr: raw decoded image (H,W,3) BGR uint8, e.g. from cv2.imdecode.
    Returns a (CLIP_FRAMES, 3, IMG_SIZE, IMG_SIZE) tensor, ImageNet-normalized,
    same frame repeated -- since we're hooking vjepa2 directly (not going
    through TribeModel.predict()), there's no offset/duration constraint to
    satisfy, just vjepa2's own expected clip length.
    """
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
    npz_files = sorted(general_preds_dir.glob("*.npz"))
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
        data = np.load(npz_path)
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
    preds_path = Path(target_preds_npz)
    if preds_path.is_dir():
        npz_files = sorted(preds_path.rglob("*.npz"))
    elif preds_path.is_file():
        npz_files = [preds_path]
    else:
        npz_files = sorted(Path(".").rglob(str(preds_path)))

    if not npz_files:
        raise FileNotFoundError(f"No target .npz files found matching {target_preds_npz}")

    zip_path = Path(target_zip)
    if zip_path.is_dir():
        zip_files = sorted(zip_path.rglob("*.zip"))
    elif zip_path.is_file():
        zip_files = [zip_path]
    else:
        zip_files = sorted(Path(".").rglob(str(target_zip)))

    if not zip_files:
        raise FileNotFoundError(f"No target .zip files found matching {target_zip}")

    samples = []
    for npz_file in npz_files:
        data = np.load(npz_file)
        preds = data["preds"]
        filenames = data["filenames"]

        # find matching zip by stem, or fallback to first available zip
        matching_zips = [z for z in zip_files if z.stem == npz_file.stem]
        zf_path = matching_zips[0] if matching_zips else zip_files[0]

        with zipfile.ZipFile(zf_path, "r") as zf:
            for i, name in enumerate(filenames):
                try:
                    img = decode_image_from_zip(zf, str(name))
                    y = float(preds[i][mask].mean())
                    samples.append((img, y))
                except Exception as e:
                    print(f"  [WARN] target/{name}: {e}")

    print(f"  Loaded {len(samples)} target images across {len(npz_files)} file(s)")
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
                               target_samples, general_samples):
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
    return np.stack(X), np.array(y_list, dtype=np.float32)


# ── Direction finding (reused verbatim from abliteration.py) ─────────────────

def find_directions(X, y, n_components, label):
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
    parser.add_argument("--general-preds-dir", default=GENERAL_PREDS_DIR, type=Path,
                        help="Folder of category .npz files (default: ./fairface_preds)")
    parser.add_argument("--general-zips-dir", default=GENERAL_ZIPS_DIR, type=Path,
                        help="Folder of matching category .zip files (default: ./fairface)")
    parser.add_argument("--target-preds-npz", default=TARGET_PREDS_DIR, type=Path,
                        help="Target person .npz file or directory (default: ./target_preds)")
    parser.add_argument("--target-zip", default=TARGET_DIR, type=Path,
                        help="Zip file or directory containing target raw images (default: ./target)")
    parser.add_argument("--cache-folder", default=CACHE_DIR, type=Path)
    parser.add_argument("--tolerance", type=float, default=-1.0,
                        help="-1=full suppression, 0=neutral, +1=amplify")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--include-secondary", action="store_true",
                        help="Include TP+ATL in the face mask (small positive margin "
                             "per the diagnostic, excluded by default).")
    parser.add_argument("--general-sample-size", type=int, default=2000,
                        help="Total general-population images to sample for activation "
                             "collection (spread across category buckets).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

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
    for layer_idx in target_layers:
        print(f"\nCollecting activations at L{layer_idx}...")
        X, y = collect_layer_activations(vjepa2_module, encoder_blocks, layer_idx,
                                         target_samples, general_samples)
        print(f"  X.shape={X.shape} y range=[{y.min():.4f},{y.max():.4f}]")
        dirs = find_directions(X, y, args.n_components, f"L{layer_idx}")
        dirs_by_layer[layer_idx] = dirs
        np.save(OUT_DIR / f"directions_L{layer_idx}.npy", dirs)
        del X, y
        free()

    print("\n" + "="*60)
    print("PHASE 4 -- Surgery")
    print("="*60)
    apply_surgery(encoder_blocks, dirs_by_layer, args.tolerance)

    tag = f"t{args.tolerance}_c{args.n_components}_L{len(target_layers)}"
    out_name = f"vjepa2_face_abliterated_{tag}.pt"
    if HAVE_CHUNK_UTILS:
        chunk_utils.save_chunked(vjepa2_module.state_dict(), OUT_DIR / out_name)
        chunk_utils.save_chunked(vjepa2_module.state_dict(), OUT_DIR / "vjepa2_face_abliterated.pt")
        print(f"\n  Saved (chunked) -> {OUT_DIR / out_name}")
    else:
        torch.save(vjepa2_module.state_dict(), OUT_DIR / out_name)
        torch.save(vjepa2_module.state_dict(), OUT_DIR / "vjepa2_face_abliterated.pt")
        print(f"\n  Saved -> {OUT_DIR / out_name}")

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
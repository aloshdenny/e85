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
  here -- it's read directly from the already-saved per-image preds under
  fairface + ffhq preds/ and target_preds/*.npz (mask-averaged per image),
  since those already contain the full predicted vertex vector.

DATA LAYOUT (merged FairFace+FFHQ baseline):
  --general-preds-dir   folder of per-image .npz
                         each with 'preds' (20484,) and 'filename' (str)
  --general-zip         single zip of the matching images (may be chunked as
                         {stem}_chunk_NNN.zip; fused automatically)
  --target-preds-npz    target person's .npz
                         'preds' (N,20484) + 'filenames' (N,)  OR per-image form
  --target-zip          zip of the target person's raw images

Usage:
  python scripts/abliteration.py \
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
from chunk_utils import (
    discover_npz,
    load_npz,
    preds_as_image_vectors,
    npz_image_names,
    ensure_fused_zip,
    resolve_zip_member,
)

GENERAL_PREDS_DIR = Path("./fairface + ffhq preds")
GENERAL_ZIP = Path("./fairface + ffhq/fairface + ffhq.zip")
TARGET_PREDS_NPZ = Path("./target_preds/mia.npz")
TARGET_ZIP = Path("./target/mia.zip")

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
    member = resolve_zip_member(zf, name)
    data = zf.read(member)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode {name}")
    return img


# ── Sample general-population images matched to their npz preds ────────────

def sample_general_images(general_preds_dir: Path, general_zip: Path,
                           mask: np.ndarray, total_sample: int, seed: int = 0):
    """
    Returns list of (img_bgr, y) tuples, y = preds[mask].mean() for that image.

    Expects the merged flat layout: one .npz per image under general_preds_dir,
    and one zip (possibly chunked) of the matching images. Samples uniformly
    across the npz pool (no demographic bucket budget).
    """
    rng = random.Random(seed)
    npz_files = discover_npz(general_preds_dir)
    if not npz_files:
        raise FileNotFoundError(f"No npz files in {general_preds_dir}")

    zip_path = ensure_fused_zip(general_zip)
    n_take = min(total_sample, len(npz_files))
    chosen = rng.sample(npz_files, n_take)
    print(f"Sampling {n_take}/{len(npz_files)} general-pop images from {zip_path.name}...")

    samples = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for npz_path in chosen:
            try:
                data = load_npz(npz_path)
                preds = preds_as_image_vectors(data["preds"])
                names = npz_image_names(data)
                if preds.shape[0] != len(names):
                    raise ValueError(
                        f"{npz_path.name}: preds rows {preds.shape[0]} != "
                        f"names {len(names)}"
                    )
                for i, name in enumerate(names):
                    img = decode_image_from_zip(zf, name)
                    y = float(preds[i][mask].mean())
                    samples.append((img, y))
            except Exception as e:
                print(f"  [WARN] {npz_path.name}: {e}")

    print(f"  Collected {len(samples)} general-population samples")
    return samples


def load_target_images(target_preds_npz: Path, target_zip: Path, mask: np.ndarray):
    """
    Load target images. Accepts multi-image npz (preds (N,20484) + filenames)
    or a single per-image npz (preds (20484,) + filename).
    """
    zip_path = ensure_fused_zip(target_zip)
    data = load_npz(target_preds_npz)
    preds = preds_as_image_vectors(data["preds"])
    names = npz_image_names(data)
    if preds.shape[0] != len(names):
        raise ValueError(
            f"{target_preds_npz}: preds rows {preds.shape[0]} != names {len(names)}"
        )

    samples = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for i, name in enumerate(names):
            try:
                img = decode_image_from_zip(zf, name)
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


# ── Held-out generalization search ────────────────────────────────────────────
# Replaces pure |r| profiling as the FINAL layer-selection criterion. The L37/38
# regression (worse selectivity than layers 4-6 despite a HIGHER |r| score, and
# a visibly flatter residual singular-value spectrum + more sign-flips at those
# layers) is the textbook symptom of a proxy metric that doesn't check
# generalization. |r| profiling is still used here, but only as a cheap first
# pass to narrow ~40 layers down to a manageable CANDIDATE POOL -- the actual
# decision comes from held-out contrastive separation: does the direction we'd
# extract at this layer, using ONLY train-split data, still separate held-out
# target from held-out general? A layer whose train separation is strong but
# whose held-out separation collapses is exactly what should be excluded, and
# |r| profiling alone cannot detect that difference.

def precompute_candidate_directions(vjepa2_module, encoder_blocks, candidate_pool,
                                    target_samples, general_samples, n_components,
                                    use_weighted_pca_only, cache_dir,
                                    direction_method="lda", lda_shrink=0.1):
    """
    Direction extraction for every candidate layer, using ALL target/general
    samples. No held-out split -- heldout_sep and decoy_sep both turned out
    to correlate with layer depth rather than identity-specificity. The only
    signal that has ever agreed with real selectivity is Tier-2 confirmation
    itself, so that's what scores candidates below, not a vjepa2-space proxy.
    """
    n_target = len(target_samples)
    dirs_by_layer = {}
    print(f"\nPrecomputing directions for {len(candidate_pool)} candidate layers "
          f"(train on all {len(target_samples)} target + {len(general_samples)} general "
          f"images -- scoring happens via real Tier-2 runs below, not a held-out split)...")
    for layer_idx in candidate_pool:
        X, y = collect_layer_activations(vjepa2_module, encoder_blocks, layer_idx,
                                         target_samples, general_samples, cache_dir=cache_dir)
        method = "wpca" if use_weighted_pca_only else direction_method
        dirs = find_directions_for_method(method, X, y, n_target, n_components,
                                          f"L{layer_idx}", lda_shrink)
        dirs_by_layer[layer_idx] = dirs
        del X, y
        free()
    return dirs_by_layer


def find_directions_lda(X, y, n_target, n_components, label, shrink: float = 0.1):
    """
    Mahalanobis discriminant: q = (Sigma + lam I)^-1 (mu_target - mu_general),
    Sigma pooled within-group, lam a fraction of its mean eigenvalue.

    WHY THIS REPLACES DIFFERENCE-OF-MEANS. analyze_directions.py measured, on
    the cached activations at every depth, two things about the old
    difference-of-means direction:

      top20PC_leak = 0.83 - 1.00   it lies almost entirely inside the general
                                   face population's dominant variance subspace
      AUC_heldout  = 0.84 - 0.99   it does separate target from general

    Both at once is the whole problem. The surgery deletes the component of
    every token along q, so a q that lives in generic-face variance deletes
    generic face processing -- which is exactly what ab_surgery_modes.py
    measured end-to-end: target and general responses moved by nearly identical
    amounts in every surgery mode at every depth. Difference-of-means is the
    optimal discriminant only under isotropic noise; a ViT residual stream is
    violently anisotropic, so it collapses onto the top shared PCs.

    Dividing by the covariance fixes exactly that. Same measurement on the same
    activations gives lda top20PC_leak = 0.000 - 0.014 with AUC_heldout = 1.000
    from L10 up: it separates target from general perfectly while being
    essentially orthogonal to the structure that must be preserved.

    Secondary components (n_components > 1) come from deflation -- remove q1
    from the data and re-solve -- so each is the best remaining discriminant and
    is orthogonal to its predecessors by construction. Notably NOT from
    find_directions()'s weighted PCA, whose components measure top20PC_leak
    ~0.99: they are generic face variance almost by definition.
    """
    X = np.asarray(X, dtype=np.float64)
    Xt, Xg = X[:n_target], X[n_target:]
    if len(Xt) < 2 or len(Xg) < 2:
        raise ValueError(f"[{label}] need >=2 samples per group for a covariance estimate")

    dirs = []
    Xt_d, Xg_d = Xt.copy(), Xg.copy()
    for i in range(n_components):
        mu_t, mu_g = Xt_d.mean(0), Xg_d.mean(0)
        Xc = np.concatenate([Xt_d - mu_t, Xg_d - mu_g], 0)
        S = (Xc.T @ Xc) / max(len(Xc) - 2, 1)
        lam = shrink * np.trace(S) / S.shape[0]
        q = np.linalg.solve(S + lam * np.eye(S.shape[0]), mu_t - mu_g)
        n = np.linalg.norm(q)
        if n < 1e-12:
            print(f"  [{label}] component {i} degenerate, stopping at {len(dirs)}")
            break
        q /= n
        dirs.append(q)

        pt, pg = Xt @ q, Xg @ q
        print(f"  [{label}] lda dir {i}: sep={pt.mean() - pg.mean():+.4f}  "
              f"|proj| target={np.abs(pt).mean():.3f} general={np.abs(pg).mean():.3f}  "
              f"proj_ratio={np.abs(pt).mean() / max(np.abs(pg).mean(), 1e-12):.2f}")

        # Deflate so the next solve cannot rediscover this direction.
        Xt_d = Xt_d - np.outer(Xt_d @ q, q)
        Xg_d = Xg_d - np.outer(Xg_d @ q, q)

    return np.stack(dirs).astype(np.float32)


def find_directions_facenet(X, y, n_target, n_components, label,
                            attrs_path=None, alpha_frac=0.1):
    """
    Direction taken from the part of activation space that provably encodes
    IDENTITY, rather than from a target-vs-general contrast.

    Every contrast-derived direction failed the facenet referee (r = -0.06..+0.08
    with held-out facenet target-similarity) INCLUDING the LDA one that separates
    held-out target photos at AUC 1.000. Perfect separation with no identity
    correlation means the separated thing is the target's photographic
    provenance, which is perfectly confounded with the target because all its
    photos come from one source.

    But identity is present: a ridge from these same pooled activations to
    facenet's VGGFace2 embedding retrieves the right face out of a ~391-image
    held-out fold 28.1% of the time at L35 against 0.26% chance. The contrast
    simply never pointed at it.

    So: fit activations -> facenet on GENERAL rows only, so the map cannot absorb
    the target's provenance; take the target axis in facenet space, where it
    means "looks like this person" and is trained to be invariant to lighting,
    pose and camera; pull it back through the map. Scores r_facenet_ho = +0.33 at
    L30 with top-20-PC leak 0.000.

    Extra components come from deflation, same as the LDA path.
    """
    from pathlib import Path as _P
    attrs_path = _P(attrs_path) if attrs_path else (OUT_DIR / "attributes.npz")
    if not attrs_path.exists():
        raise FileNotFoundError(
            f"{attrs_path} not found -- run scripts/face_attributes.py first; it "
            f"produces the facenet embeddings aligned row-for-row with the "
            f"activation cache.")
    d = np.load(attrs_path, allow_pickle=True)
    F, det = d["facenet"].astype(np.float64), d["detected"]
    if len(F) != len(X):
        raise ValueError(f"[{label}] attributes has {len(F)} rows but X has {len(X)} -- "
                         f"these must be the same images in the same order.")

    X = np.asarray(X, dtype=np.float64)
    gen = np.zeros(len(X), dtype=bool)
    gen[n_target:] = True
    fit_rows = gen & det
    a_t = F[:n_target][det[:n_target]].mean(0) - F[fit_rows].mean(0)
    a_t /= max(np.linalg.norm(a_t), 1e-12)

    dirs = []
    Xd = X.copy()
    for i in range(n_components):
        Xf = Xd[fit_rows]
        mu, sd = Xf.mean(0), np.maximum(Xf.std(0), 1e-9)
        Xs = (Xf - mu) / sd
        G = Xs.T @ Xs
        alpha = alpha_frac * np.trace(G) / G.shape[0]
        W = np.linalg.solve(G + alpha * np.eye(G.shape[0]), Xs.T @ (F[fit_rows] - F[fit_rows].mean(0)))
        q = (W @ a_t) / sd
        n = np.linalg.norm(q)
        if n < 1e-12:
            print(f"  [{label}] facenet component {i} degenerate, stopping at {len(dirs)}")
            break
        q /= n
        dirs.append(q)

        pt, pg = X[:n_target] @ q, X[n_target:] @ q
        print(f"  [{label}] facenet dir {i}: sep={pt.mean() - pg.mean():+.4f}  "
              f"|proj| target={np.abs(pt).mean():.3f} general={np.abs(pg).mean():.3f}  "
              f"proj_ratio={np.abs(pt).mean() / max(np.abs(pg).mean(), 1e-12):.2f}")
        Xd = Xd - np.outer(Xd @ q, q)

    return np.stack(dirs).astype(np.float32)


DIRECTION_METHODS = ("contrastive", "lda", "wpca", "facenet")


def find_directions_for_method(method, X, y, n_target, n_components, label, shrink=0.1):
    if method == "facenet":
        return find_directions_facenet(X, y, n_target, n_components, label)
    if method == "lda":
        return find_directions_lda(X, y, n_target, n_components, label, shrink)
    if method == "wpca":
        return find_directions(X, y, n_components, label)
    if method == "contrastive":
        return find_directions_contrastive(X, y, n_target, n_components, label)
    raise ValueError(f"Unknown direction method {method!r}")


# ── Surgery ──────────────────────────────────────────────────────────────────
# Two edits that look symmetric but live in DIFFERENT vector spaces. `q` is
# found by hooking VJEPA2Layer's OUTPUT, so it lives in the RESIDUAL STREAM
# basis, and each edit is only meaningful where that basis actually applies:
#
#   read-block  (input side)   W <- W (I - q qᵀ)
#     Valid where the matrix READS the residual stream. In VJEPA2Layer that is
#     attention.{query,key,value} = nn.Linear(hidden, all_head_size), fed
#     norm1(residual). Stops the layer attending on the identity direction.
#
#   write-block (output side)  W <- (I - q qᵀ) W,  b <- b - (b·q) q
#     Valid where the matrix WRITES the residual stream. In VJEPA2Layer that is
#     attention.proj and mlp.fc2, both nn.Linear(*, hidden), whose outputs are
#     added straight back onto the residual. Stops the layer putting the
#     identity direction back into the stream. This is the canonical
#     abliteration edit.
#
# The previous version applied the READ form to attention.proj. proj's input is
# the concatenated-head basis, NOT the residual basis, so a residual-space q
# there is an arbitrary rank-1 perturbation: it damages general face processing
# without specifically removing the identity direction, which is exactly the
# non-selectivity ("general drops as much as target") this project kept hitting.
# Preserved as mode="legacy" so the two can be A/B'd rather than argued about.
#
# mlp.fc2 is new to "write"/"both": it is the layer's OTHER residual writer, so
# leaving it alone lets the MLP re-inject the direction attention no longer
# writes.

SURGERY_MODES = {
    "legacy": [("attention.value", "read"), ("attention.proj", "read")],
    "read":   [("attention.value", "read")],
    "write":  [("attention.proj", "write"), ("mlp.fc2", "write")],
    "both":   [("attention.value", "read"),
               ("attention.proj", "write"), ("mlp.fc2", "write")],
}
DEFAULT_SURGERY_MODE = "both"


def _orthonormalize(dirs, device):
    dirs_t = torch.tensor(dirs, dtype=torch.float32).to(device)
    ortho = []
    for d in dirs_t:
        for q in ortho:
            d = d - (d @ q) * q
        n = d.norm()
        if n > 1e-6:
            ortho.append(d / n)
    return torch.stack(ortho) if ortho else None


def apply_surgery(encoder_blocks, all_dirs_by_layer, tolerance,
                  mode: str = DEFAULT_SURGERY_MODE, verbose: bool = True):
    if mode not in SURGERY_MODES:
        raise ValueError(f"Unknown surgery mode {mode!r}, expected one of {sorted(SURGERY_MODES)}")
    targets = SURGERY_MODES[mode]

    for layer_idx, dirs in all_dirs_by_layer.items():
        ortho = _orthonormalize(dirs, DEVICE)
        if ortho is None:
            print(f"  [WARN] No valid directions for layer {layer_idx}, skipping")
            continue
        block = encoder_blocks[layer_idx]
        if verbose:
            print(f"\n  Surgery on encoder.layer[{layer_idx}] -- {len(ortho)} direction(s), "
                  f"tolerance={tolerance}, mode={mode}")
        for attr_path, side in targets:
            mod = block
            for part in attr_path.split("."):
                mod = getattr(mod, part)
            W = mod.weight.data.clone()
            for q in ortho:
                if side == "read":
                    # remove q from the INPUT space: W (I - q qᵀ)
                    W += tolerance * torch.outer(W @ q, q)
                else:
                    # remove q from the OUTPUT space: (I - q qᵀ) W
                    W += tolerance * torch.outer(q, q @ W)
            mod.weight.data = W
            if side == "write" and getattr(mod, "bias", None) is not None:
                b = mod.bias.data.clone()
                for q in ortho:
                    b += tolerance * (b @ q) * q
                mod.bias.data = b
            if verbose:
                print(f"    {attr_path}: {tuple(W.shape)} {side}-blocked")


# ── neuralset dispatches on the model_name STRING, not on the checkpoint ──
# HuggingFaceVideo._get_data (neuralset/extractors/video.py) routes a video
# model to the real video path only if the model_name contains one of
# _HFVideoModel.MODELS:
#
#     if not any(z in self.image.model_name for z in _HFVideoModel.MODELS):
#         yield from self._get_data_from_image_model(events)
#
# and _HFVideoModel.predict() likewise picks the "videos" kwarg (rather than
# "images") via `"vjepa2" in self.model_name`. Our redirect replaces
# model_name with a local DIRECTORY PATH, so that path must still contain the
# family token. When it does not, the vjepa2 checkpoint is silently driven
# through the per-frame IMAGE path, which calls processor(images=...) on a
# VJEPA2VideoProcessor and dies with:
#
#     TypeError: BaseVideoProcessor.__call__() missing 1 required
#                positional argument: 'videos'
#
# HuggingFaceVideo.model_post_init additionally rejects any model_name
# containing "video" (unless it is "videomae"), so the path must not
# introduce that substring either -- checked here too, since a rename of an
# ancestor directory is enough to trip it.
MODEL_FAMILY_TOKEN = "vjepa2"


def assert_redirectable_path(path: Path, what: str = "checkpoint dir") -> str:
    """Returns the resolved path string to hand to redirect_model_name(),
    after verifying neuralset's substring dispatch will still route it to the
    video path. Fails loudly here rather than 20 minutes into a search."""
    resolved = str(Path(path).resolve())
    if MODEL_FAMILY_TOKEN not in resolved:
        raise ValueError(
            f"{what} {resolved!r} does not contain {MODEL_FAMILY_TOKEN!r}. "
            f"neuralset picks the video vs. image code path by substring-matching "
            f"model_name, so a redirect target without that token is processed "
            f"frame-by-frame as images and crashes in the video processor. "
            f"Rename the directory to include {MODEL_FAMILY_TOKEN!r}."
        )
    if "video" in resolved.replace("videomae", ""):
        raise ValueError(
            f"{what} {resolved!r} contains 'video', which makes "
            f"HuggingFaceVideo.model_post_init raise NotImplementedError "
            f"(it only whitelists 'videomae'). Rename the directory."
        )
    return resolved


def pick_trial_scratch_root(out_dir: Path, explicit: Path | None = None) -> Path:
    """Where to write the per-trial checkpoints. Defaults to out_dir, EXCEPT
    when out_dir sits on a WSL DrvFs mount (/mnt/c/...): that is the Windows
    disk seen through a translation layer -- slow for the multi-GB write every
    trial does, and on this box it also runs at ~100% full. Falls back to the
    native ext4 home, which has room and is an order of magnitude faster."""
    if explicit is not None:
        root = Path(explicit)
    else:
        resolved = out_dir.resolve()
        on_drvfs = str(resolved).startswith("/mnt/") and len(resolved.parts) > 2
        root = (Path.home() / "e85_trial_scratch") if on_drvfs else resolved
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_trial_checkpoint(vjepa2_module, source_model_name, trial_dir: Path,
                          original_snapshot_dir: Path, max_shard_size="20MB"):
    """
    Builds a trial checkpoint directory. ROOT CAUSE (confirmed via
    smoke_test_processor_resolution.py's Check 2, which proved local-
    directory redirection to the ORIGINAL unmodified snapshot works fine --
    ruling out model_name redirection itself as the problem): copying
    config.json from the original snapshot wasn't sufficient, because
    vjepa2_module.save_pretrained() OVERWRITES config.json afterward with
    its own version (a sub-component's save, not the full original repo's
    config), and that overwritten version is what broke Auto-class/
    processor resolution -- not file copying, not model_name redirection,
    not caching. Two prior fixes both missed this because they focused on
    what gets copied BEFORE save_pretrained(), when the actual divergence
    happens AFTER it.

    Fix: after save_pretrained() writes weights + its own config.json,
    RESTORE the original snapshot's config.json over it. This guarantees
    trial_dir's config.json is byte-identical to the exact file that Check 2
    proved resolves correctly, no matter what save_pretrained() does.

    Also skips copying large weight files up front (model.safetensors,
    original/model.pth, etc.) -- those get written fresh by
    save_pretrained() anyway, and copying the multi-GB original/model.pth
    every trial was the direct cause of a separate real I/O error seen
    earlier on a WSL /mnt/c path.
    """
    import shutil
    trial_dir.mkdir(parents=True, exist_ok=True)

    skip_patterns = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5")
    skip_dirs = {"original"}  # holds the raw upstream checkpoint, never needed here

    copied, skipped = 0, 0
    for item in original_snapshot_dir.iterdir():
        if item.is_dir():
            if item.name in skip_dirs:
                skipped += 1
                continue
            shutil.copytree(item, trial_dir / item.name, dirs_exist_ok=True)
            copied += 1
        else:
            if item.name.lower().endswith(skip_patterns):
                skipped += 1
                continue
            shutil.copy2(item, trial_dir / item.name)
            copied += 1

    # Writes fresh weight shards from our modified weights -- but ALSO
    # writes its own config.json, which is the actual root cause below.
    vjepa2_module.save_pretrained(trial_dir, max_shard_size=max_shard_size)

    # THE FIX: restore the ORIGINAL config.json, proven correct by
    # smoke_test_processor_resolution.py's Check 2, overwriting whatever
    # save_pretrained() just wrote there.
    original_config = original_snapshot_dir / "config.json"
    if original_config.exists():
        shutil.copy2(original_config, trial_dir / "config.json")

    return copied, skipped


def greedy_tier2_search(vjepa2_module, encoder_blocks, dirs_by_layer, candidate_pool,
                        n_layers_wanted, tolerance, val_dir: Path, target_prefix: str,
                        mask, cache_folder: Path, out_dir: Path, source_model_name: str,
                        min_effect_size: float = 0.001,
                        surgery_mode: str = DEFAULT_SURGERY_MODE,
                        trial_scratch_root: Path | None = None,
                        max_shard_size: str = "20MB"):
    """
    Greedy forward selection scored by REAL Tier-2 confirmation (actual
    TribeModel.predict() calls), not any vjepa2-space proxy -- replaces both
    the |r|-only selection (picked L37/38, Tier-2 showed 0.92x) and the
    held-out-generalization + decoy-specificity search (picked L15-19,
    Tier-2 showed 0.91-0.92x). Both proxies correlated with layer DEPTH, not
    identity-specificity.

    Each round: for every remaining candidate, apply surgery for
    (already_selected + candidate), build a trial checkpoint (see
    make_trial_checkpoint -- copies the known-working processor files
    verbatim, only weights are ours), redirect a confirmation model to it,
    measure the ACTUAL selectivity ratio on val_dir images. Add whichever
    candidate gives the best ratio this round; stop early if nothing
    improves rather than forcing n_layers_wanted regardless.
    """
    sys.path.append(str(Path(__file__).parent))
    from validation import (
        get_tmp_root, discover_val_images, build_clips_once,
        run_predict_on_clips, redirect_model_name,
    )
    from huggingface_hub import snapshot_download
    import tempfile, shutil

    target_paths, general_paths = discover_val_images(val_dir, target_prefix)
    if not target_paths:
        raise ValueError(f"No images matched prefix '{target_prefix}' in {val_dir} -- "
                         f"greedy Tier-2 search requires real val images to score against.")
    print(f"\nGreedy Tier-2 search: {len(target_paths)} target, {len(general_paths)} general "
          f"images in {val_dir}, {len(candidate_pool)} candidates, up to {n_layers_wanted} layers")

    # Fail fast: every trial redirects model_name to a directory under the
    # scratch root, and neuralset routes video-vs-image by substring on that
    # string. Check the shape of that path now rather than after the
    # (expensive) baseline pass.
    scratch_root = pick_trial_scratch_root(out_dir, trial_scratch_root)
    assert_redirectable_path(scratch_root / f"{MODEL_FAMILY_TOKEN}_search_trial_probe", "trial dir")
    print(f"Trial checkpoints -> {scratch_root} (shard size {max_shard_size}), surgery mode "
          f"{surgery_mode!r}")

    tmp_root = get_tmp_root()
    all_paths = target_paths + general_paths

    print("Computing baseline (pre-surgery) predictions once, reused across every trial...")
    base_tmp_dir = Path(tempfile.mkdtemp(prefix="greedy_base_", dir=tmp_root))
    try:
        base_rows = build_clips_once(all_paths, base_tmp_dir, duration=1.0, fps=2)
        model_before = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)
        preds_before = run_predict_on_clips(model_before, base_rows, duration=1.0)
        del model_before
        free()
    finally:
        shutil.rmtree(base_tmp_dir, ignore_errors=True)

    print(f"Resolving original snapshot for {source_model_name} (local HF cache, fast if "
          f"already downloaded) -- used as the known-working processor template for every "
          f"trial, never round-tripped through AutoVideoProcessor.save_pretrained()...")
    original_snapshot_dir = Path(snapshot_download(repo_id=source_model_name))

    # NOTE: no shared confirmation model here on purpose. A TribeModel instance
    # can only be redirected reliably BEFORE its first predict() call -- after
    # that the pydantic config tree freezes, redirect_model_name() silently
    # falls back to rebuilding a parent node, and the extractor that predict()
    # actually uses keeps pointing at the PREVIOUS trial's checkpoint. Reusing
    # one instance across trials therefore does not merely crash when that
    # directory is gone (which is how it was caught); while the directory still
    # exists it scores the wrong weights and reports a plausible ratio. Every
    # trial gets a fresh instance, and check_redirect_took_effect() below turns
    # any remaining staleness into a hard failure instead of a quiet wrong number.

    original_state = {k: v.clone() for k, v in vjepa2_module.state_dict().items()}
    trial_dirs_created = []

    def summarize(preds_after, paths):
        deltas = []
        for p in paths:
            name = p.name
            if name not in preds_before or name not in preds_after:
                continue
            deltas.append(float(preds_after[name][mask].mean()) - float(preds_before[name][mask].mean()))
        return np.mean(deltas) if deltas else 0.0

    last_preds = {"tag": None, "preds": None}

    def check_redirect_took_effect(preds_after, trial_tag):
        """A stale redirect does not announce itself -- it returns perfectly
        well-formed predictions belonging to the wrong weights. Two cheap
        invariants catch it: surgery always perturbs the encoder, so the
        predictions must differ from the pre-surgery baseline, and no two
        consecutive trials edit the same layer set, so they must differ from
        the previous trial's."""
        def identical(a, b):
            common = set(a) & set(b)
            if not common:
                return False
            return all(np.array_equal(a[k], b[k]) for k in common)

        if identical(preds_after, preds_before):
            raise RuntimeError(
                f"trial {trial_tag}: predictions are bit-identical to the pre-surgery "
                f"baseline. The redirect did not take effect (or predict() served a "
                f"cached result), so this trial would score the UNMODIFIED model.")
        if last_preds["preds"] is not None and identical(preds_after, last_preds["preds"]):
            raise RuntimeError(
                f"trial {trial_tag}: predictions are bit-identical to trial "
                f"{last_preds['tag']}, which used a different layer set. The model is "
                f"stale -- it is still running the previous trial's checkpoint.")
        last_preds["tag"], last_preds["preds"] = trial_tag, preds_after

    def run_trial(trial_layers, trial_tag):
        vjepa2_module.load_state_dict(original_state)
        apply_surgery(encoder_blocks, {l: dirs_by_layer[l] for l in trial_layers},
                      tolerance, mode=surgery_mode, verbose=(trial_tag == 1))

        # Name carries MODEL_FAMILY_TOKEN on purpose -- see
        # assert_redirectable_path(): this string IS the model_name neuralset
        # substring-matches to choose the video code path.
        trial_dir = scratch_root / f"{MODEL_FAMILY_TOKEN}_search_trial_{trial_tag}_{os.getpid()}"
        trial_dirs_created.append(trial_dir)
        copied, skipped = make_trial_checkpoint(
            vjepa2_module, source_model_name, trial_dir, original_snapshot_dir,
            max_shard_size=max_shard_size)
        if trial_tag == 1:
            print(f"    [first trial] template copy: {copied} item(s) copied, "
                  f"{skipped} large/weight item(s) skipped (written fresh by "
                  f"save_pretrained() instead)")

        # Fresh instance per trial: see the note above the search loop.
        confirm_model = TribeModel.from_pretrained("facebook/tribev2",
                                                   cache_folder=cache_folder)
        redirect_model_name(confirm_model, assert_redirectable_path(trial_dir, "trial dir"))

        trial_tmp_dir = Path(tempfile.mkdtemp(prefix=f"greedy_trial_{trial_tag}_", dir=tmp_root))
        try:
            trial_rows = build_clips_once(all_paths, trial_tmp_dir, duration=1.0, fps=2)
            preds_after = run_predict_on_clips(confirm_model, trial_rows, duration=1.0)
        finally:
            shutil.rmtree(trial_tmp_dir, ignore_errors=True)
            shutil.rmtree(trial_dir, ignore_errors=True)
            del confirm_model
            free()

        check_redirect_took_effect(preds_after, trial_tag)

        target_delta = summarize(preds_after, target_paths)
        general_delta = summarize(preds_after, general_paths)
        ratio = abs(target_delta) / max(abs(general_delta), 1e-9)
        return target_delta, general_delta, ratio

    selected = []
    remaining = list(candidate_pool)
    best_ratio_so_far = 0.0
    history = []
    trial_counter = 0

    try:
        for round_num in range(n_layers_wanted):
            print(f"\n--- Round {round_num + 1}: {len(remaining)} candidates remaining, "
                  f"current set {selected} ---")
            round_results = []
            for candidate in remaining:
                trial_counter += 1
                trial_layers = selected + [candidate]
                target_delta, general_delta, ratio = run_trial(trial_layers, trial_counter)
                print(f"  {trial_layers}: target={target_delta:+.5f} general={general_delta:+.5f} "
                      f"ratio={ratio:.2f}x")
                round_results.append({
                    "candidate": candidate, "trial_layers": list(trial_layers),
                    "target_delta": target_delta, "general_delta": general_delta, "ratio": ratio,
                })
            history.append(round_results)

            viable = [r for r in round_results if abs(r["target_delta"]) >= min_effect_size]
            pool_for_best = viable if viable else round_results
            best = max(pool_for_best, key=lambda r: r["ratio"])

            print(f"  Best this round: L{best['candidate']} -> ratio={best['ratio']:.2f}x "
                  f"(target={best['target_delta']:+.5f}, general={best['general_delta']:+.5f})")

            if best["ratio"] <= best_ratio_so_far:
                print(f"  No improvement over current best ({best_ratio_so_far:.2f}x) -- "
                      f"stopping early rather than forcing {n_layers_wanted} layers.")
                break

            selected.append(best["candidate"])
            remaining.remove(best["candidate"])
            best_ratio_so_far = best["ratio"]
    finally:
        for d in trial_dirs_created:
            shutil.rmtree(d, ignore_errors=True)

    print(f"\nFinal selected layers: {selected} (selectivity ratio: {best_ratio_so_far:.2f}x)")

    vjepa2_module.load_state_dict(original_state)
    apply_surgery(encoder_blocks, {l: dirs_by_layer[l] for l in selected}, tolerance,
                  mode=surgery_mode)
    free()

    return selected, {l: dirs_by_layer[l] for l in selected}, history, best_ratio_so_far


# ── Optional Tier-2 confirmation: real end-to-end check via TribeModel.predict() ──
# The search above operates entirely in vjepa2's own hidden-state space -- it
# never confirms the effect survives TRIBE's remaining transformer layers and
# projector. This runs ONE real validation pass (same technique validated in
# validation.py: save the surgically-modified weights as an HF checkpoint,
# redirect model_name, run model.predict() on real held-out photos) so the
# search's output comes with a genuine selectivity number, not just a
# vjepa2-space proxy -- however improved that proxy now is.

def run_tier2_confirmation(vjepa2_module, val_dir: Path, target_prefix: str,
                           hf_checkpoint_dir: Path, mask, cache_folder: Path):
    print("\n" + "="*60)
    print("TIER 2 -- Real end-to-end confirmation (via TribeModel.predict())")
    print("="*60)
    sys.path.append(str(Path(__file__).parent))
    from validation import (
        get_tmp_root, discover_val_images, build_clips_once,
        run_predict_on_clips, redirect_model_name,
    )
    import tempfile, shutil

    target_paths, general_paths = discover_val_images(val_dir, target_prefix)
    if not target_paths:
        print(f"  [WARN] no images matched prefix '{target_prefix}' in {val_dir} -- skipping.")
        return None
    print(f"  {len(target_paths)} target, {len(general_paths)} general in {val_dir}")

    tmp_root = get_tmp_root()
    tmp_dir = Path(tempfile.mkdtemp(prefix="tier2_confirm_", dir=tmp_root))
    all_paths = target_paths + general_paths

    try:
        rows = build_clips_once(all_paths, tmp_dir, duration=1.0, fps=2)

        print("  Loading model_before (baseline weights)...")
        model_before = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)
        preds_before = run_predict_on_clips(model_before, rows, duration=1.0)
        del model_before
        free()

        print("  Loading model_after (surgically-modified checkpoint)...")
        model_after = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)
        redirect_model_name(model_after, assert_redirectable_path(hf_checkpoint_dir,
                                                          "hf checkpoint dir"))
        preds_after = run_predict_on_clips(model_after, rows, duration=1.0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def summarize(paths):
        deltas = []
        for p in paths:
            name = p.name
            if name not in preds_before or name not in preds_after:
                continue
            yb = float(preds_before[name][mask].mean())
            ya = float(preds_after[name][mask].mean())
            deltas.append(ya - yb)
        return np.mean(deltas) if deltas else 0.0

    target_delta = summarize(target_paths)
    general_delta = summarize(general_paths)
    ratio = abs(target_delta) / max(abs(general_delta), 1e-9)

    print(f"\n  TARGET delta:  {target_delta:+.5f}")
    print(f"  GENERAL delta: {general_delta:+.5f}")
    print(f"  Selectivity ratio: {ratio:.2f}x")

    return {"target_delta": target_delta, "general_delta": general_delta, "ratio": ratio}


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-preds-dir", default=GENERAL_PREDS_DIR, type=Path,
                        help='Per-image general-pop .npz folder '
                             '(default: "./fairface + ffhq preds")')
    parser.add_argument("--general-zip", default=GENERAL_ZIP, type=Path,
                        help='Single zip of general-pop images '
                             '(default: "./fairface + ffhq/fairface + ffhq.zip"; '
                             'chunked {stem}_chunk_NNN.zip is fused automatically)')
    parser.add_argument("--target-preds-npz", default=TARGET_PREDS_NPZ, type=Path)
    parser.add_argument("--target-zip", default=TARGET_ZIP, type=Path)
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
                             "collection (uniform sample over per-image npzs).")
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
                             "~30%% higher contrast than the general population, and that "
                             "leaked substantially into the extracted direction. Kept as "
                             "an option for direct before/after comparison.")
    parser.add_argument("--search-candidates", type=int, default=15,
                        help="Size of the candidate layer pool, narrowed from all 40 layers "
                             "via the cheap |r| profiling pass (Phase 2). Each candidate gets "
                             "a REAL Tier-2 trial (save checkpoint + TribeModel.predict() on "
                             "val images) during the greedy search below, so this is a genuine "
                             "cost knob, not a cheap heuristic -- keep it modest (6-15) unless "
                             "you have time for many trial runs.")
    parser.add_argument("--confirm-val-dir", type=Path, required=True,
                        help="REQUIRED. Folder of val images (same format as validation.py) "
                             "used as the greedy search's actual scoring function -- every "
                             "candidate layer combination is evaluated by real "
                             "TribeModel.predict() calls on these images, not a vjepa2-space "
                             "proxy. Every proxy tried previously (|r|, held-out generalization, "
                             "decoy specificity) turned out to correlate with layer depth "
                             "rather than identity-specificity.")
    parser.add_argument("--confirm-target-prefix", default="mia",
                        help="Filename prefix identifying the target person in --confirm-val-dir.")
    parser.add_argument("--direction-method", default="lda", choices=list(DIRECTION_METHODS),
                        help="How the ablation direction is found. 'lda' (default) divides the "
                             "target-vs-general mean difference by the pooled covariance; "
                             "'contrastive' is the raw mean difference, measured to sit almost "
                             "entirely inside general face variance; 'wpca' is weighted PCA.")
    parser.add_argument("--lda-shrink", type=float, default=0.1,
                        help="Covariance shrinkage toward the identity for --direction-method lda.")
    parser.add_argument("--surgery-mode", default=DEFAULT_SURGERY_MODE,
                        choices=sorted(SURGERY_MODES),
                        help="Which weight matrices to edit, and in which space. "
                             "'both' (default) read-blocks attention.value and "
                             "write-blocks attention.proj + mlp.fc2. 'legacy' reproduces "
                             "the earlier behaviour, which read-blocked attention.proj "
                             "using a residual-space direction in the head-concat basis.")
    parser.add_argument("--trial-scratch-dir", type=Path, default=None,
                        help="Where per-trial checkpoints are written. Defaults to a native "
                             "filesystem dir when the out dir is on a /mnt WSL DrvFs mount "
                             "(slow, and typically near-full), else the out dir itself.")
    parser.add_argument("--min-effect-size", type=float, default=0.001,
                        help="Minimum |target_delta| for a trial to be considered when picking "
                             "the best candidate each round -- filters out degenerate high "
                             "ratios from near-zero deltas on both sides.")
    args = parser.parse_args()

    random.seed(args.seed)

    global ENABLE_PHOTOMETRIC_NORMALIZATION
    ENABLE_PHOTOMETRIC_NORMALIZATION = not args.disable_photometric_norm
    print(f"Photometric normalization: {'ENABLED' if ENABLE_PHOTOMETRIC_NORMALIZATION else 'DISABLED'}")

    # Both the search trials and the final checkpoint are consumed by
    # redirecting model_name at a local path; validate the naming up front.
    assert_redirectable_path(OUT_DIR / f"{MODEL_FAMILY_TOKEN}_search_trial_probe", "trial dir")
    assert_redirectable_path(OUT_DIR / f"{MODEL_FAMILY_TOKEN}_hf_checkpoint", "hf checkpoint dir")

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
        args.general_preds_dir, args.general_zip, mask,
        total_sample=args.general_sample_size, seed=args.seed,
    )

    print(f"\nTarget y: min={min(y for _,y in target_samples):.4f} "
          f"max={max(y for _,y in target_samples):.4f} "
          f"mean={np.mean([y for _,y in target_samples]):.4f}")
    print(f"General y: min={min(y for _,y in general_samples):.4f} "
          f"max={max(y for _,y in general_samples):.4f} "
          f"mean={np.mean([y for _,y in general_samples]):.4f}")

    print("\n" + "="*60)
    print("PHASE 2 -- Candidate pool (cheap |r| profiling, narrowing only)")
    print("="*60)
    profile = profile_layers(vjepa2_module, encoder_blocks, n_layers,
                             target_samples, general_samples)
    candidate_pool = pick_top_layers(profile, args.search_candidates)
    print(f"\nCandidate pool for the real search below: {candidate_pool}")

    print("\n" + "="*60)
    print("PHASE 3 -- Greedy search scored by REAL Tier-2 confirmation")
    print("="*60)
    activation_cache_dir = OUT_DIR / f"raw_activations_{'norm' if ENABLE_PHOTOMETRIC_NORMALIZATION else 'unnorm'}"
    all_candidate_dirs = precompute_candidate_directions(
        vjepa2_module, encoder_blocks, candidate_pool,
        target_samples, general_samples, args.n_components,
        args.use_weighted_pca_only, activation_cache_dir,
        direction_method=args.direction_method, lda_shrink=args.lda_shrink,
    )
    target_layers, dirs_by_layer, search_history, search_ratio = greedy_tier2_search(
        vjepa2_module, encoder_blocks, all_candidate_dirs, candidate_pool,
        args.n_layers, args.tolerance, args.confirm_val_dir, args.confirm_target_prefix,
        mask, args.cache_folder, OUT_DIR, args.source_model_name,
        min_effect_size=args.min_effect_size,
        surgery_mode=args.surgery_mode,
        trial_scratch_root=args.trial_scratch_dir,
        max_shard_size=args.max_shard_size,
    )
    for layer_idx, dirs in dirs_by_layer.items():
        np.save(OUT_DIR / f"directions_L{layer_idx}.npy", dirs)

    print("\n" + "="*60)
    print("PHASE 4 -- Surgery")
    print("="*60)
    print(f"(Surgery for the final selected layer set {target_layers} was already applied ")
    print(f" in-place at the end of the greedy search -- this phase just confirms/saves it.)")
    for layer_idx in target_layers:
        print(f"  L{layer_idx}: surgery applied, {dirs_by_layer[layer_idx].shape[0]} direction(s)")

    # HF-format checkpoint for validation.py's model_name redirect. Uses the
    # SAME snapshot-copy approach as the search trials above -- copies the
    # ORIGINAL repo's known-working processor files byte-for-byte rather than
    # round-tripping through AutoVideoProcessor.save_pretrained()/
    # from_pretrained(), which resolves to the wrong class (BaseVideoProcessor
    # instead of the VJEPA2-specific subclass) on reload from a local
    # directory -- this is the same bug that broke every search trial before
    # the fix, and would have broken this final checkpoint identically.
    from huggingface_hub import snapshot_download
    print(f"\nResolving original snapshot for {args.source_model_name} "
          f"(processor template for the final checkpoint too)...")
    original_snapshot_dir = Path(snapshot_download(repo_id=args.source_model_name))

    hf_checkpoint_dir = OUT_DIR / "vjepa2_hf_checkpoint"
    make_trial_checkpoint(vjepa2_module, args.source_model_name, hf_checkpoint_dir,
                          original_snapshot_dir, max_shard_size=args.max_shard_size)
    print(f"  HF-format checkpoint (sharded, max_shard_size={args.max_shard_size}) "
          f"for validation.py's model_name redirect -> {hf_checkpoint_dir}")
    shard_files = sorted(hf_checkpoint_dir.glob("model*.safetensors"))
    print(f"  {len(shard_files)} shard file(s):")
    for f in shard_files:
        print(f"    {f.name}: {f.stat().st_size/1e6:.2f} MB")
    print(f"  Processor files copied verbatim from the original snapshot -> "
          f"{hf_checkpoint_dir} (this directory is now validation-ready on its own)")

    tier2_result = None
    if args.confirm_val_dir is not None:
        tier2_result = run_tier2_confirmation(
            vjepa2_module, args.confirm_val_dir, args.confirm_target_prefix,
            hf_checkpoint_dir, mask, args.cache_folder,
        )

    surgery_log = {
        "mask_rois": "OFA+FFA+TP+ATL" if args.include_secondary else "OFA+FFA",
        "n_mask_verts": int(mask.sum()),
        "tolerance": args.tolerance,
        "n_components": args.n_components,
        "n_layers": args.n_layers,
        "layers_operated": target_layers,
        "n_target_images": len(target_samples),
        "n_general_images": len(general_samples),
        "search_candidate_pool": candidate_pool,
        "search_final_ratio": search_ratio,
        "search_history": [
            [{"trial_layers": t["trial_layers"], "target_delta": t["target_delta"],
              "general_delta": t["general_delta"], "ratio": t["ratio"]}
             for t in round_results]
            for round_results in search_history
        ],
        "tier2_confirmation_independent": tier2_result,
    }
    with open(OUT_DIR / "surgery_log.json", "w") as f:
        json.dump(surgery_log, f, indent=2)
    print(f"  Log -> {OUT_DIR / 'surgery_log.json'}")
    if tier2_result is not None:
        print(f"\n  Search's own final ratio during selection: {search_ratio:.2f}x")
        print(f"  Independent post-hoc confirmation ratio:    {tier2_result['ratio']:.2f}x")
        if abs(tier2_result["ratio"] - search_ratio) > 0.3:
            print(f"  [!!] These differ notably -- worth investigating why before trusting "
                  f"either number strongly.")
    print("\nDone.")


if __name__ == "__main__":
    main()
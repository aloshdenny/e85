"""
plot_postop_activity.py

Predicted brain activity in face cortex, before vs after the suppression
surgery, for each target -- next to what the same model predicts for an
ordinary person's face.

WHY BEFORE *AND* AFTER. A post-surgery map alone is unreadable: the targets
start ABOVE the general-population baseline (that elevation is the identity
signal being attacked), so after a real drop they can still sit near or above
an ordinary face. Showing only the "after" panel would make a working edit
look like a failed one. The drop is the result, so the drop is what's drawn.

THE CONTROL COLUMN. The "after" panel for ordinary faces has the SAME patch
installed (Mia's) -- ordinary faces run through an edited model. If the
surgery were sloppy this column would move too. It doesn't, and that is the
no-collateral-damage claim rendered rather than asserted.

HOW IT'S COMPUTED WITHOUT THE ENCODER. Surgery only edits the final readout:
post = X @ (W0 + U@V) + b0. Baseline predictions already on disk were made by
that same frozen readout (base = X @ W0 + b0), and W0 has full column rank, so
X is recovered exactly by least squares -- verified by round-trip here, not
assumed. Michael has no stored baseline, so his X comes from one Modal GPU
pass (scripts/modal_mj_bottleneck.py).

Usage:
  python scripts/plot_postop_activity.py                  # fetches the checkpoint
  python scripts/plot_postop_activity.py --ckpt best.ckpt # or point at a local copy
"""
import sys, argparse, random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from nilearn import datasets as nl_datasets
from nilearn import plotting as nl_plotting

sys.path.append(str(Path(__file__).parent))
from chunk_utils import load_npz, npz_exists, preds_as_image_vectors, discover_npz
from measure_identity_signal import build_masks

# Repo-relative so the script runs from a fresh clone regardless of cwd.
REPO = Path(__file__).resolve().parent.parent
ABL = REPO / "abliterated"
OUT = REPO / "analysis" / "postop_activity.png"
HEMI = "right"          # face processing is right-lateralised; one hemi keeps it readable
VIEW = "ventral"

PATCH = {
    "Mia":     ABL / "mia_suppress_readout_v2_face_top_lam15.npz",
    "Sins":    ABL / "sins_suppress_readout_v2_face_top_lam15.npz",
    "Michael": ABL / "mj_suppress_readout_v2_face_top_lam15.npz",
}
BASELINE_PREDS = {"Mia": REPO / "target_preds" / "mia.npz",
                  "Sins": REPO / "target_preds" / "sins.npz"}
MJ_BOTTLENECK = ABL / "mj_bottleneck.npz"


TRIBE_REPO = "facebook/tribev2"
TRIBE_CKPT = "best.ckpt"


def resolve_ckpt(explicit=None):
    """Return a path to TRIBE's checkpoint, downloading it if needed.

    Only the readout matrix is used here, but it lives inside the 709MB
    checkpoint. Requiring the caller to have fetched it by hand made this
    script un-runnable from a fresh clone, so fetch it on demand and let
    huggingface_hub cache it for subsequent runs.
    """
    if explicit:
        return Path(explicit)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required to fetch the checkpoint automatically.\n"
            "  pip install huggingface_hub   (or pass --ckpt /path/to/best.ckpt)")
    print(f"fetching {TRIBE_REPO}/{TRIBE_CKPT} (709MB, cached after first run)...")
    return Path(hf_hub_download(repo_id=TRIBE_REPO, filename=TRIBE_CKPT))


def load_readout(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = obj.get("state_dict", obj)
    W0 = np.squeeze(sd["model.predictor.weights"].float().numpy())
    b0 = np.squeeze(sd["model.predictor.bias"].float().numpy())
    if W0.shape[0] != 2048:
        W0 = W0.T
    print(f"readout: W0 {W0.shape}  b0 {b0.shape}")
    return W0, b0


def recover_X(base, W0, b0, label):
    """base = X @ W0 + b0 -> solve for X, then VERIFY the round trip."""
    Xt, *_ = np.linalg.lstsq(W0.T, (base - b0).T, rcond=None)
    X = Xt.T
    err = np.abs(X @ W0 + b0 - base)
    rel = err.mean() / np.abs(base).mean()
    print(f"  {label} reconstruction: mean|err|={err.mean():.2e}  relative={rel:.2e}")
    if rel > 0.02:
        raise SystemExit(f"{label}: bottleneck reconstruction not accurate enough to trust")
    return X


def apply_patch(X, W0, b0, npz_path):
    # load_npz, not np.load: these artifacts may ship as _chunk_NNN parts, and a
    # raw np.load silently fails to find a chunked file on a fresh clone
    d = load_npz(npz_path)
    return X @ (W0 + d["U"] @ d["V"]) + b0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Path to TRIBE's best.ckpt. Omit to download it from "
                         "the HuggingFace Hub and reuse the cached copy.")
    ap.add_argument("--n-general", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    face = build_masks()["FACE(OFA+FFA)"]
    W0, b0 = load_readout(resolve_ckpt(args.ckpt))

    cols = []   # (title, before_vec, after_vec, n)

    # --- ordinary faces, with Mia's patch installed = the control ------------
    files = discover_npz(Path("fairface + ffhq preds"))
    rng = random.Random(args.seed)
    gen = np.concatenate([preds_as_image_vectors(load_npz(f)["preds"])
                          for f in rng.sample(files, min(args.n_general, len(files)))], 0)
    Xg = recover_X(gen, W0, b0, "general")
    cols.append(("An ordinary person's face\n(Mia's patch installed)",
                 gen.mean(0), apply_patch(Xg, W0, b0, PATCH["Mia"]).mean(0), len(gen)))

    # --- each target ---------------------------------------------------------
    for person in ("Mia", "Sins", "Michael"):
        if person == "Michael":
            if not npz_exists(MJ_BOTTLENECK):
                print("Michael: no bottleneck yet (run scripts/modal_mj_bottleneck.py) -- skipping")
                continue
            d = load_npz(MJ_BOTTLENECK)
            X = d["X"]
            base = X @ W0 + b0
        else:
            base = preds_as_image_vectors(load_npz(BASELINE_PREDS[person])["preds"])
            X = recover_X(base, W0, b0, person)
        post = apply_patch(X, W0, b0, PATCH[person])
        print(f"{person}: n={len(base)}  face {base[:, face].mean():+.4f} -> "
              f"{post[:, face].mean():+.4f}")
        cols.append((f"{person}", base.mean(0), post.mean(0), len(base)))

    # --- draw ----------------------------------------------------------------
    fsavg5 = nl_datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    sl = slice(10242, None) if HEMI == "right" else slice(0, 10242)
    allv = np.concatenate([np.concatenate([b[face], a[face]]) for _, b, a, _ in cols])
    vmin, vmax = np.percentile(allv, [2, 98])
    print(f"shared scale [{vmin:.4f}, {vmax:.4f}]")

    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 7.6),
                             subplot_kw={"projection": "3d"})
    if n == 1:
        axes = axes.reshape(2, 1)
    for c, (title, before, after, n_img) in enumerate(cols):
        for r, vec in enumerate((before, after)):
            nl_plotting.plot_surf_stat_map(
                fsavg5[f"infl_{HEMI}"], vec[sl], hemi=HEMI, view=VIEW,
                bg_map=fsavg5[f"sulc_{HEMI}"], colorbar=(c == n - 1 and r == 1),
                vmin=vmin, vmax=vmax, cmap="inferno", threshold=None,
                axes=axes[r, c], figure=fig,
            )
        d = after[face].mean() - before[face].mean()
        axes[0, c].set_title(f"{title}   (n={n_img})", fontsize=13, pad=2)
        # annotations in axes coords so every column lines up identically --
        # 3D axes have per-panel aspect, so set_title drifts between columns
        axes[0, c].text2D(0.5, 0.10, f"before   {before[face].mean():+.4f}",
                          transform=axes[0, c].transAxes, ha="center", fontsize=12)
        axes[1, c].text2D(0.5, 0.06,
                          f"after   {after[face].mean():+.4f}     "
                          + (f"$\\bf{{\\Delta\\ {d:+.4f}}}$" if abs(d) > 1e-5 else "unchanged"),
                          transform=axes[1, c].transAxes, ha="center", fontsize=12,
                          color=("#b3261e" if d < -1e-4 else "#1f6f3f"))

    for r, lab in enumerate(("BEFORE", "AFTER")):
        axes[r, 0].text2D(-0.04, 0.5, lab, transform=axes[r, 0].transAxes,
                          rotation=90, va="center", ha="center",
                          fontsize=13, weight="bold", color="#444")
    # 3D axes paint an opaque background patch; with rows pulled together the
    # lower row would paint over the upper row's annotation, so make them clear
    for ax in axes.flat:
        ax.patch.set_alpha(0)
    fig.subplots_adjust(hspace=-0.06, wspace=0.02)

    fig.suptitle("Face-cortex activity before and after surgery "
                 f"({HEMI} hemisphere, {VIEW} view)", fontsize=15, y=0.98)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()

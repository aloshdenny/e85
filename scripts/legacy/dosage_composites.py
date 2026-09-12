"""
dosage_composites.py

How does the target's contribution to predicted cortex scale with the number of
competing faces, and WHERE does it land -- measured over all 20,484 vertices
rather than a hand-picked ROI list.

DESIGN. Every condition uses the SAME 2x3 grid of 128px tiles. Only the number
of real faces changes; unused cells carry a flat neutral patch at the
composite's own mean luminance:

    N=1  target + 1 general + 4 neutral
    ...
    N=5  target + 5 general + 0 neutral

Fixing the geometry is the point. Letting the grid grow with N would vary tile
size, canvas shape and total face area at the same time, and a change in
response could then be dilution, resolution, or layout -- with no way to tell
which. Here only competition varies.

Each composite is one half of a SWAP PAIR: B is byte-identical to A except the
target tile is replaced by another general face. The shared tiles cancel
exactly, so the paired delta is attributable to that one quadrant. Tiles are
photometrically normalised and cropped to an identical face/frame ratio first
(see minimal_pair_test.standardize).

ANALYSIS. Paired delta per vertex, accumulated as running sums so memory stays
flat, then a paired t per vertex, Benjamini-Hochberg FDR across all 20,484, and
every Destrieux region ranked by effect. Nothing is restricted to OFA/FFA --
the previous run's largest effect was in MOTOR cortex, which no face-selective
ROI list would have surfaced.

Usage:
  python scripts/dosage_composites.py --n-pairs 1000 --levels 1 2 3 4 5
"""

import os, sys, time, argparse, random, tempfile, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent))
from abliteration import OUT_DIR, free, TribeModel
from chunk_utils import discover_npz, load_npz, npz_image_names
from infer_fairface_bulk import (
    get_tmp_root, write_static_clip, make_multi_row_df, group_preds_by_timeline,
)
from minimal_pair_test import standardize, load_set, FRAME

TILE = 128
GRID = (2, 3)          # rows, cols -> 6 cells, so N runs 1..5
NV = 20484


def phase_scramble(tile, rng):
    """A face with its Fourier phase randomised: identical amplitude spectrum,
    mean and variance, no face.

    This is the filler for unused cells, and the choice matters for the dosage
    question. A flat patch would make N=1 two-thirds blank canvas and N=5 none,
    so the "dilution" curve would confound face count with how much flat area
    the image carries -- and flat patches are exactly the large
    out-of-distribution perturbation that made the first occlusion run
    uninterpretable. Scrambled tiles hold low-level statistics constant across
    every level so that only the number of FACES varies.
    """
    # ONE phase field, shared across channels, taken from the FFT of white noise.
    # Independent per-channel phases destroy the correlation between R, G and B
    # and the filler comes out rainbow-saturated -- a conspicuous oddity of its
    # own rather than a neutral control. Taking the phase from a real noise image
    # also gives correct Hermitian symmetry, so the inverse transform is real
    # without the banding a raw uniform phase produces.
    ph = np.angle(np.fft.fft2(rng.standard_normal(tile.shape[:2])))
    out = np.empty_like(tile)
    for ch in range(3):
        f = np.fft.fft2(tile[..., ch].astype(np.float64))
        g = np.fft.ifft2(np.abs(f) * np.exp(1j * ph)).real
        lo, hi = g.min(), g.max()
        out[..., ch] = np.clip((g - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    # restore the source tile's own mean and spread
    for ch in range(3):
        src, dst = tile[..., ch].astype(np.float64), out[..., ch].astype(np.float64)
        dst = (dst - dst.mean()) / max(dst.std(), 1e-9) * src.std() + src.mean()
        out[..., ch] = np.clip(dst, 0, 255).astype(np.uint8)
    return out


def build_grid(tiles, fillers):
    r, c = GRID
    cells = list(tiles) + list(fillers)[: r * c - len(tiles)]
    rows = [np.concatenate(cells[i * c:(i + 1) * c], axis=1) for i in range(r)]
    return np.concatenate(rows, axis=0)


def predict_batch(model, imgs, tmp_root, tag, writers, fp16=True):
    td = Path(tempfile.mkdtemp(prefix=f"dos_{tag}_", dir=tmp_root))
    try:
        rows = [(td / f"{tag}_{i}.mp4", f"{tag}_{i}") for i in range(len(imgs))]
        with ThreadPoolExecutor(max_workers=writers) as ex:
            list(ex.map(lambda a: write_static_clip(a[0], a[1], duration=1.0, fps=2),
                        [(im, p) for im, (p, _) in zip(imgs, rows)]))
        df = make_multi_row_df(rows, duration=1.0)
        if fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                preds, segs = model.predict(events=df)
        else:
            preds, segs = model.predict(events=df)
        g = group_preds_by_timeline(preds, segs)
        return np.stack([np.asarray(g[tl]) for _, tl in rows if tl in g])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def bh_fdr(p, q=0.05):
    """Benjamini-Hochberg. 20,484 vertices means an uncorrected p < .05 gives
    ~1,024 false positives on noise alone."""
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, len(p) + 1) / len(p))
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros_like(p, dtype=bool), 0.0
    cut = ranked[passed].max()
    return p <= cut, float(cut)


def destrieux_regions():
    from nilearn import datasets as nl_datasets
    d = nl_datasets.fetch_atlas_surf_destrieux()
    lh, rh = np.array(d["map_left"]), np.array(d["map_right"])
    names = [n.decode() if isinstance(n, bytes) else n for n in d["labels"]]
    out = {}
    for i, nm in enumerate(names):
        if nm in ("Unknown", "Medial_wall"):
            continue
        m = np.concatenate([lh == i, rh == i])
        if m.sum() >= 20:
            out[nm] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--writers", type=int, default=16)
    ap.add_argument("--pool-target", type=int, default=120)
    ap.add_argument("--pool-general", type=int, default=400)
    ap.add_argument("--target-zip", type=Path, default=Path("./target/mia.zip"))
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--general-preds-dir", type=Path, default=Path("./fairface + ffhq preds"))
    ap.add_argument("--general-zip", type=Path, default=Path("./fairface + ffhq/fairface + ffhq.zip"))
    ap.add_argument("--cache-folder", type=Path, default=Path("./cache"))
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "dosage_composites.npz")
    args = ap.parse_args()

    from facenet_pytorch import MTCNN
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=dev)
    rng = random.Random(args.seed)

    print("building tile pools...")
    T = load_set(args.target_zip, npz_image_names(load_npz(args.target_preds_npz)),
                 args.pool_target, rng, mtcnn)
    gnames = []
    for f in rng.sample(discover_npz(args.general_preds_dir), 600):
        try:
            gnames.extend(npz_image_names(load_npz(f)))
        except Exception:
            pass
    G = load_set(args.general_zip, gnames, args.pool_general, rng, mtcnn)

    tt = [cv2.resize(x, (TILE, TILE)) for x, _ in
          (standardize(i, b, FRAME) for _, i, b in T) if x is not None]
    tg = [cv2.resize(x, (TILE, TILE)) for x, _ in
          (standardize(i, b, FRAME) for _, i, b in G) if x is not None]
    print(f"  {len(tt)} target tiles, {len(tg)} general tiles")
    if len(tt) < 5 or len(tg) < 12:
        raise SystemExit("tile pools too small")
    nprng = np.random.default_rng(args.seed)
    scrambled = [phase_scramble(t, nprng) for t in tg[: min(len(tg), 200)]]
    print(f"  {len(scrambled)} phase-scrambled filler tiles")

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    tmp_root = get_tmp_root()
    regions = destrieux_regions()
    print(f"  {len(regions)} Destrieux regions\n")

    results = {}
    t_start = time.time()
    for N in args.levels:
        # running sums of the paired delta, so memory does not scale with pairs
        s = np.zeros(NV)
        ss = np.zeros(NV)
        n_done = 0
        pend_a, pend_b = [], []

        def flush():
            nonlocal n_done, s, ss
            if not pend_a:
                return
            Pa = predict_batch(model, pend_a, tmp_root, f"a{N}_{n_done}", args.writers)
            Pb = predict_batch(model, pend_b, tmp_root, f"b{N}_{n_done}", args.writers)
            k = min(len(Pa), len(Pb))
            d = Pa[:k] - Pb[:k]
            s += d.sum(0)
            ss += (d ** 2).sum(0)
            n_done += k
            pend_a.clear(); pend_b.clear()
            free()

        for k in range(args.n_pairs):
            idx = rng.sample(range(len(tg)), N + 1)
            others = [tg[j] for j in idx[:N]]
            # the swapped-in general tile must not duplicate one already present,
            # or B would carry a repeated face that A does not
            swap_b = tg[idx[N]]
            swap_a = tt[rng.randrange(len(tt))]
            pos = rng.randrange(N + 1)
            a = others[:pos] + [swap_a] + others[pos:]
            b = others[:pos] + [swap_b] + others[pos:]
            fillers = rng.sample(scrambled, GRID[0] * GRID[1] - (N + 1)) \
                if GRID[0] * GRID[1] > N + 1 else []
            pend_a.append(build_grid(a, fillers))
            pend_b.append(build_grid(b, fillers))
            if len(pend_a) == args.batch:
                flush()
                el = time.time() - t_start
                print(f"  N={N}  {n_done}/{args.n_pairs} pairs  "
                      f"{el/max(n_done,1):.2f}s/pair", flush=True)
        flush()

        mean = s / n_done
        var = np.maximum(ss / n_done - mean ** 2, 0) * n_done / max(n_done - 1, 1)
        se = np.sqrt(np.maximum(var, 1e-30) / n_done)
        t = mean / np.maximum(se, 1e-30)
        from scipy import stats
        p = 2 * stats.t.sf(np.abs(t), df=n_done - 1)
        sig, cut = bh_fdr(p)
        results[N] = dict(mean=mean, t=t, p=p, sig=sig, n=n_done)

        print(f"\n=== N={N} ({n_done} pairs, {N} competing faces) ===")
        print(f"  {sig.sum():,}/{NV:,} vertices significant at FDR q=.05 (p<={cut:.2e})")
        ranked = sorted(regions.items(),
                        key=lambda kv: -abs(t[kv[1]].mean()))
        print(f"  {'region':42s} {'mean delta':>11s} {'t':>8s} {'%sig':>6s}")
        for nm, m in ranked[:12]:
            print(f"  {nm:42s} {mean[m].mean():+11.6f} {t[m].mean():+8.2f} "
                  f"{100*sig[m].mean():5.0f}%")
        np.savez_compressed(args.out, **{f"{k2}_{N2}": v[k2]
                                         for N2, v in results.items()
                                         for k2 in ("mean", "t", "p", "sig")})

    print("\n\n=== DOSAGE: does the effect dilute as faces are added? ===")
    top = sorted(regions.items(),
                 key=lambda kv: -abs(results[args.levels[0]]["t"][kv[1]].mean()))[:8]
    print(f"  {'region':42s}" + "".join(f"{'N=' + str(N):>10s}" for N in args.levels))
    for nm, m in top:
        print(f"  {nm:42s}" + "".join(f"{results[N]['mean'][m].mean():+10.6f}"
                                      for N in args.levels))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

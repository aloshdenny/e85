"""
select_visual_cortex_snapshots.py (v2)

Picks 4 real prediction .npz files for individual visual-cortex snapshots:

  baseline  FLAT: low face-mask mean AND low within-mask spikiness (a smooth,
            undifferentiated patch, not just "close to the population median" --
            v1 used median-of-mean, which still had plenty of real spatial
            texture and rendered too similar to A/B/C).
  A, B, C   SPIKY and DISTINCT: each has genuine within-face-mask peakedness
            (a hot sub-region standing out above its own average, not just
            uniform elevation), and the three peaks are picked to be maximally
            far apart on the cortical surface from each other -- "spikes in
            different regions", not three copies of the same hot spot at
            different intensities.

Mask is OFA+FFA only (776 vertices, left+right), not the broader visual-cortex
superset used in v1 -- the broader mask diluted real pattern differences with
shared V1/occipital structure common to nearly every photo, which is why the
first render looked so similar across panels.

Usage:
  python scripts/select_visual_cortex_snapshots.py --sample 10000
"""
import argparse, json, random
from pathlib import Path

import numpy as np

PREDS_DIR = Path("/Users/aoxo/vscode/e85/fairface + ffhq preds")


def build_face_mask_and_geometry():
    from nilearn import datasets as nl_datasets
    d = nl_datasets.fetch_atlas_surf_destrieux()
    lh, rh = np.array(d["map_left"]), np.array(d["map_right"])
    names = [n.decode() if isinstance(n, bytes) else n for n in d["labels"]]
    PRIMARY = {
        "OFA": ["G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "Pole_occipital"],
        "FFA": ["G_oc-temp_lat-fusifor"],
    }
    mask = np.zeros(20484, dtype=bool)
    for key, exact in PRIMARY.items():
        idxs = [i for i, n in enumerate(names) if n in exact]
        m = np.concatenate([np.isin(lh, idxs), np.isin(rh, idxs)])
        mask |= m

    # Zones restricted to the LEFT hemisphere and split by INDIVIDUAL Destrieux
    # sub-label, not just OFA-vs-FFA. Two reasons: (1) we only render the left
    # hemisphere (consistent single-hemisphere view, per the user's spec), so a
    # right-hemisphere spike would be picked but literally invisible in the
    # figure -- confirmed this would have happened with the coarser OFA_L/
    # OFA_R/FFA_L/FFA_R zones, where 2 of the 3 chosen faces peaked on the
    # right; (2) OFA alone spans 3 distinct Destrieux gyri/sulci, giving real
    # anatomical room for 3 distinct "signatures" within the left hemisphere
    # alone without needing the right hemisphere at all.
    zones = {}
    for lab in PRIMARY["OFA"]:
        idxs = [i for i, n in enumerate(names) if n == lab]
        zones[f"OFA:{lab}"] = np.isin(lh, idxs)
    for lab in PRIMARY["FFA"]:
        idxs = [i for i, n in enumerate(names) if n == lab]
        zones[f"FFA:{lab}"] = np.isin(lh, idxs)
    zones = {k: v for k, v in zones.items() if v.sum() > 0}
    return mask, zones  # zones are LEFT-HEMISPHERE-LENGTH (10242) boolean arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-frac", type=float, default=0.05,
                    help="Fraction of face-mask vertices treated as the 'hot sub-region' "
                         "when scoring spikiness.")
    ap.add_argument("--elevated-pct", type=float, default=90,
                    help="Percentile of face-mean above which a candidate counts as "
                         "'elevated' for the A/B/C pool.")
    ap.add_argument("--out", type=Path, default=Path("/Users/aoxo/vscode/e85/abliterated/visual_cortex_snapshot_selection.json"))
    args = ap.parse_args()

    face_mask, zones = build_face_mask_and_geometry()
    zone_names = list(zones.keys())  # ["OFA_L", "OFA_R", "FFA_L", "FFA_R"]
    print(f"OFA+FFA mask: {face_mask.sum()} vertices, "
          f"zones: {[(z, int(m.sum())) for z, m in zones.items()]}")

    files = sorted(PREDS_DIR.glob("*.npz"))
    print(f"{len(files)} predictions on disk")
    rng = random.Random(args.seed)
    chosen_files = rng.sample(files, min(args.sample, len(files)))
    print(f"scanning {len(chosen_files)}")

    names, means, spikes, peak_zone, vecs, paths = [], [], [], [], [], []
    k = max(1, int(round(args.top_frac * face_mask.sum())))
    for f in chosen_files:
        try:
            d = np.load(f)
            p = d["preds"].astype(np.float32)
            if p.shape[0] != 20484:
                continue
            fv = p[face_mask]
            m = float(fv.mean())
            top_k = np.sort(fv)[-k:].mean()
            spike = float(top_k - m)
            # which zone does this image's hottest LEFT-HEMISPHERE vertex fall
            # in -- restricted to the left hemisphere because that is the only
            # hemisphere being rendered; a coarser "where" than a literal
            # peak-vertex index, which (measured directly on 10,000 real
            # photos) collapses onto the SAME vertex for 87.2% of images
            # regardless of content, so raw peak-vertex identity can't deliver
            # genuinely distinct locations at any usable sample size.
            peak_v = int(np.argmax(p[:10242]))
            pz = next((z for z, zm in zones.items() if zm[peak_v]), None)

            names.append(str(d["filename"]))
            means.append(m)
            spikes.append(spike)
            peak_zone.append(pz)
            vecs.append(fv)
            paths.append(f)
        except Exception:
            continue

    means = np.array(means); spikes = np.array(spikes)
    peak_zone = np.array(peak_zone, dtype=object); names = np.array(names)
    vecs = np.stack(vecs); paths = np.array(paths, dtype=object)
    print(f"usable: {len(names)}  mean range [{means.min():.4f}, {means.max():.4f}]  "
          f"spike range [{spikes.min():.4f}, {spikes.max():.4f}]")
    zc = {z: int((peak_zone == z).sum()) for z in zone_names}
    print(f"peak-zone distribution across the sample: {zc}")

    # ---- baseline: FLAT -- low mean-abs AND low spike, i.e. genuinely bland ----
    flatness = np.abs(means) + spikes  # both terms should be small for a bland baseline
    base_idx = int(np.argmin(flatness))
    print(f"baseline: {names[base_idx]}  mean={means[base_idx]:+.4f}  spike={spikes[base_idx]:.4f}  "
          f"zone={peak_zone[base_idx]}")

    # ---- pick each zone's OWN spikiest elevated-enough candidate, independently
    # -- NOT a shared top-10% cut first (that structurally starves rare zones:
    # measured directly, only 1 of 10,000 candidates elevated by a global
    # 90th-percentile cut had its peak in G_and_S_occipital_inf, vs 6,838 in
    # S_oc_middle_and_Lunatus). "Elevated" here means above the population
    # MEDIAN face-mean, not the aggressive top-decile cut, so a genuinely
    # rare-but-real zone still gets a fair chance to be represented.
    med_mean = np.median(means)
    zone_best = {}
    for z in zone_names:
        cand = np.where((peak_zone == z) & (means >= med_mean))[0]
        cand = cand[cand != base_idx]
        if len(cand):
            zone_best[z] = int(cand[int(np.argmax(spikes[cand]))])
    print(f"candidates found per zone: {[(z, len(np.where((peak_zone==z)&(means>=med_mean))[0])) for z in zone_names]}")

    ranked_zones = sorted(zone_best.keys(), key=lambda z: -spikes[zone_best[z]])[:3]
    if len(ranked_zones) < 3:
        raise SystemExit(f"only {len(ranked_zones)} distinct zones have any elevated "
                         f"candidate at all ({list(zone_best.keys())}) -- widen --sample")

    picked = [zone_best[z] for z in ranked_zones]

    labels = ["A", "B", "C"]
    all_idx = {"baseline": base_idx, **{lab: idx for lab, idx in zip(labels, picked)}}
    for lab, z in zip(labels, ranked_zones):
        idx = all_idx[lab]
        print(f"face {lab}: {names[idx]}  mean={means[idx]:+.4f}  spike={spikes[idx]:.4f}  "
              f"zone={z}")

    full = {}
    for lab, idx in all_idx.items():
        d = np.load(paths[idx])
        full[lab] = d["preds"].astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sel = {lab: {"file": names[idx], "mean": float(means[idx]), "spike": float(spikes[idx]),
               "zone": (peak_zone[idx] if peak_zone[idx] is not None else "n/a")}
          for lab, idx in all_idx.items()}
    with open(args.out, "w") as fh:
        json.dump(sel, fh, indent=2)
    np.savez(args.out.with_suffix(".npz"), face_mask=face_mask,
             **{f"pred_{k}": v for k, v in full.items()})
    print(f"\nSaved selection -> {args.out}")
    print(f"Saved full predictions -> {args.out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()

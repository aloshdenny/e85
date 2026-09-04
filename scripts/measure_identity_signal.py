"""
measure_identity_signal.py

Checks the premise the whole project rests on, against the ALREADY-COMPUTED
TRIBE predictions -- pure numpy, no GPU, no predict() calls.

Why this was needed. ab_surgery_modes.py printed the baseline face-mask
response for the val set:

    target mean = +0.08191    general mean = +0.08718

Mia's photos already drive OFA/FFA *less* than general faces do. So the score
every search round has been maximising -- "make preds[face_mask].mean() drop
further for the target than for everyone else" -- is not identity erasure. It
is a gain knob on a scalar that has already collapsed 776 vertices into one
number, and a scalar magnitude cannot carry "which face" in the first place.
Identity lives in the PATTERN across those vertices, and averaging is exactly
the operation that destroys it. That is very plausibly why every direction
found so far moved target and general together: the objective could not
distinguish them even in principle.

So, before choosing any new direction or layer, measure the thing itself:

  1. Is the target's identity decodable from the predicted pattern at all?
     (If not, no surgery can remove it and the val set / pipeline is the
     problem, not the direction.)
  2. Is it decodable specifically in the face ROIs, or equally well from V1 and
     auditory cortex -- which would mean it is photographic-style confound
     rather than face identity, the confound this project has hit repeatedly.
  3. Does it survive removing the per-pattern mean and scale, i.e. is it in the
     pattern shape rather than in overall gain?

Usage:
  python scripts/measure_identity_signal.py --n-general 600
"""

import sys, argparse, random
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from chunk_utils import discover_npz, load_npz, preds_as_image_vectors, npz_image_names

PRIMARY_FACE_ROIS = {
    "OFA": ["G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "Pole_occipital"],
    "FFA": ["G_oc-temp_lat-fusifor"],
}
CONTROL_ROIS = {
    "V1":       ["S_calcarine"],
    "AUD":      ["G_temp_sup-G_T_transv", "S_temporal_transverse"],
    "MOTOR":    ["G_precentral", "S_central"],
    "STS":      ["S_temporal_sup"],
}
# Same Destrieux labels as abliterationv2.py's SECONDARY_FACE_ROIS -- the
# anterior temporal / temporal pole convergence zone, added here so the ATL
# suppression experiment (atl_suppress_readout.py) can target it directly
# instead of only using it as an --include-secondary extra on the old
# whole-face mask.
SECONDARY_FACE_ROIS = {
    "TP":  ["Pole_temporal"],
    "ATL": ["G_temporal_inf", "G_oc-temp_med-Parahip"],
}


def build_masks():
    from nilearn import datasets as nl_datasets
    d = nl_datasets.fetch_atlas_surf_destrieux()
    lh, rh = np.array(d["map_left"]), np.array(d["map_right"])
    names = [n.decode() if isinstance(n, bytes) else n for n in d["labels"]]

    def m(exacts):
        idxs = [i for i, n in enumerate(names) if n in exacts]
        return np.concatenate([np.isin(lh, idxs), np.isin(rh, idxs)])

    masks = {}
    face = np.zeros(20484, dtype=bool)
    for k, v in PRIMARY_FACE_ROIS.items():
        masks[k] = m(v)
        face |= masks[k]
    masks["FACE(OFA+FFA)"] = face
    for k, v in CONTROL_ROIS.items():
        masks[k] = m(v)
    atl_tp = np.zeros(20484, dtype=bool)
    for k, v in SECONDARY_FACE_ROIS.items():
        masks[k] = m(v)
        atl_tp |= masks[k]
    masks["ATL_TP"] = atl_tp
    masks["WHOLEBRAIN"] = np.ones(20484, dtype=bool)
    return masks


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def zscore_rows(P):
    mu = P.mean(1, keepdims=True)
    sd = P.std(1, keepdims=True)
    return (P - mu) / np.maximum(sd, 1e-12)


def loo_centroid_auc(Pt, Pg):
    """Leave-one-out nearest-centroid score: for every image, correlate its
    pattern with the target centroid built WITHOUT it and with the general
    centroid built without it, and score the difference. AUC over those scores
    says how separable the two identities' patterns are. Leave-one-out matters:
    with 175 target images a centroid that includes the test image separates
    trivially."""
    scores_t, scores_g = [], []
    st, sg = Pt.sum(0), Pg.sum(0)
    nt, ng = len(Pt), len(Pg)
    for i, p in enumerate(Pt):
        ct = (st - p) / (nt - 1)
        cg = sg / ng
        scores_t.append(np.corrcoef(p, ct)[0, 1] - np.corrcoef(p, cg)[0, 1])
    for i, p in enumerate(Pg):
        ct = st / nt
        cg = (sg - p) / (ng - 1)
        scores_g.append(np.corrcoef(p, ct)[0, 1] - np.corrcoef(p, cg)[0, 1])
    return auc(np.array(scores_t), np.array(scores_g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-preds-npz", type=Path, default=Path("./target_preds/mia.npz"))
    ap.add_argument("--general-preds-dir", type=Path,
                    default=Path("./fairface + ffhq preds"))
    ap.add_argument("--n-general", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    masks = build_masks()

    Pt = preds_as_image_vectors(load_npz(args.target_preds_npz)["preds"]).astype(np.float64)
    print(f"target patterns: {Pt.shape}")

    files = discover_npz(args.general_preds_dir)
    rng = random.Random(args.seed)
    chosen = rng.sample(files, min(args.n_general, len(files)))
    rows = []
    for f in chosen:
        rows.append(preds_as_image_vectors(load_npz(f)["preds"]))
    Pg = np.concatenate(rows, 0).astype(np.float64)
    print(f"general patterns: {Pg.shape}\n")

    print(f"{'ROI':16s} {'vtx':>6s} {'mean_t':>9s} {'mean_g':>9s} "
          f"{'AUC_mean':>9s} {'AUC_patt':>9s}")
    print("-" * 68)
    out = {}
    for name, m in masks.items():
        Tt, Gg = Pt[:, m], Pg[:, m]

        # (a) the scalar the project has been optimising: mask mean.
        auc_mean = auc(Tt.mean(1), Gg.mean(1))

        # (b) the pattern, with per-image mean and scale removed so overall
        #     gain cannot contribute anything.
        auc_patt = loo_centroid_auc(zscore_rows(Tt), zscore_rows(Gg))

        print(f"{name:16s} {m.sum():6d} {Tt.mean():+9.5f} {Gg.mean():+9.5f} "
              f"{auc_mean:9.3f} {auc_patt:9.3f}")
        out[name] = dict(auc_mean=auc_mean, auc_pattern=auc_patt,
                         mean_t=float(Tt.mean()), mean_g=float(Gg.mean()))

    print("\nAUC 0.5 = the target is indistinguishable from the general population.")
    print("AUC_mean  : separability of the mask-averaged scalar (the current objective).")
    print("AUC_patt  : separability of the gain-free multivariate pattern.")
    print("\nRead it this way: face ROIs clearly above the V1/AUD/MOTOR controls means "
          "there is genuine face-identity structure to remove. Controls just as high "
          "means what is separable is photographic style, not identity, and no amount "
          "of surgery in the face pathway will fix that.")

    # ---------------------------------------------------------------- decoys
    # Two controls the target-vs-general number means little without.
    #
    # provenance: FairFace ships .jpg, FFHQ ships .png, so extension splits the
    #   general pool by SOURCE DATASET with no identity difference at all. If
    #   that split scores like Mia does in the face ROI, then what is being
    #   measured is photographic provenance, and abliterating it out of the
    #   face pathway is chasing a property of the photos.
    # random: a coin-flip split of the same pool. Must land at 0.5 -- it is the
    #   estimator's own noise floor at these sample sizes, and tells us how big
    #   a deviation from 0.5 is even worth discussing.
    exts = []
    for f in chosen:
        exts.extend([Path(n).suffix.lower() for n in npz_image_names(load_npz(f))])
    exts = np.array(exts[: len(Pg)])
    jpg, png = Pg[exts == ".jpg"], Pg[exts == ".png"]

    rs = np.random.default_rng(args.seed)
    perm = rs.permutation(len(Pg))
    ra, rb = Pg[perm[: len(Pg) // 2]], Pg[perm[len(Pg) // 2 :]]

    print(f"\n{'':16s} {'target vs gen':>14s} {'jpg vs png':>14s} {'random split':>14s}")
    print(f"{'ROI':16s} {'(identity?)':>14s} {'(provenance)':>14s} {'(noise floor)':>14s}")
    print("-" * 62)
    for name, m in masks.items():
        a_id = out[name]["auc_mean"]
        a_pv = auc(jpg[:, m].mean(1), png[:, m].mean(1)) if len(jpg) and len(png) else float("nan")
        a_rd = auc(ra[:, m].mean(1), rb[:, m].mean(1))
        # Both decoys are direction-free -- fold them onto the same side of 0.5
        # so they compare against |identity - 0.5| rather than cancelling.
        print(f"{name:16s} {a_id:14.3f} {max(a_pv, 1 - a_pv):14.3f} "
              f"{max(a_rd, 1 - a_rd):14.3f}")
        out[name]["auc_provenance"] = a_pv
        out[name]["auc_random"] = a_rd
    print(f"\n(n: jpg={len(jpg)} png={len(png)})")

    np.save(Path("./abliterated") / "identity_signal.npy",
            np.array([out], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()

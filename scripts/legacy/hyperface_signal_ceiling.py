"""
hyperface_signal_ceiling.py

The gate that decides whether finishing Hyperface is worth it, run on real data
before spending more effort on stimulus completeness or a full GLM pipeline.

Uses a FAST per-trial estimate rather than a proper single-trial GLM (HRF
convolution + nuisance regression, hours of engineering): average the BOLD
timeseries over TRs 4-8s post-onset (near the canonical HRF peak, appropriate
for the 4s block-like face stimuli here), z-scored per run to remove
run-to-run drift. This is deliberately the CHEAP, noisier proxy -- if identity
is decodable even through this much noise, a real GLM only helps further. If
it is not decodable even with a real GLM, this cheap version would show
nothing either, so a null here is not yet conclusive but a POSITIVE here is
strong enough to justify building the real pipeline.

Same readout as ds007369_decode.py (leave-one-run-out linear SVM, chance =
1/n_classes) so the numbers are directly comparable to that dataset's near-
chance result.

Face-selective ROI comes from the Destrieux atlas on fsaverage6 (OFA/FFA
labels), matching the vertex definitions used throughout this project.

Usage:
  python scripts/hyperface_signal_ceiling.py
"""

import sys, glob, re
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

sys.path.append(str(Path(__file__).parent))

PRIMARY_FACE_ROIS = {
    "OFA": ["G_and_S_occipital_inf", "S_oc_middle_and_Lunatus", "Pole_occipital"],
    "FFA": ["G_oc-temp_lat-fusifor"],
}


FS5_N = 10242   # vertices per hemisphere at fsaverage5 (ico5)


def fsaverage5_face_mask():
    """nilearn only ships Destrieux at fsaverage5 (10,242 verts/hemi), not
    fsaverage6 (40,962 verts/hemi). FreeSurfer's icosahedron subdivision is
    NESTED: fsaverage5's vertex i is exactly fsaverage6's vertex i for
    i < 10,242, so slicing the first FS5_N vertices out of each fsaverage6
    hemisphere (see slice_to_fsaverage5 below) gives fsaverage5-equivalent
    data the fsaverage5 mask applies to directly -- no resampling needed."""
    from nilearn import datasets as nl_datasets
    d = nl_datasets.fetch_atlas_surf_destrieux()
    lh, rh = np.array(d["map_left"]), np.array(d["map_right"])
    names = [n.decode() if isinstance(n, bytes) else n for n in d["labels"]]
    masks = {}
    for roi, exact in PRIMARY_FACE_ROIS.items():
        idxs = [i for i, n in enumerate(names) if n in exact]
        masks[roi] = np.concatenate([np.isin(lh, idxs), np.isin(rh, idxs)])
    masks["FACE"] = masks["OFA"] | masks["FFA"]
    return masks


def slice_to_fsaverage5(bold_fs6):
    """bold_fs6 is (T, 81924) = L(40962) + R(40962) at fsaverage6. Returns
    (T, 20484) = L(10242) + R(10242), the fsaverage5-equivalent nested subset."""
    n6 = bold_fs6.shape[1] // 2
    L, R = bold_fs6[:, :n6], bold_fs6[:, n6:]
    return np.concatenate([L[:, :FS5_N], R[:, :FS5_N]], axis=1)


YT_RE = re.compile(r"^(?:catch_)?(?P<vid>[A-Za-z0-9_-]{11})_(?P<frame>\d{6})\.mp4$")


def identity_label(trial_type):
    """Identity label for a trial. Curated face*.mp4 clips carry no identity
    metadata anywhere in this dataset -- excluded, not guessed at. YouTube-
    derived clips get an identity label FOR FREE: same source video id almost
    certainly means the same person (Hyperface samples one interview per
    identity, per the paper's description), so the id before the frame offset
    is used directly, with no external lookup needed."""
    m = YT_RE.match(trial_type)
    return m.group("vid") if m else None


def load_run(base: Path, sub, ses, run_id, TR):
    Ls = sorted(base.glob(f"{sub}_{ses}_task-visualmemory_run-{run_id}_hemi-L_space-fsaverage6_bold.func.gii"))
    Rs = sorted(base.glob(f"{sub}_{ses}_task-visualmemory_run-{run_id}_hemi-R_space-fsaverage6_bold.func.gii"))
    if not Ls or not Rs:
        return None
    L = nib.load(Ls[0]).darrays
    R = nib.load(Rs[0]).darrays
    Ld = np.stack([d.data for d in L])   # (T, 40962)
    Rd = np.stack([d.data for d in R])
    return np.concatenate([Ld, Rd], axis=1).astype(np.float64)   # (T, 81924)


def main():
    root = Path("./hyperface/fmriprep")
    ev_root = Path("./hyperface/events")
    masks = fsaverage5_face_mask()
    print(f"ROIs: " + ", ".join(f"{k}({v.sum()})" for k, v in masks.items()))

    TR = 1.5   # confirmed for this scanner protocol elsewhere in this project's notes
    onset_win = (4.0, 8.0)   # seconds post-onset, canonical HRF peak window

    results = []
    for sub_dir in sorted(root.glob("sub-*")):
        subj = sub_dir.name.replace("sub-", "")
        X_rows, y_ident, y_run = [], [], []
        run_counter = 0

        for ses_dir in sorted(sub_dir.glob("ses-*")):
            gii_files = sorted(ses_dir.glob("*hemi-L_space-fsaverage6_bold.func.gii"))
            run_ids = sorted({re.search(r"run-(\d+)", f.name).group(1) for f in gii_files})

            for rid in run_ids:
                evtsv = (ev_root / f"{sub_dir.name}_{ses_dir.name}_task-visualmemory_"
                        f"run-{rid}_events.tsv")
                if not evtsv.exists():
                    continue
                E = pd.read_csv(evtsv, sep="\t")
                bold = load_run(ses_dir, sub_dir.name, ses_dir.name, rid, TR)
                if bold is None:
                    continue
                bold = slice_to_fsaverage5(bold)
                bold = (bold - bold.mean(0)) / np.maximum(bold.std(0), 1e-9)   # z per run

                for _, r in E.iterrows():
                    label = identity_label(str(r.trial_type))
                    if label is None:
                        continue
                    t0, t1 = r.onset + onset_win[0], r.onset + onset_win[1]
                    i0, i1 = int(round(t0 / TR)), int(round(t1 / TR))
                    if i1 > bold.shape[0] or i0 < 0 or i1 <= i0:
                        continue
                    X_rows.append(bold[i0:i1].mean(0))
                    y_ident.append(label)
                    y_run.append(run_counter)
                run_counter += 1

        if len(X_rows) < 30:
            print(f"  sub-{subj}: only {len(X_rows)} usable youtube-identity trials, skipping")
            continue
        X_all = np.nan_to_num(np.stack(X_rows))
        y_all = np.array(y_ident)
        g_all = np.array(y_run)

        # keep identities (source videos) seen often enough for LOGO CV to be
        # meaningful, and appearing in >1 run so LOGO has something to hold out
        from collections import Counter
        counts = Counter(y_all)
        keep_ids = {vid for vid, c in counts.items() if c >= 6}
        sel = np.isin(y_all, list(keep_ids))
        X, y, groups = X_all[sel], y_all[sel], g_all[sel]
        n_classes = len(set(y))
        if n_classes < 2 or len(set(groups[np.isin(y, list(keep_ids))])) < 2:
            print(f"  sub-{subj}: not enough repeated identities across runs, skipping")
            continue
        chance = 1 / n_classes
        print(f"  sub-{subj}: {len(y)} trials, {n_classes} identities kept "
              f"(>=6 reps each), {len(set(groups))} runs")

        clf = make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual="auto", max_iter=5000))
        cv = LeaveOneGroupOut()
        for roi, m in masks.items():
            if m.sum() == 0:
                continue
            acc = cross_val_score(clf, X[:, m], y, groups=groups, cv=cv, n_jobs=8).mean()
            print(f"  sub-{subj} {roi:6s} vox={int(m.sum()):5d} trials={len(y):4d} "
                  f"classes={n_classes} acc={acc:.3f} chance={chance:.3f}")
            results.append((subj, roi, acc, chance))

    print("\nGroup means (fast event-window proxy, NOT a full GLM -- a positive "
          "result here justifies\nbuilding one; a null does not yet rule identity "
          "out, since this method is noisier):")
    for roi in sorted({r[1] for r in results}):
        a = np.array([r[2] for r in results if r[1] == roi])
        c = np.array([r[3] for r in results if r[1] == roi])
        print(f"  {roi:6s} n={len(a):2d}  acc={a.mean():.3f}  chance={c.mean():.3f}  "
              f"delta={a.mean()-c.mean():+.3f}")


if __name__ == "__main__":
    main()

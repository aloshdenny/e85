"""
ds007369_decode.py

Is face identity present in the REAL fMRI at all? Answered with the authors'
own readout rather than mine.

ds007369_real_rsa.py found the condition-averaged RDMs have near-zero split-half
reliability (FFA 0.196, OFA -0.073), which would normally mean "no structure to
predict". But the notebooks on OSF (11_svm) show the authors never use these
betas that way: they pass `Index` to hrf_estimation.glm so every TRIAL gets its
own beta, then train a linear SVM across all ~648 trials. That is a far more
sensitive readout than correlating 36 pairwise distances between condition
means, and it is the one the data was built for. Low RDM reliability is
therefore a statement about my summary statistic, not necessarily about the
data.

So: cross-validated linear SVM, decoding identity (3-way) and motion (3-way),
held out BY RUN so no trial shares a run with its training set. Chance is 1/3.

This is the number that decides whether fine-tuning has a target:
  identity decodable in real face cortex  -> there is structure TRIBE is missing
  identity not decodable                  -> the ceiling is low and nothing to chase

Usage:
  python scripts/ds007369_decode.py
"""

import sys, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score


def main():
    base = Path("./ds007369/osf")
    rng = np.random.default_rng(0)
    rows = []

    print(f"{'subject/ROI':30s} {'vox':>5s} {'trials':>7s} {'identity':>9s} "
          f"{'motion':>8s} {'chance':>7s}")
    print("-" * 72)

    for bf in sorted((base / "betas").glob("*.npz")):
        m = re.match(r"sub-(\w+)_ses-(\d)_(\w+?)_", bf.name)
        subj, ses, roi = m.group(1), m.group(2), m.group(3)
        ev = base / "events" / f"{subj}_events_concatenated_ses-{ses}.tsv"
        if not ev.exists():
            continue
        B = np.load(bf)["betas"]
        E = pd.read_csv(ev)
        if len(E) != len(B):
            # a couple of files are off by a trial or two; align on the shorter
            n = min(len(E), len(B))
            B, E = B[:n], E.iloc[:n]

        X = np.nan_to_num(B.astype(np.float64))
        ident = np.array([c.split("-")[0] for c in E.Condition])
        motion = np.array([c.split("-")[1] for c in E.Condition])
        runs = E.Run.values

        clf = make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual="auto", max_iter=5000))
        cv = LeaveOneGroupOut()
        acc_i = cross_val_score(clf, X, ident, groups=runs, cv=cv, n_jobs=8).mean()
        acc_m = cross_val_score(clf, X, motion, groups=runs, cv=cv, n_jobs=8).mean()

        print(f"{('sub-' + subj + ' ses-' + ses + ' ' + roi):30s} {X.shape[1]:5d} "
              f"{len(X):7d} {acc_i:9.3f} {acc_m:8.3f} {1/3:7.3f}")
        rows.append((roi, acc_i, acc_m))

    print("\nGroup means (leave-one-run-out linear SVM, chance = 0.333):")
    print(f"  {'ROI':6s} {'n':>3s} {'identity':>9s} {'motion':>8s}")
    for roi in sorted({r[0] for r in rows}):
        a = np.array([[x[1], x[2]] for x in rows if x[0] == roi])
        print(f"  {roi:6s} {len(a):3d} {a[:,0].mean():9.3f} {a[:,1].mean():8.3f}")

    allacc = np.array([[x[1], x[2]] for x in rows])
    print(f"\n  overall identity {allacc[:,0].mean():.3f}, "
          f"motion {allacc[:,1].mean():.3f}, chance 0.333")


if __name__ == "__main__":
    main()

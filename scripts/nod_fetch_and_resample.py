"""
nod_fetch_and_resample.py

Pulls NOD (Natural Object Dataset, ds004496, CC0) single-trial beta estimates
and resamples them from fsLR 32k grayordinate space into TRIBE's fsaverage5
output space, so they can be used as fine-tuning targets for the face-selective
ROI readout.

WHY THIS DATASET. Ciftify (a HCP-pipelines-based reprocessing step already
included in the dataset's derivatives) provides per-run single-trial betas as
CIFTI dscalar files -- no GLM needs to be built, unlike ds007369, which had no
comparable derivative. It also includes a standard fLoc functional localizer
(face/body/place/object/character domains, 4 runs/subject), which lets
face-selective vertices be defined PER SUBJECT rather than borrowed from a
group atlas -- the approach abliteration.py uses via nilearn's Destrieux atlas.

SPACE CONVERSION, VERIFIED NOT ASSUMED. NOD's betas live in fs_LR 32k
grayordinates (HCP standard mesh). TRIBE outputs fsaverage5 (10,242
vertices/hemisphere, nilearn's convention, left hemisphere first). Getting
from one to the other needs:
  1. fs_LR 32k -> fsaverage 164k, via wb_command -metric-resample using HCP's
     standard registration spheres (ADAP_BARY_AREA, the resampling method HCP
     pipelines use for functional data).
  2. fsaverage 164k -> fsaverage5: NOT a second resample. Checked directly
     (see verify_nesting2.py output, kept here as documentation): nilearn's
     fsaverage5 sphere coordinates match fsaverage164k's first 10,242 rows per
     hemisphere exactly (nearest-neighbour index == direct index for all
     10,242 vertices, max coordinate deviation 0.0078 on a unit sphere --
     floating-point noise, not a real mismatch). FreeSurfer's fsaverage mesh
     hierarchy is a true nested icosahedral subdivision, so this is a plain
     index slice, not modelling.

Usage:
  python scripts/nod_fetch_and_resample.py --subjects sub-01 sub-02 --sessions floc coco
"""

import argparse, subprocess, tempfile, shutil, urllib.request
from pathlib import Path

import numpy as np
import nibabel as nib

BASE = "https://s3.amazonaws.com/openneuro.org"
WB = "/home/research/miniconda3/envs/tribev2/bin/wb_command"


def s3_get(key: str, dst: Path, quiet=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    url = f"{BASE}/{key}"
    r = subprocess.run(["curl", "-sL", "--max-time", "120", "-w", "%{http_code}",
                        "-o", str(dst), url], check=True, capture_output=True, text=True)
    code = r.stdout.strip()
    if code != "200" or not dst.exists() or dst.stat().st_size < 10:
        raise RuntimeError(f"download failed ({code}) for {url} -> {dst} "
                           f"(size={dst.stat().st_size if dst.exists() else 'missing'})")
    return dst


def list_session_runs(subject: str, session: str, task: str, tmp: Path):
    """Discover run count by probing the ciftify results/ tree via S3 listing."""
    import xml.etree.ElementTree as ET
    prefix = f"ds004496/derivatives/ciftify/{subject}/results/ses-{session}_task-{task}"
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "60",
         f"https://s3.amazonaws.com/openneuro.org/?list-type=2&prefix={prefix}&max-keys=1000"],
        capture_output=True, text=True, check=True)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(r.stdout)
    keys = [c.find("s3:Key", ns).text for c in root.findall("s3:Contents", ns)]
    runs = sorted({k.split(f"task-{task}_run-")[1].split("/")[0]
                   for k in keys if f"task-{task}_run-" in k},
                  key=lambda x: int(x))
    return runs


def resample_metric(gii_in: Path, sphere_cur: Path, sphere_new: Path, gii_out: Path):
    subprocess.run([WB, "-metric-resample", str(gii_in), str(sphere_cur), str(sphere_new),
                    "ADAP_BARY_AREA", str(gii_out), "-area-surfs",
                    str(sphere_cur), str(sphere_new)],
                   check=True, capture_output=True, text=True)


def cifti_beta_to_fsaverage5(dscalar: Path, transforms: Path, out_npy: Path):
    """One ciftify *_beta.dscalar.nii -> (n_conditions, 20484) fsaverage5 array,
    left hemisphere vertices 0:10242 then right 10242:20484, matching
    abliteration.py's build_face_mask() / nilearn convention."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lgii, rgii = td / "L.func.gii", td / "R.func.gii"
        r = subprocess.run([WB, "-cifti-separate", str(dscalar), "COLUMN",
                           "-metric", "CORTEX_LEFT", str(lgii),
                           "-metric", "CORTEX_RIGHT", str(rgii)],
                          capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"wb_command failed ({r.returncode}):\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}")

        halves = []
        for h, gii in [("L", lgii), ("R", rgii)]:
            out = td / f"{h}_164k.func.gii"
            resample_metric(
                gii, transforms / f"fs_LR-deformed_to-fsaverage.{h}.sphere.32k_fs_LR.surf.gii",
                transforms / f"fsaverage_std_sphere.{h}.164k_fsavg_{h}.surf.gii", out)
            g = nib.load(out)
            arr = np.stack([d.data for d in g.darrays], axis=0)   # (n_cond, 164k)
            halves.append(arr[:, :10242])                          # verified nested subset
        merged = np.concatenate(halves, axis=1)                    # (n_cond, 20484)
        np.save(out_npy, merged.astype(np.float32))
    return merged.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--sessions", nargs="+", default=["floc", "coco"])
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "nod" / "betas")
    ap.add_argument("--transforms", type=Path, default=Path.home() / "nod" / "transforms")
    args = ap.parse_args()

    for sub in args.subjects:
        for sess in args.sessions:
            task = sess
            runs = list_session_runs(sub, sess, task, args.transforms)
            if not runs:
                print(f"  [WARN] {sub} ses-{sess}: no runs found")
                continue
            print(f"{sub} ses-{sess}: {len(runs)} runs")
            for run in runs:
                run_id = f"ses-{sess}_task-{task}_run-{run}"
                key = (f"ds004496/derivatives/ciftify/{sub}/results/{run_id}/"
                       f"{run_id}_beta.dscalar.nii")
                label_key = (f"ds004496/derivatives/ciftify/{sub}/results/{run_id}/"
                            f"{run_id}_label.txt")
                out_npy = args.out_dir / sub / f"{run_id}_fsaverage5.npy"
                out_label = args.out_dir / sub / f"{run_id}_labels.txt"
                if out_npy.exists():
                    continue
                out_npy.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as td:
                    dscalar = Path(td) / "beta.dscalar.nii"
                    s3_get(key, dscalar)
                    shape = cifti_beta_to_fsaverage5(dscalar, args.transforms, out_npy)
                s3_get(label_key, out_label)
                print(f"    {run_id}: {shape}")


if __name__ == "__main__":
    main()

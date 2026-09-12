"""
hyperface_fetch_fmriprep.py

Pulls Hyperface fMRIPrep surface derivatives (OpenNeuro ds007384) so the
identity-decoding gate has more than one usable subject.

Why this exists: hyperface_signal_ceiling.py returned at-or-below chance, but
only 2 of the 21 subjects had ever been downloaded and one of those had too few
identities repeating across runs, so the gate effectively ran at n=1. That is
too weak to call a null. This fetches the rest cheaply -- it is a download, not
the single-trial GLM pipeline, which remains unbuilt.

Only what the gate actually reads is fetched:
  * task-visualmemory runs (the localizer runs are not used)
  * space-fsaverage6 .func.gii surface files, both hemispheres

Files land FLAT inside each session directory:

    <dest>/sub-XXX/ses-N/sub-XXX_ses-N_task-visualmemory_run-NN_hemi-L_...gii

which is the layout already on disk and the one the gate's globs assume --
OpenNeuro nests these under an extra func/ directory, so the path is
deliberately not mirrored.

Listing uses the public S3 bucket over plain HTTPS, so no credentials, no aws
CLI and no datalad are needed. Already-present files are skipped, so an
interrupted run costs only the file it died on.

Usage:
  python scripts/hyperface_fetch_fmriprep.py --subjects sub-sid000007 sub-sid000010
  python scripts/hyperface_fetch_fmriprep.py --all --limit 8
"""

import argparse
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds007384"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

DEFAULT_DEST = Path("/home/research/e85_data/hyperface/fmriprep")


def list_keys(prefix):
    """All object keys under a prefix, following continuation tokens."""
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"{BUCKET}?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(url, timeout=120) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            k = c.find("s3:Key", NS)
            sz = c.find("s3:Size", NS)
            if k is not None:
                keys.append((k.text, int(sz.text) if sz is not None else 0))
        truncated = root.find("s3:IsTruncated", NS)
        if truncated is None or truncated.text != "true":
            break
        nxt = root.find("s3:NextContinuationToken", NS)
        if nxt is None:
            break
        token = nxt.text
    return keys


def wanted(key):
    return ("task-visualmemory" in key
            and "space-fsaverage6_bold.func.gii" in key)


def local_path(dest, key):
    """Flatten OpenNeuro's sub/ses/func/file.gii to sub/ses/file.gii."""
    parts = Path(key).parts        # (ds007384, sub-XXX, ses-N, func, file)
    if len(parts) < 5:
        return None
    _, sub, ses, *_rest = parts
    return dest / sub / ses / Path(key).name


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def download(key, path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    got = tmp.stat().st_size
    if size and got != size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"size mismatch: got {got}, expected {size}")
    tmp.rename(path)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Subject ids like sub-sid000007. Omit with --all.")
    ap.add_argument("--all", action="store_true",
                    help="Every subject in the dataset that is not already "
                         "present locally.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many subjects, for a bounded first pass.")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.subjects and not args.all:
        raise SystemExit("pass --subjects or --all")

    if args.all:
        top = list_keys(f"{DATASET}/sub-")
        # Each subject also has a sibling sub-XXX.html fMRIPrep report at the
        # dataset root; those are not subject directories and would otherwise
        # eat slots out of --limit.
        subs = sorted({p[1] for p in (Path(k).parts for k, _ in top)
                       if len(p) > 2 and p[1].startswith("sub-") and "." not in p[1]})
        # A subject counts as done only with a full complement of runs, so a
        # download interrupted midway is retried rather than silently skipped.
        have = {d.name for d in args.dest.glob("sub-*")
                if len(list(d.rglob("*space-fsaverage6_bold.func.gii"))) >= 20}
        subs = [s for s in subs if s not in have]
        if have:
            print(f"already complete, skipping: {', '.join(sorted(have))}")
    else:
        subs = args.subjects

    if args.limit:
        subs = subs[:args.limit]
    if not subs:
        print("nothing to fetch")
        return
    print(f"fetching {len(subs)} subject(s): {', '.join(subs)}\n")

    grand_bytes = grand_files = 0
    for sub in subs:
        keys = [(k, s) for k, s in list_keys(f"{DATASET}/{sub}/") if wanted(k)]
        if not keys:
            print(f"{sub}: no task-visualmemory fsaverage6 files found, skipping")
            continue
        todo = []
        for k, s in keys:
            p = local_path(args.dest, k)
            if p is None:
                continue
            if p.exists() and p.stat().st_size == s:
                continue
            todo.append((k, p, s))
        total = sum(s for _, _, s in todo)
        print(f"{sub}: {len(keys)} matching runs, {len(todo)} to download "
              f"({human(total)})")
        if args.dry_run:
            continue
        for i, (k, p, s) in enumerate(todo, 1):
            try:
                got = download(k, p, s)
                grand_bytes += got
                grand_files += 1
                print(f"  [{i}/{len(todo)}] {p.name}  {human(got)}", flush=True)
            except Exception as e:
                print(f"  [{i}/{len(todo)}] FAILED {p.name}: {e}", flush=True)

    print(f"\ndone: {grand_files} files, {human(grand_bytes)}")
    print("re-run the gate with:  python scripts/hyperface_signal_ceiling.py")


if __name__ == "__main__":
    main()

"""
modal_strip_data_history.py

Rewrites janice-stfu's history to remove data/ entirely, on Modal.

WHY MODAL. The repo is ~14.7GB and clones to this Mac at ~0.05MB/s (~82h).
Modal's network does it in minutes. This step is READ-ONLY with respect to
GitHub: it clones, rewrites LOCALLY, verifies, and stores the result in a
Modal Volume. Nothing is pushed, and no GitHub credential is used here --
the push is a separate, explicitly confirmed step.

Commits that contained ONLY data/ files become empty once data/ is gone, and
filter-repo's default --prune-empty auto drops them (user's choice: drop, not
keep as empty no-ops). Every other commit keeps its message, order, authorship
and date; only the SHAs change, which rewriting makes unavoidable.

Run:
  modal run scripts/modal_strip_data_history.py
"""
import modal

app = modal.App("janice-strip-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")
    .pip_install("git-filter-repo")
)

vol = modal.Volume.from_name("janice-filtered", create_if_missing=True)

REPO = "https://github.com/aloshdenny/janice-stfu.git"
STRIP = "data"


@app.function(image=image, timeout=7200, volumes={"/vol": vol},
              cpu=4.0, memory=16384)
def strip():
    import subprocess, os, shutil

    def sh(cmd, cwd=None, check=True):
        r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                           capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{cmd}\nSTDOUT{r.stdout}\nSTDERR{r.stderr}")
        return r.stdout.strip()

    work = "/tmp/janice-mirror"
    shutil.rmtree(work, ignore_errors=True)
    print("cloning...", flush=True)
    sh(f"git clone --mirror {REPO} {work}")
    before_size = sh(f"du -sm {work}").split()[0]
    before_commits = sh("git rev-list --count --all", cwd=work)
    print(f"BEFORE: {before_size} MB, {before_commits} commits", flush=True)

    # exactly which commits are data-only (these are the ones that would be
    # dropped under filter-repo's default --prune-empty auto)
    data_only = []
    shas = sh("git rev-list --all", cwd=work).split()
    for s in shas:
        files = sh(f"git show --pretty=format: --name-only {s}", cwd=work, check=False)
        names = [f for f in files.split("\n") if f.strip()]
        if names and all(f.startswith(STRIP + "/") or f == STRIP for f in names):
            msg = sh(f"git log -1 --format=%s {s}", cwd=work)
            data_only.append((s[:8], len(names), msg))
    print(f"\ndata-only commits (WILL BE DROPPED): {len(data_only)}")
    for s, n, m in data_only:
        print(f"  {s}  {n:5d} files   {m}")

    print("\nfiltering...", flush=True)
    sh(f"git filter-repo --path {STRIP} --invert-paths --prune-empty auto "
       f"--prune-degenerate auto --force", cwd=work)

    after_commits = sh("git rev-list --count --all", cwd=work)
    sh("git reflog expire --expire=now --all", cwd=work, check=False)
    sh("git gc --prune=now --aggressive", cwd=work, check=False)
    after_size = sh(f"du -sm {work}").split()[0]

    # VERIFY: no object anywhere in history still lives under data/
    leftover = sh(
        "git rev-list --objects --all | git cat-file --batch-check="
        "'%(objecttype) %(objectname) %(rest)' | "
        f"awk '$1==\"blob\"' | grep -c '  *{STRIP}/' || true", cwd=work)
    print(f"\nAFTER: {after_size} MB, {after_commits} commits")
    print(f"VERIFY blobs still under {STRIP}/: {leftover}  (must be 0)")

    if leftover.strip() not in ("0", ""):
        raise RuntimeError(f"{STRIP}/ still present after filtering -- aborting")
    expected = int(before_commits) - len(data_only)
    if int(after_commits) != expected:
        raise RuntimeError(f"commit count {after_commits}, expected {expected} "
                           f"({before_commits} - {len(data_only)} data-only)")

    after_log = sh("git log --all --format='%H %ad %an %s' --date=short", cwd=work)
    open("/vol/after_log.txt", "w").write(after_log)
    bundle = "/vol/janice-filtered.bundle"
    sh(f"git bundle create {bundle} --all", cwd=work)
    bundle_mb = sh(f"du -sm {bundle}").split()[0]
    vol.commit()
    print(f"\nbundle: {bundle_mb} MB -> {bundle}")

    return {"before_size_mb": before_size, "after_size_mb": after_size,
            "before_commits": before_commits, "after_commits": after_commits,
            "bundle_mb": bundle_mb, "data_only": data_only}


@app.local_entrypoint()
def main():
    r = strip.remote()
    print("\n==== SUMMARY ====")
    print(f"size    {r['before_size_mb']} MB -> {r['after_size_mb']} MB")
    print(f"commits {r['before_commits']} -> {r['after_commits']} (preserved)")
    print(f"bundle  {r['bundle_mb']} MB")
    print(f"data-only commits kept as empty: {len(r['data_only'])}")

"""
verify_reproducible.py

Proves a fresh clone of this repo is actually runnable: every oversized artifact
ships as chunks, every chunk set really re-fuses, and no script reaches for a
chunked file with a raw loader that would silently not find it.

WHY THIS EXISTS. Files over GitHub's 100MB limit are committed as
`{stem}_chunk_NNN{ext}` parts, so the fused file does NOT exist in a fresh
clone. Code that calls `np.load(path)` or `path.exists()` on the logical name
then fails -- or worse, quietly skips the data, which is how Michael Jackson
silently vanished from plot_postop_activity.py's output. chunk_utils has
chunk-aware equivalents (`load_npz`, `npz_exists`, `save_npz`, `load_chunked`,
`save_chunked`, `ensure_fused_zip`); this checks they're the ones being used.

Run from the repo root:
  python scripts/verify_reproducible.py
Exit code 0 = reproducible, 1 = something would break for a fresh cloner.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from chunk_utils import get_chunk_paths, npz_exists, load_npz, fuse_chunks_to_file

REPO = Path(__file__).resolve().parent.parent
GITHUB_HARD_LIMIT = 100 * 1024 * 1024
CHUNK_RE = re.compile(r"^(.*)_chunk_(\d{3})(\..+)$")

failures, warnings, skipped = [], [], []


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    return [REPO / f for f in out.stdout.splitlines() if f]


def check_no_oversized(files):
    print("\n[1] no committed file exceeds GitHub's 100MB hard limit")
    bad = [f for f in files if f.is_file() and f.stat().st_size > GITHUB_HARD_LIMIT]
    for f in bad:
        failures.append(f"{f.relative_to(REPO)} is {f.stat().st_size/1e6:.0f}MB (>100MB)")
    print(f"    checked {len(files)} tracked files, {len(bad)} oversized")


def check_chunk_sets(files):
    print("\n[2] every chunk set re-fuses and loads")
    bases = set()
    for f in files:
        m = CHUNK_RE.match(f.name)
        if m:
            bases.add(f.parent / (m.group(1) + m.group(3)))
    if not bases:
        print("    no chunked artifacts found")
    for base in sorted(bases):
        rel = base.relative_to(REPO)
        parts = get_chunk_paths(base)
        if not parts:
            # Tracked but not checked out: normal in a partial/sparse clone, which
            # is the recommended way to get just the code. Not a defect -- only an
            # artifact that IS present and fails to reconstruct is a real failure.
            skipped.append(str(rel))
            continue
        if base.exists():
            warnings.append(f"{rel}: fused file present locally -- gitignore it so "
                            f"the chunked path is what actually gets exercised")
        try:
            if base.suffix == ".npz":
                data = load_npz(base)
                print(f"    OK  {rel}  ({len(parts)} parts -> {len(data)} arrays)")
            else:
                tmp = Path("/tmp") / f"_verify{base.suffix}"
                fuse_chunks_to_file(base, tmp)
                print(f"    OK  {rel}  ({len(parts)} parts -> {tmp.stat().st_size/1e6:.0f}MB)")
                tmp.unlink(missing_ok=True)
        except Exception as e:
            failures.append(f"{rel}: chunk set does not reconstruct ({e})")
    if skipped:
        print(f"    {len(skipped)} artifact(s) tracked but not checked out "
              f"(partial/sparse clone) -- skipped, not a failure:")
        for sk in skipped:
            print(f"      - {sk}")


def check_loaders(files):
    """Static check: flag a raw loader only when it is actually applied to a
    COMMITTED chunked artifact.

    Precision matters more than recall here. An earlier version flagged any raw
    loader in a file that merely mentioned a chunked name somewhere, which
    produced 21 warnings that were nearly all false (torch.load of the TRIBE
    checkpoint, .exists() on unrelated paths). A check people learn to ignore
    protects nothing. So: resolve the variables that actually hold a chunked
    artifact path, then flag only loaders applied to those.
    """
    print("\n[3] no script uses a raw loader on a committed chunked artifact")
    chunked_stems = set()
    for f in files:
        m = CHUNK_RE.match(f.name)
        if m:
            chunked_stems.add(m.group(1) + m.group(3))

    raw_call = re.compile(r"\b(np\.load|np\.savez|np\.savez_compressed|torch\.load|torch\.save)\s*\(\s*([A-Za-z_][\w\.\[\]\"\']*)")
    raw_exists = re.compile(r"([A-Za-z_][\w\.]*)\.exists\s*\(\s*\)")
    flagged = 0

    for py in sorted((REPO / "scripts").glob("*.py")):
        if py.name in ("chunk_utils.py", "verify_reproducible.py"):
            continue
        lines = py.read_text(errors="ignore").splitlines()

        # variables bound to a committed chunked artifact
        bound = set()
        for line in lines:
            if any(stem in line for stem in chunked_stems) and "=" in line:
                lhs = line.split("=", 1)[0].strip()
                if re.fullmatch(r"[A-Za-z_]\w*", lhs):
                    bound.add(lhs)

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            targets = []
            m = raw_call.search(line)
            if m:
                targets.append((m.group(1), m.group(2)))
            m2 = raw_exists.search(line)
            if m2:
                targets.append((".exists()", m2.group(1)))
            for call, arg in targets:
                base = arg.split(".")[0].split("[")[0].strip("\"'")
                hits_var = base in bound
                hits_lit = any(stem in line for stem in chunked_stems)
                if hits_var or hits_lit:
                    flagged += 1
                    failures.append(
                        f"{py.name}:{i} `{call}` on chunked artifact `{base}` "
                        f"-> use chunk_utils (load_npz/npz_exists/load_chunked)")

    print(f"    {len(chunked_stems)} committed chunked artifact(s); {flagged} raw-loader use(s)")


def main():
    files = tracked_files()
    check_no_oversized(files)
    check_chunk_sets(files)
    check_loaders(files)

    print("\n" + "=" * 62)
    for w in warnings:
        print(f"  WARN  {w}")
    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        print(f"\nNOT REPRODUCIBLE: {len(failures)} blocking issue(s)")
        return 1
    tail = ""
    if skipped:
        tail = (f"; {len(skipped)} artifact(s) not checked out in this clone, "
                f"run `git sparse-checkout disable` to fetch and verify them")
    print(f"\nREPRODUCIBLE: chunk sets reconstruct, nothing exceeds the size limit{tail}"
          f"{', ' + str(len(warnings)) + ' warning(s)' if warnings else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

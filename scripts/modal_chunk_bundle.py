"""
modal_chunk_bundle.py

Splits the filtered bundle into verifiable chunks.

WHY. A single 9.1GB download over this link corrupted mid-stream (git hit
"inflate returned -5" at offset ~5.9GB). `git bundle verify` did NOT catch it
-- it only validates the bundle header, refs and prerequisites, not the pack
payload. Chunking with per-chunk sha256 makes corruption detectable and makes
only the bad chunk need re-fetching instead of all 9GB.

Run:
  modal run scripts/modal_chunk_bundle.py
"""
import modal

app = modal.App("janice-chunk-bundle")
image = modal.Image.debian_slim(python_version="3.11").apt_install("coreutils")
vol = modal.Volume.from_name("janice-filtered")


@app.function(image=image, timeout=3600, volumes={"/vol": vol}, cpu=4.0)
def chunk():
    import subprocess, os, hashlib, glob

    src = "/vol/janice-filtered.bundle"
    out = "/vol/chunks"
    subprocess.run(f"rm -rf {out} && mkdir -p {out}", shell=True, check=True)

    size = os.path.getsize(src)
    whole = hashlib.sha256()
    with open(src, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            whole.update(b)
    print(f"bundle {size} bytes  sha256={whole.hexdigest()}", flush=True)

    subprocess.run(f"split -b 512M -d -a 3 {src} {out}/part_", shell=True, check=True)

    lines = [f"WHOLE {size} {whole.hexdigest()}"]
    for p in sorted(glob.glob(f"{out}/part_*")):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        lines.append(f"{os.path.basename(p)} {os.path.getsize(p)} {h.hexdigest()}")
    open(f"{out}/MANIFEST.txt", "w").write("\n".join(lines) + "\n")
    vol.commit()
    print(f"{len(lines)-1} chunks written")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(chunk.remote())

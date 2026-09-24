"""
modal_mj_bottleneck.py

Captures Michael Jackson's 2048-dim bottleneck features on a Modal GPU.

WHY THIS EXISTS. Post-surgery activity is base + X @ (U@V), so everything
needed to draw a post-surgery brain map is local EXCEPT X, the bottleneck,
which only the encoder can produce. Mia and Sins already have stored baseline
predictions locally (X is recoverable from those by least squares), MJ does
not -- so he needs one real forward pass. This runs only that pass and brings
back X, not a whole training job.

Run:
  modal run scripts/modal_mj_bottleneck.py
"""
import modal

app = modal.App("e85-mj-bottleneck")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.6.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "curl -sL -o /tmp/tribev2.tar.gz "
        "https://github.com/facebookresearch/tribev2/archive/refs/heads/main.tar.gz",
        "mkdir -p /opt/tribev2 && tar xzf /tmp/tribev2.tar.gz -C /opt/tribev2 --strip-components=1",
        "cd /opt/tribev2 && pip install -e .",
    )
    .pip_install("opencv-python-headless", "nilearn", "pandas")
    .add_local_dir("scripts", "/root/e85/scripts", copy=True)
    .add_local_file("target/mj.zip", "/root/e85/target/mj.zip", copy=True)
)

# HF cache kept in a Volume so the ~700MB TRIBE checkpoint (plus the V-JEPA2
# backbone) downloads once, not on every retry.
hf_cache = modal.Volume.from_name("e85-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600,
              volumes={"/cache": hf_cache})
def capture_mj():
    import sys, os
    os.chdir("/root/e85")
    sys.path.insert(0, "/root/e85/scripts")
    os.environ["HF_HOME"] = "/cache/hf"

    import numpy as np
    from pathlib import Path
    from mia_suppress_readout import capture_bottlenecks, find_fmri_encoder
    from atl_suppress_readout import load_all_from_zip
    from infer_fairface_bulk import get_tmp_root

    items = load_all_from_zip(Path("target/mj.zip"))
    print(f"loaded {len(items)} MJ images", flush=True)

    model, fe = find_fmri_encoder(Path("/cache/tribe"))
    W0 = fe.predictor.weights[0].detach().cpu().numpy()
    b0 = fe.predictor.bias[0].detach().cpu().numpy()
    print(f"readout W0 {W0.shape} b0 {b0.shape}", flush=True)

    X, names = capture_bottlenecks(model, fe, items, 16, get_tmp_root())
    print(f"captured X {X.shape}", flush=True)

    hf_cache.commit()
    return {"X": X, "names": names, "W0": W0, "b0": b0}


@app.local_entrypoint()
def main():
    import sys
    import numpy as np
    sys.path.append("scripts")
    from chunk_utils import save_npz
    out = capture_mj.remote()
    # save_npz, not np.savez: this file is ~169MB, over GitHub's 100MB limit, so
    # it must ship as _chunk_NNN parts for the repo to stay clone-and-run
    save_npz("abliterated/mj_bottleneck.npz",
             X=out["X"], names=np.array(out["names"]),
             W0=out["W0"], b0=out["b0"])
    print(f"saved abliterated/mj_bottleneck.npz  X={out['X'].shape}")

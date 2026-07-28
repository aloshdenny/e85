"""
smoke_test_model_identity.py

Everything we've tried assumed model.predict() eventually calls forward()
on the SAME vjepa2_module object we hold a reference to and mutate via
load_state_dict(). smoke_test_vjepa2.py proved that calling
vjepa2_module(...) DIRECTLY responds correctly to weight changes. But
validation.py's full model.predict() path -- even across two fully
independent TribeModel instances with separate cache folders, i.e. with
caching entirely ruled out -- still shows exactly zero delta.

That combination only makes sense if predict()'s internal extraction path
is NOT using the object we mutated at all. The config showed
model.data.video_feature.image.infra.keep_in_ram = False -- in a job-
execution framework (exca/neuralset MapInfra), that plausibly means the
extractor reconstructs a fresh model internally per job/call rather than
reusing a persisted object, discarding whatever we mutated beforehand.

This script tests that directly: capture the vjepa2 module's Python object
id AND a weight checksum before and after a trivial predict() call (no
surgery involved at all -- just load, predict once, recheck).

  If id changes           -> the extractor is creating a NEW model object
                              internally during predict(). Any external
                              mutation via load_state_dict() before calling
                              predict() is irrelevant -- proven root cause.
  If id is same but        -> something else swapped/reset the weights of
  checksum reverts             the same object during predict() (less likely
                                but distinct from the above -- worth knowing
                                which one it is).
  If id and checksum        -> the object truly persists across predict()
  are both stable               calls, which would mean our surgery mutation
                                 SHOULD have worked and something else is
                                 going on (e.g. wrong module path targeted,
                                 or the checkpoint applied to a different
                                 attribute than the one predict() reads).

Usage:
  python scripts/diagnostics/smoke_test_model_identity.py
"""

import os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2
import torch

import sys
sys.path.append(str(Path(__file__).parent))
from infer_fairface_bulk import get_tmp_root, write_static_clip, make_multi_row_df
from tribev2.demo_utils import TribeModel


def checksum(module_or_tensor):
    if isinstance(module_or_tensor, torch.Tensor):
        t = module_or_tensor
    else:
        t = module_or_tensor.weight
    return float(t.detach().sum().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-folder", default="./cache_identity_test", type=Path)
    parser.add_argument("--probe-layer", type=int, default=5)
    args = parser.parse_args()

    tmp_root = get_tmp_root()
    tmp_dir = Path(tempfile.mkdtemp(prefix="identity_test_", dir=tmp_root))

    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)

    # ── Capture BEFORE any predict() call ────────────────────────────────
    vjepa2_before = model.data.video_feature.image.model.model
    id_before = id(vjepa2_before)
    encoder_before = vjepa2_before.encoder.layer
    W_before = checksum(encoder_before[args.probe_layer].attention.value.weight)

    print(f"\nBEFORE predict():")
    print(f"  id(vjepa2_module)     = {id_before}")
    print(f"  id(model.data.video_feature.image.model.model) same object? "
          f"{model.data.video_feature.image.model.model is vjepa2_before}")
    print(f"  weight checksum @ L{args.probe_layer} = {W_before:.6f}")

    # ── Trivial predict() call, no surgery, just to see if anything moves ──
    try:
        img = np.zeros((64, 64, 3), dtype=np.uint8)  # blank frame, content irrelevant
        clip_path = tmp_dir / "probe.mp4"
        write_static_clip(img, clip_path, duration=1.0, fps=2)
        df = make_multi_row_df([(clip_path, "identity_probe")], duration=1.0)
        print("\nRunning a single trivial predict() call...")
        preds, segments = model.predict(events=df)
        print(f"  predict() completed, preds.shape={preds.shape}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Capture AFTER predict() ──────────────────────────────────────────
    vjepa2_after = model.data.video_feature.image.model.model
    id_after = id(vjepa2_after)
    encoder_after = vjepa2_after.encoder.layer
    W_after = checksum(encoder_after[args.probe_layer].attention.value.weight)

    print(f"\nAFTER predict():")
    print(f"  id(model.data.video_feature.image.model.model) = {id_after}")
    print(f"  weight checksum @ L{args.probe_layer} = {W_after:.6f}")

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    if id_before != id_after:
        print(">> OBJECT IDENTITY CHANGED. The extractor reconstructs a NEW model")
        print("   object internally during predict() -- this is the root cause.")
        print("   Any load_state_dict() call on the object we hold BEFORE calling")
        print("   predict() gets discarded, because predict() never uses that")
        print("   object -- it builds its own fresh copy from model_name=")
        print("   'facebook/vjepa2-vitg-fpc64-256' (or similar) internally.")
        print("   FIX: the abliterated weights need to be saved to a local")
        print("   directory in HF checkpoint format, and the extractor's")
        print("   model_name/pretrained path needs to be pointed at that local")
        print("   directory instead of the original HF hub identifier, so")
        print("   whatever object it reconstructs each call loads OUR weights.")
    elif abs(W_after - W_before) > 1e-6:
        print(">> SAME object identity, but weight checksum changed after predict().")
        print("   Something reset/reloaded weights on the SAME object during the")
        print("   call -- different mechanism than full reconstruction, worth")
        print("   digging into video_feature.image's _get_data / prepare methods.")
    else:
        print(">> Object identity AND weights both stable across predict().")
        print("   The module we're mutating genuinely does persist. If load_state_dict")
        print("   before predict() still shows zero effect in validation.py, re-check:")
        print("   (a) is model.data.video_feature.image.model.model really the exact")
        print("       callable predict() invokes, or is there another wrapper layer")
        print("       (e.g. a torch.jit.trace / torch.compile artifact) sitting on top")
        print("       that was captured at construction time and never re-reads")
        print("       updated parameters;")
        print("   (b) whether load_result.missing_keys/unexpected_keys were truly")
        print("       both empty on the actual run, not just this smoke test.")


if __name__ == "__main__":
    main()

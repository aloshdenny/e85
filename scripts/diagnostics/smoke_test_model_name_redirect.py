"""
smoke_test_model_name_redirect.py

smoke_test_model_identity.py proved the extractor reconstructs a fresh
vjepa2 model internally on every predict() call (object id changed, and
"Loading weights" printed again during predict()). That means the fix is
NOT to mutate the module we hold a reference to -- it's to redirect WHERE
the extractor reloads from, so that whatever fresh object it builds each
call loads OUR abliterated weights instead of the original HF checkpoint.

This test checks the specific mechanism: does
  model.data.video_feature.image.model_name
get read LIVE on every predict() call, or was it resolved once at
TribeModel construction time and is now fixed regardless of what we set
afterward?

Method: set model_name to an obviously bogus string, then call predict().
  If predict() raises an error trying to load that bogus name/path
    -> model_name IS read live. Redirecting it to a local HF-format
       checkpoint directory (containing our abliterated weights) should
       work as the real fix.
  If predict() succeeds anyway (loads normally, no error)
    -> model_name is fixed/cached elsewhere at construction time, and this
       simple attribute mutation won't work. A different approach is
       needed (e.g. overwriting the HF cache snapshot our repo id
       resolves to, or monkeypatching the extractor's load method).

Usage:
  python scripts/diagnostics/smoke_test_model_name_redirect.py
"""

import os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import tempfile
import shutil
from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).parent))
from infer_fairface_bulk import get_tmp_root, write_static_clip, make_multi_row_df
from tribev2.demo_utils import TribeModel


def main():
    cache_folder = Path("./cache_redirect_test")
    tmp_root = get_tmp_root()
    tmp_dir = Path(tempfile.mkdtemp(prefix="redirect_test_", dir=tmp_root))

    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)

    original_name = model.data.video_feature.image.model_name
    print(f"Original model_name: {original_name}")

    bogus_name = "this/does-not-exist-anywhere-xyz123"
    model.data.video_feature.image.model_name = bogus_name
    print(f"Set model_name to bogus value: {bogus_name}")

    try:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        clip_path = tmp_dir / "probe.mp4"
        write_static_clip(img, clip_path, duration=1.0, fps=2)
        df = make_multi_row_df([(clip_path, "redirect_probe")], duration=1.0)

        print("\nRunning predict() with bogus model_name set...")
        preds, segments = model.predict(events=df)
        print(f"predict() SUCCEEDED anyway. preds.shape={preds.shape}")
        print("\n>> VERDICT: model_name is NOT read live -- it was resolved/cached")
        print("   elsewhere at construction time. Simple attribute redirection won't")
        print("   work as the fix. Need a different approach (e.g. overwriting the")
        print("   HF cache snapshot the original repo id resolves to, or")
        print("   monkeypatching the extractor's internal load call).")
    except Exception as e:
        print(f"\npredict() FAILED as expected: {type(e).__name__}: {e}")
        print("\n>> VERDICT: model_name IS read live on every predict() call.")
        print("   Redirecting it to a local HF-format checkpoint directory containing")
        print("   our abliterated weights (via vjepa2_module.save_pretrained(...))")
        print("   should work as the real fix.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

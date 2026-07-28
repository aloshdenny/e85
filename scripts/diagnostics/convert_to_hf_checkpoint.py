"""
convert_to_hf_checkpoint.py

You already have vjepa2_face_abliterated.pt (a raw state_dict) from the full
abliteration run. Rather than re-running that entire (expensive) pipeline
just to get save_pretrained() called at the end, this loads a fresh vjepa2
model, applies your existing state_dict, and saves it in HF checkpoint
format (config.json + weights) -- the format validation.py's model_name
redirect actually needs, since TribeModel's extractor reconstructs the
model via AutoModel.from_pretrained(model_name), not load_state_dict().

Usage:
  python scripts/diagnostics/convert_to_hf_checkpoint.py \
      --state-dict ./abliterated_face/vjepa2_face_abliterated.pt \
      --out-dir ./abliterated_face/vjepa2_hf_checkpoint
"""

import argparse
import warnings, logging, os
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
from pathlib import Path
from tribev2.demo_utils import TribeModel


def load_checkpoint_state_dict(checkpoint_path: Path):
    try:
        import chunk_utils
        if chunk_utils.check_chunked_exists(checkpoint_path):
            print("  Detected chunked checkpoint -- fusing via chunk_utils...")
            return chunk_utils.load_chunked(checkpoint_path, map_location="cpu")
    except ImportError:
        pass
    return torch.load(checkpoint_path, map_location="cpu")


def checksum(module, probe_layer=5, param_path="attention.value.weight"):
    mod = module.encoder.layer[probe_layer]
    for part in param_path.split("."):
        mod = getattr(mod, part)
    return float(mod.detach().sum().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dict", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache-folder", default="./cache", type=Path)
    parser.add_argument("--probe-layer", type=int, default=5)
    parser.add_argument("--source-model-name", default="facebook/vjepa2-vitg-fpc64-256",
                        help="Original HF model id, used only to fetch the (unmodified) "
                             "video processor config to save alongside your weights.")
    args = parser.parse_args()

    print("Loading TribeModel (to get a correctly-configured vjepa2 instance)...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    vjepa2_module = model.data.video_feature.image.model.model

    checksum_before = checksum(vjepa2_module, args.probe_layer)
    print(f"Checksum BEFORE loading state_dict (L{args.probe_layer}): {checksum_before:.6f}")

    print(f"\nLoading state_dict: {args.state_dict}")
    state_dict = load_checkpoint_state_dict(args.state_dict)
    load_result = vjepa2_module.load_state_dict(state_dict, strict=False)
    n_total_keys = len(dict(vjepa2_module.named_parameters()))
    print(f"  missing_keys={len(load_result.missing_keys)} "
          f"unexpected_keys={len(load_result.unexpected_keys)} "
          f"(module has {n_total_keys} total parameter tensors)")
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"  [!!] missing: {load_result.missing_keys[:5]}")
        print(f"  [!!] unexpected: {load_result.unexpected_keys[:5]}")

    checksum_after = checksum(vjepa2_module, args.probe_layer)
    print(f"Checksum AFTER loading state_dict (L{args.probe_layer}):  {checksum_after:.6f}")

    diff = abs(checksum_after - checksum_before)
    print(f"\n|checksum_after - checksum_before| = {diff:.8f}")
    if diff < 1e-8:
        print("  [!!!] IDENTICAL. load_state_dict did NOT change this parameter --")
        print("        the state_dict file's keys likely don't match this module's")
        print("        parameter names even though strict=False silently allowed it")
        print("        through. Whatever gets saved below will be the ORIGINAL")
        print("        pretrained weights, not your abliteration. STOP HERE and")
        print("        inspect state_dict.keys() vs vjepa2_module.state_dict().keys()")
        print("        directly before proceeding -- this is very likely the actual")
        print("        root cause of the zero-delta result in validation.py, unrelated")
        print("        to any caching or redirect issue.")
    else:
        print("  >> DIFFERENT. load_state_dict genuinely changed this parameter.")

    print(f"\nSaving HF-format checkpoint -> {args.out_dir}")
    vjepa2_module.save_pretrained(args.out_dir)

    # The extractor's _HFVideoModel loads BOTH the model AND a video processor
    # via Processor.from_pretrained(model_name, ...) -- our local directory only
    # had the model until now. The processor itself is unmodified by surgery
    # (we only touched model weights), so it's correct to just copy the
    # ORIGINAL processor config in alongside our modified weights.
    print(f"Saving (unmodified) video processor config alongside it...")
    from transformers import AutoVideoProcessor
    processor = AutoVideoProcessor.from_pretrained(args.source_model_name)
    processor.save_pretrained(args.out_dir)
    print(f"  Processor config saved -> {args.out_dir}")

    # ── Reload from disk and re-check -- catches any save/load round-trip bug ──
    print("\nVerifying round-trip: reloading the saved checkpoint from disk...")
    from transformers import AutoModel
    reloaded = AutoModel.from_pretrained(args.out_dir)
    checksum_reloaded = checksum(reloaded, args.probe_layer)
    print(f"Checksum after save+reload from disk: {checksum_reloaded:.6f}")
    if abs(checksum_reloaded - checksum_after) < 1e-8:
        print("  >> Round-trip matches in-memory value. save_pretrained()/from_pretrained() are consistent.")
    else:
        print("  [!!!] Round-trip MISMATCH -- something is lost/altered in the save or reload step itself.")

    print("\nDone. Point validation.py's model_name redirect at this directory.")


if __name__ == "__main__":
    main()

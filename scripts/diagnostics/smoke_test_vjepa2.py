"""
smoke_test_vjepa2.py

Isolates WHERE the zero-delta bug lives. Two independent checks:

  CHECK A -- Is the checkpoint actually different from the original weights?
    Compares a specific parameter tensor before/after load_state_dict(),
    and reports load_state_dict()'s missing/unexpected keys (which are
    silently swallowed if you don't capture the return value).

  CHECK B -- Does a RAW vjepa2 forward pass differ, bypassing
    model.predict() / video_feature / caching entirely?
    Calls vjepa2_module(pixel_values_videos=...) directly on a fixed
    synthetic input, before and after loading the checkpoint, with no
    TribeModel.predict(), no events dataframe, no cache layer involved
    at all.

How to read the result:
  A differs, B differs     -> weights loaded fine, forward pass responds.
                               Bug is in video_feature's caching/pipeline
                               (it's not using this module object, or
                               caching survives everything we've tried).
  A differs, B does NOT    -> load_state_dict() "succeeded" (no error) but
                               isn't actually changing the module that
                               executes forward() -- e.g. a wrapper/copy
                               issue. Check load_state_dict's return value
                               printed below for missing/unexpected keys.
  A does NOT differ        -> the checkpoint itself has identical weights
                               to the original. Something went wrong in
                               face_abliteration.py's save step, or the
                               object it saved wasn't the one it modified.

Usage:
  python scripts/diagnostics/smoke_test_vjepa2.py --checkpoint ./abliterated_face/vjepa2_face_abliterated.pt
"""

import argparse
import warnings, logging, os
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
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


def extract_hidden(output):
    """
    VJEPA2Model's top-level forward returns a VJEPA2WithMaskedInputModelOutput
    dataclass (ModelOutput), not a plain tuple -- different from the per-layer
    hook outputs used elsewhere in this pipeline, which ARE tuples.
    """
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "to_tuple"):
        return output.to_tuple()[0]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-folder", default="./cache", type=Path)
    parser.add_argument("--probe-layer", type=int, default=5,
                        help="Layer index to inspect for CHECK A. Should be one "
                             "of the layers you actually operated on.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache_folder)
    vjepa2_module = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    n_layers = len(encoder_blocks)
    vjepa2_module.eval()
    vjepa2_module.to(device)
    print(f"Encoder layers: {n_layers}")
    print(f"vjepa2_module id: {id(vjepa2_module)}")
    print(f"video_feature.image.model.model path resolves to: {type(vjepa2_module)}")

    # ── Fixed synthetic input for CHECK B ────────────────────────────────
    torch.manual_seed(0)
    fixed_input = torch.randn(1, 16, 3, 256, 256).to(device)

    with torch.no_grad():
        out_before = vjepa2_module(pixel_values_videos=fixed_input)
    hidden_before = extract_hidden(out_before)
    hidden_before = hidden_before.detach().float().cpu().numpy()

    # ── CHECK A: parameter comparison + load_state_dict return value ────
    print("\n" + "="*60)
    print(f"CHECK A -- parameter diff at encoder.layer[{args.probe_layer}]")
    print("="*60)
    probe_param_name = "attention.value.weight"
    mod = encoder_blocks[args.probe_layer]
    for part in probe_param_name.split("."):
        mod = getattr(mod, part)
    W_before = mod.detach().clone().cpu().numpy() if isinstance(mod, torch.Tensor) else mod.data.clone().cpu().numpy()
    print(f"  W_before checksum: {W_before.sum():.6f}  norm: {np.linalg.norm(W_before):.6f}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    state_dict = load_checkpoint_state_dict(args.checkpoint)
    load_result = vjepa2_module.load_state_dict(state_dict, strict=False)
    print(f"  missing_keys:    {len(load_result.missing_keys)}  {load_result.missing_keys[:5]}")
    print(f"  unexpected_keys: {len(load_result.unexpected_keys)}  {load_result.unexpected_keys[:5]}")
    if load_result.missing_keys or load_result.unexpected_keys:
        print("  [!!] Non-empty missing/unexpected keys -- this alone could mean "
              "the checkpoint's keys don't line up with this module's state_dict "
              "(e.g. saved from a differently-wrapped object), so load_state_dict "
              "silently did less than you think.")

    mod2 = encoder_blocks[args.probe_layer]
    for part in probe_param_name.split("."):
        mod2 = getattr(mod2, part)
    W_after = mod2.detach().clone().cpu().numpy() if isinstance(mod2, torch.Tensor) else mod2.data.clone().cpu().numpy()
    print(f"\n  W_after checksum:  {W_after.sum():.6f}  norm: {np.linalg.norm(W_after):.6f}")

    param_diff = np.abs(W_after - W_before).max()
    print(f"\n  max |W_after - W_before| = {param_diff:.8f}")
    if param_diff < 1e-8:
        print("  >> CHECK A: IDENTICAL. The checkpoint's weights at this layer "
              "match the original exactly. Something is wrong upstream -- likely "
              "in face_abliteration.py's save step (saved before surgery mutated "
              "the module, or saved a different object than the one modified).")
    else:
        print("  >> CHECK A: DIFFERENT. Checkpoint genuinely contains modified "
              "weights at this layer, and load_state_dict applied them here.")

    # ── CHECK B: raw forward pass, same object, same input ──────────────
    print("\n" + "="*60)
    print("CHECK B -- raw vjepa2 forward pass (bypasses predict()/cache entirely)")
    print("="*60)
    with torch.no_grad():
        out_after = vjepa2_module(pixel_values_videos=fixed_input)
    hidden_after = extract_hidden(out_after)
    hidden_after = hidden_after.detach().float().cpu().numpy()

    forward_diff = np.abs(hidden_after - hidden_before).max()
    print(f"  max |hidden_after - hidden_before| = {forward_diff:.8f}")
    if forward_diff < 1e-8:
        print("  >> CHECK B: IDENTICAL forward output despite (possibly) different "
              "weights. If CHECK A showed a real weight diff, this means "
              "vjepa2_module's forward() isn't sensitive to this parameter for "
              "this input, OR something is re-fetching original weights between "
              "the two calls (e.g. .eval()/device placement resetting a cached "
              "compiled graph, or a wrapper re-instantiating the model).")
    else:
        print("  >> CHECK B: DIFFERENT. Raw forward pass responds to the weight "
              "change. If model.predict() still shows zero delta, the bug is "
              "confirmed to be in video_feature's pipeline/caching layer -- it's "
              "either not using this exact module object, or some other cache "
              "survived both our clearing attempts.")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  CHECK A (weights differ):  {'YES' if param_diff > 1e-8 else 'NO'}")
    print(f"  CHECK B (forward differs): {'YES' if forward_diff > 1e-8 else 'NO'}")


if __name__ == "__main__":
    main()

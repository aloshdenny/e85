"""
infer_face.py

Infer a TribeV2 brain-response map from a single static face image.

TribeModel's pipeline (get_events_dataframe -> predict) is built around video
input — it predicts a time-resolved response, not a single-frame one. Since we
want to reuse the exact same trained pipeline (no internals hacking), this
script:

  1. Synthesizes a short static video by holding the face image for
     `--duration` seconds at `--fps` (so the model sees a constant "stimulus"
     with no motion/scene-cut signal).
  2. Runs it through the identical get_events_dataframe -> predict path used
     in infer.py.
  3. Collapses the resulting per-timestep predictions into a single
     representative brain map (mean over time — appropriate since the input
     is static and any timestep-to-timestep variance is just model/decoding
     noise, not signal).
  4. Renders and saves the brain map as a PNG, and saves the raw per-ROI
     prediction vector as .npy so you can feed it directly into the
     difference-in-means step later.

Usage:
    python scripts/infer_face.py --image /path/to/face.jpg --out ./face_outputs/face_0001
    python scripts/infer_face.py --image /path/to/face.jpg --out ./face_outputs/face_0001 --duration 3 --fps 5

Batch usage (called from a driver script iterating over FairFace + target set):
    from scripts.infer_face import run_single_image
    pred_vector = run_single_image(image_path, out_dir, model=model)
"""

import os
import sys
import warnings
import logging
import argparse
import time
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain


# ── Xvfb setup (same as infer.py) ────────────────────────────────────────────

def setup_xvfb(display_num=99):
    display = f":{display_num}"
    os.environ["DISPLAY"] = display
    lock_file = f"/tmp/.X{display_num}-lock"
    socket_file = f"/tmp/.X11-unix/X{display_num}"

    is_running = False
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                content = f.read().strip()
                if content:
                    pid = int(content.split()[0])
                    try:
                        os.kill(pid, 0)
                        is_running = True
                    except OSError:
                        pass
        except Exception:
            pass

    if is_running:
        print(f"Xvfb already running on display {display}.")
        return

    for path in [lock_file, socket_file]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Warning: could not remove stale X11 file {path}: {e}")

    print(f"Starting Xvfb on display {display}...")
    os.system(f"Xvfb {display} -screen 0 1024x768x24 > /dev/null 2>&1 &")
    time.sleep(1)


# ── Image -> static video ────────────────────────────────────────────────────

def image_to_static_video(image_path, out_video_path, duration=2.0, fps=5, size=None):
    """
    Hold a single face image for `duration` seconds at `fps` to produce a
    short static video TribeModel can ingest via get_events_dataframe.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if size is not None:
        img = cv2.resize(img, size)

    h, w = img.shape[:2]
    n_frames = max(1, int(round(duration * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        writer.write(img)
    writer.release()

    return out_video_path


# ── Core single-image inference ──────────────────────────────────────────────

def run_single_image(image_path, out_dir, model=None, cache_folder=Path("./cache"),
                      duration=2.0, fps=5, save_render=True):
    """
    Run TribeV2 inference on a single face image and return the mean
    prediction vector (collapsed over the synthetic-video timesteps).

    Returns
    -------
    mean_pred : np.ndarray, shape (n_rois,) or whatever preds.shape[1:] is
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_folder)

    tmp_video_path = out_dir / "_static_input.mp4"
    image_to_static_video(image_path, tmp_video_path, duration=duration, fps=fps)

    df = model.get_events_dataframe(video_path=tmp_video_path)
    preds, segments = model.predict(events=df)

    # preds shape: (n_timesteps, n_rois) — collapse timesteps since the
    # stimulus is static. Keep the raw array too in case you want to inspect
    # timestep-to-timestep stability as a sanity check.
    np.save(out_dir / "preds_raw.npy", preds)
    mean_pred = preds.mean(axis=0)
    np.save(out_dir / "pred_mean.npy", mean_pred)

    if save_render:
        render_brain_map(mean_pred, segments[0] if len(segments) else None, out_dir / "brain_map.png")

    # clean up the synthetic video, keep only the arrays/render
    try:
        tmp_video_path.unlink()
    except OSError:
        pass

    return mean_pred


def render_brain_map(pred_vector, segment, out_path):
    """
    Render a single-timestep brain map for a (mean) prediction vector.
    pred_vector needs a leading timestep axis for PlotBrain.plot_timesteps,
    so we reshape to (1, n_rois).
    """
    setup_xvfb(99)

    plotter = PlotBrain(mesh="fsaverage5")
    preds_chunk = pred_vector.reshape(1, -1)
    segments_chunk = [segment] if segment is not None else None

    fig = plotter.plot_timesteps(
        preds_chunk,
        segments=segments_chunk,
        cmap="fire",
        norm_percentile=99,
        alpha_cmap=(0, 0.2),
        show_stimuli=False,
    )
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    plt.figure(figsize=(img.shape[1] / 150, img.shape[0] / 150))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved brain map -> {Path(out_path).resolve()}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Infer TribeV2 brain map for a single face image.")
    parser.add_argument("--image", required=True, help="Path to face image (jpg/png).")
    parser.add_argument("--out", required=True, help="Output directory for arrays + rendered map.")
    parser.add_argument("--duration", type=float, default=2.0, help="Synthetic static-video duration (s).")
    parser.add_argument("--fps", type=int, default=5, help="Synthetic static-video frame rate.")
    parser.add_argument("--cache-folder", default="./cache", help="TribeModel cache folder.")
    args = parser.parse_args()

    setup_xvfb(99)

    print(f"Loading TribeV2...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=Path(args.cache_folder))

    print(f"Running inference on: {args.image}")
    mean_pred = run_single_image(
        image_path=args.image,
        out_dir=args.out,
        model=model,
        duration=args.duration,
        fps=args.fps,
    )

    print(f"Prediction vector shape: {mean_pred.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
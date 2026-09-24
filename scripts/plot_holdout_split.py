"""
plot_holdout_split.py

One figure explaining WHY the holdout had to change.

A random holdout samples the whole score distribution, so most of the photos
it tests on are the target's WEAK ones (bad angle, poor light, half-turned) --
easy to suppress, and the reported drop flatters the method. The face_top
split instead holds out the highest-scoring photos under the frozen readout:
the ones the model is most confident about. Same gallery, much harder test.

Scores here are the frozen readout's own FACE(OFA+FFA) mean per photo, from
the baseline predictions already committed in target_preds/.

Usage:
  python scripts/plot_holdout_split.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from chunk_utils import load_npz, preds_as_image_vectors
from measure_identity_signal import build_masks

OUT = Path("abliterated/brain_maps/holdout_split.png")
HOLDOUT_FRAC = 0.20
SEED = 0


def main():
    masks = build_masks()
    face = masks["FACE(OFA+FFA)"]

    P = preds_as_image_vectors(load_npz("target_preds/mia.npz")["preds"])
    scores = P[:, face].mean(axis=1)
    n_ho = int(round(len(scores) * HOLDOUT_FRAC))

    top_idx = np.argsort(-scores)[:n_ho]
    rng = np.random.default_rng(SEED)
    rand_idx = rng.choice(len(scores), n_ho, replace=False)

    top_mean, rand_mean = scores[top_idx].mean(), scores[rand_idx].mean()
    cutoff = scores[top_idx].min()

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.hist(scores, bins=40, color="#9aa7b8", edgecolor="white", linewidth=0.6,
            label=f"all {len(scores)} photos of the target")
    ax.axvspan(cutoff, scores.max(), color="#e8563f", alpha=0.16, zorder=0)

    ax.axvline(rand_mean, color="#3b6ea8", lw=2.6,
               label=f"random holdout, mean {rand_mean:+.4f}")
    ax.axvline(top_mean, color="#e8563f", lw=2.6,
               label=f"top holdout (what we use), mean {top_mean:+.4f}")

    ax.annotate("", xy=(top_mean, ax.get_ylim()[1] * 0.74),
                xytext=(rand_mean, ax.get_ylim()[1] * 0.74),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6))
    ax.text((rand_mean + top_mean) / 2, ax.get_ylim()[1] * 0.78,
            f"the test got {top_mean - rand_mean:+.4f} harder",
            ha="center", fontsize=11, weight="bold", color="#222")

    ax.text(cutoff + (scores.max() - cutoff) * 0.5, ax.get_ylim()[1] * 0.30,
            "her strongest\nphotos", ha="center", fontsize=10,
            color="#a8341f", style="italic")

    ax.set_xlabel("how strongly the frozen model's face cortex responds to this photo", fontsize=11)
    ax.set_ylabel("number of photos", fontsize=11)
    ax.set_title("A random holdout tests the easy photos", fontsize=14, pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"random holdout mean {rand_mean:+.5f} | top holdout mean {top_mean:+.5f} "
          f"| delta {top_mean - rand_mean:+.5f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()

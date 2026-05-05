"""
demo_stitch_eval.py
===================
Minimal smoke-test that exercises all three functions in stitch_eval.py
without requiring real point-cloud data.
"""

import numpy as np
import matplotlib.pyplot as plt
from surfile.stitcher.comparator import compute_deltas, plot_stitching_comparison, colorize_deltas

rng = np.random.default_rng(0)

# ── synthetic "ground truth" surface ──────────────────────────────────────
N = 300
t = np.linspace(0, 4 * np.pi, N)
base = np.column_stack([t, np.sin(t), np.cos(t)])   # helix, shape (300, 3)

# ── three "stitched" results with progressively larger errors ──────────────
noise_scales = [0.02, 0.08, 0.15]
labels        = ["Fine stitch", "Coarse stitch", "Rough stitch"]
stitched_list = [base + rng.normal(0, s, base.shape) for s in noise_scales]

# ── compute_R: constant radius for this demo ──────────────────────────────
def compute_R(point):           # noqa: D401
    """Return a fixed neighbourhood radius."""
    return 0.4

# ── 1. compute_deltas for the first method ────────────────────────────────
deltas = compute_deltas(stitched_list[0], compute_R)
print(f"deltas shape : {deltas.shape}")           # (300, 3)
print(f"max |delta|  : {np.linalg.norm(deltas, axis=1).max():.4f}")

# ── 2. comparison plot ────────────────────────────────────────────────────
fig = plot_stitching_comparison(
    stitched_list,
    compute_R,
    noise_threshold=0.05,
    labels=labels,
    figsize_scale=4,
)
fig.savefig("stitching_comparison.png", dpi=120, bbox_inches="tight")
print("Saved  stitching_comparison.png")
plt.close(fig)

# ── 3. colorize_deltas ────────────────────────────────────────────────────
colors = colorize_deltas(deltas)
print(f"colors shape : {colors.shape}")           # (300, 4)

# Quick scatter to verify the green→red gradient
fig2, ax = plt.subplots(figsize=(10, 3))
ax.scatter(np.arange(N), np.linalg.norm(deltas, axis=1),
           c=colors, s=12, edgecolors="none")
ax.set_xlabel("point index")
ax.set_ylabel("|Δ|")
ax.set_title("colorize_deltas – green = small error, red = large error")
fig2.savefig("colorize_demo.png", dpi=120, bbox_inches="tight")
print("Saved  colorize_demo.png")
plt.close(fig2)

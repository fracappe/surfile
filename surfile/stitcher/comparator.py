"""
comparator.py
==============
Utilities for evaluating and comparing N point-cloud stitching algorithms.

Public API
----------
compute_deltas(stitched, compute_R)
    For every point in a stitched surface, find the local neighbourhood
    inside a radius R, compute the mean of that neighbourhood, and return
    the vector difference between the point and that mean.

plot_stitching_comparison(stitched_list, compute_R, noise_threshold,
                          labels=None, figsize_scale=5)
    Produce a diagnostic figure with
      • one row per stitching method   – |Δ|, Δx, Δy, Δz
      • one row per pair of methods    – same four quantities on the
                                         difference signal (method_i − method_j)
    Each subplot is annotated with the RMSE of the samples whose modulus
    exceeds `noise_threshold`.

colorize_deltas(deltas)
    Return an (N, 4) RGBA array that maps the modulus of every delta to a
    green→red gradient (0 → max), suitable for colouring a point cloud
    scatter or a matplotlib scatter plot.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial import KDTree

from surfile.stitcher import stitcher as sstitcher


def make_compute_R(point_clouds_T):
    R_vals = []

    for i in range(len(point_clouds_T) - 1):
        fixed_pts = point_clouds_T[i]
        aligned = point_clouds_T[i + 1]

        fix_sub, temp_sub = sstitcher.Isolator.isolate_common_points_kdtree(fixed_pts, aligned, bins_after_max=1, bplt=False)

        R = np.nanmean(fix_sub) - np.nanmean(temp_sub)
        R_vals.append(abs(R))

    R_value = np.mean(R_vals)

    # return the function expected by compute_deltas
    def compute_R():
        return R_value

    return compute_R
# ---------------------------------------------------------------------------
# 1.  compute_deltas
# ---------------------------------------------------------------------------

def compute_deltas(
    stitched: np.ndarray,
    compute_R: Callable[[np.ndarray], float],
) -> np.ndarray:
    """
    Compute the local-mean residual for every point in *stitched*.

    Parameters
    ----------
    stitched : ndarray, shape (N, 3)
        The stitched point cloud to analyse.
    compute_R : callable (point: ndarray shape (3,)) -> float
        External function that returns the neighbourhood radius for a given
        point.  It is called once per point.

    Returns
    -------
    deltas : ndarray, shape (N, 3)
        ``deltas[i] = stitched[i] − mean(neighbourhood_i)``
        where neighbourhood_i contains every point of *stitched* whose
        Euclidean distance from ``stitched[i]`` is ≤ R_i (inclusive of the
        point itself).
    """
    stitched = np.asarray(stitched, dtype=float)
    n = len(stitched)
    tree = KDTree(stitched)

    deltas = np.empty((n, 3), dtype=float)
    R = float(compute_R())
    
    print(f"Starting query_ball_point execution...")
    idx = tree.query_ball_point(stitched, r=R)
    
    for i, (point, neighbours) in enumerate(zip(stitched, idx)):    
        
        if i % 1000 == 0:
            print(f"Computing deltas: {i}/{n} points processed...", end="\r")
        
        neighbourhood = stitched[neighbours]          # always contains point itself
        mean_vec = neighbourhood.mean(axis=0)
        deltas[i] = point - mean_vec

    print("\n")
    return deltas

    # for i, point in enumerate(stitched):
        
    #     if i % 1000 == 0:
    #         print(f"Computing deltas: {i}/{n} points processed...")

    #     # query_ball_point returns indices of all points within radius R
    #     idx = tree.query_ball_point(point, r=R)
    #     neighbourhood = stitched[idx]          # always contains point itself
    #     mean_vec = neighbourhood.mean(axis=0)
    #     deltas[i] = point - mean_vec

    # return deltas


# ---------------------------------------------------------------------------
# 2.  plot_stitching_comparison
# ---------------------------------------------------------------------------

_COMPONENT_LABELS = ("x", "y", "z")
_COL_TITLES = ("|Δ|", "Δx", "Δy", "Δz")


def _modulus(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 norm of an (N, 3) array → (N,)."""
    return np.linalg.norm(arr, axis=1)


def _rmse_above_threshold(signal: np.ndarray, threshold: float) -> float | None:
    """RMSE of samples in *signal* whose absolute value exceeds *threshold*.

    Returns None when no sample exceeds the threshold.
    """
    mask = np.abs(signal) > threshold
    if not mask.any():
        return None
    return float(np.sqrt(np.mean(signal[mask] ** 2)))


def _annotate_rmse(ax: plt.Axes, signal: np.ndarray, threshold: float) -> None:
    """Draw a horizontal threshold line and annotate the RMSE above it."""
    ax.axhline(threshold, color="grey", linewidth=0.8, linestyle="--", alpha=0.7,
               label=f"threshold = {threshold:.3g}")
    rmse = _rmse_above_threshold(signal, threshold)
    if rmse is not None:
        ax.text(
            0.99, 0.97,
            f"RMSE>{threshold:.3g} = {rmse:.4g}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7, ec="none"),
        )


def _plot_row(
    axes: np.ndarray,          # shape (4,)
    deltas: np.ndarray,        # (N, 3)
    threshold: float,
    row_label: str,
    color: str,
) -> None:
    """Fill one row of four subplots for a given *deltas* array."""
    x = np.arange(len(deltas))
    mod = _modulus(deltas)

    signals = [mod, deltas[:, 0], deltas[:, 1], deltas[:, 2]]

    for ax, sig, col_title in zip(axes, signals, _COL_TITLES):
        ax.plot(x, sig, lw=0.8, color=color, alpha=0.85)
        ax.set_title(f"{row_label}  —  {col_title}", fontsize=8, pad=3)
        ax.set_xlabel("point index", fontsize=7)
        ax.tick_params(labelsize=7)
        _annotate_rmse(ax, sig, threshold)
        ax.legend(fontsize=6, loc="upper left")


def plot_stitching_comparison(
    stitched_list: Sequence[np.ndarray],
    compute_R: Callable[[np.ndarray], float],
    noise_threshold: float,
    labels: Sequence[str] | None = None,
    figsize_scale: float = 5.0,
) -> plt.Figure:
    """
    Produce a comprehensive diagnostic figure for N stitching methods.

    Layout
    ------
    • **Top block** – one row per method, four columns: |Δ|, Δx, Δy, Δz.
    • **Bottom block** – one row per unique pair (i, j) with i < j, same
      four columns but plotting ``deltas_i − deltas_j``.

    Each subplot shows:
      • the signal curve,
      • a dashed horizontal line at ±*noise_threshold*,
      • the RMSE computed only over samples whose |value| > noise_threshold.

    Parameters
    ----------
    stitched_list : sequence of ndarray, each (N, 3)
        One stitched surface per algorithm, **in the same point order**.
    compute_R : callable
        Passed directly to :func:`compute_deltas`.
    noise_threshold : float
        Values below this are considered noise-floor and excluded from RMSE.
    labels : sequence of str, optional
        Human-readable names for each method.  Defaults to "Method 0",
        "Method 1", …
    figsize_scale : float
        Rough width/height per subplot in inches.  Default 5.

    Returns
    -------
    fig : matplotlib.Figure
    """
    n_methods = len(stitched_list)
    if labels is None:
        labels = [f"Method {i}" for i in range(n_methods)]

    # --- compute all deltas up-front -----------------------------------
    all_deltas = [compute_deltas(s, compute_R) for s in stitched_list]

    # --- pair-wise combinations ----------------------------------------
    pairs = list(itertools.combinations(range(n_methods), 2))
    n_rows = n_methods + len(pairs)
    n_cols = 4

    palette = plt.cm.tab10.colors  # up to 10 distinct colours

    fig = plt.figure(figsize=(figsize_scale * n_cols, figsize_scale * 0.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.35)

    # --- per-method rows -----------------------------------------------
    for row_idx, (deltas, label) in enumerate(zip(all_deltas, labels)):
        axes = np.array([fig.add_subplot(gs[row_idx, col]) for col in range(n_cols)])
        color = palette[row_idx % len(palette)]
        _plot_row(axes, deltas, noise_threshold, label, color)

    # --- pair-wise difference rows -------------------------------------
    for pair_idx, (i, j) in enumerate(pairs):
        row_idx = n_methods + pair_idx
        diff = all_deltas[i] - all_deltas[j]
        pair_label = f"{labels[i]} − {labels[j]}"
        axes = np.array([fig.add_subplot(gs[row_idx, col]) for col in range(n_cols)])
        # Use a muted colour for difference rows
        color = palette[(n_methods + pair_idx) % len(palette)]
        _plot_row(axes, diff, noise_threshold, pair_label, color)

    fig.suptitle(
        "Stitching algorithm comparison\n"
        f"(noise threshold = {noise_threshold:.3g})",
        fontsize=11, y=1.01,
    )
    return fig


# ---------------------------------------------------------------------------
# 3.  colorize_deltas
# ---------------------------------------------------------------------------

def colorize_deltas(
    deltas: np.ndarray,
    alpha: float = 1.0,
    vmax: float | None = None,
) -> np.ndarray:
    """
    Map the modulus of each delta to an RGBA colour via a green → red lerp.

    The colour of point *i* is fully determined by ``|deltas[i]|`` relative
    to the maximum modulus in the array (or *vmax* if supplied), so the
    colour index is stable regardless of which subset of points you visualise.

    Parameters
    ----------
    deltas : ndarray, shape (N, 3)
        Output of :func:`compute_deltas`.
    alpha : float
        Uniform opacity for all points (0–1).  Default 1.
    vmax : float, optional
        The modulus value that maps to pure red.  Defaults to the maximum
        modulus found in *deltas*.

    Returns
    -------
    colors : ndarray, shape (N, 4)  dtype float64, values in [0, 1]
        RGBA colours – green (0, 1, 0, α) at modulus 0,
        red  (1, 0, 0, α) at modulus *vmax*.

    Notes
    -----
    The array index of each colour matches the array index of the
    corresponding point in *stitched* / *deltas*, which is guaranteed to be
    unchanged throughout the stitching pipeline.
    """
    deltas = np.asarray(deltas, dtype=float)
    mod = _modulus(deltas)                     # (N,)

    if vmax is None:
        vmax = mod.max()

    # Normalise to [0, 1], guard against all-zero case
    t = mod / vmax if vmax > 0 else np.zeros_like(mod)
    t = np.clip(t, 0.0, 1.0)

    # Lerp: green (0,1,0) → red (1,0,0)
    # R channel: 0 → 1,  G channel: 1 → 0,  B channel: 0 → 0
    r = t
    g = 1.0 - t
    b = np.zeros_like(t)
    a = np.full_like(t, float(alpha))

    return np.stack([r, g, b, a], axis=1)      # (N, 4)

"""
comparator.py
==============
Utilities for evaluating and comparing N point-cloud stitching algorithms.

BallQuery
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
-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

Dense Map Posterior (DMP) method for evaluating stitching quality.
----------
compute_dmp_error(point_clouds_T)
    For each point cloud, compute the squared distance to the nearest point
    in the combined point cloud of all other point clouds, then sum these
    errors across all point clouds.

class DensityMapPosterior
    Main class to evaluate stitching results using the Dense Map Posterior
    method. Computes alignment quality by measuring distances between
    individual point clouds and the rest of the stitched surface.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial import KDTree

from surfile.stitcher import stitcher as sstitcher
from surfile.stitcher import plotter as splotter


def make_compute_R(point_clouds_T):
    """Create a function that computes the neighbourhood radius for deltas computation.
    
    Calculates the mean Z-offset between consecutive point clouds and returns
    a callable that provides this radius value.
    
    Parameters
    ----------
    point_clouds_T : list[ndarray]
        List of point clouds (transformed), where each element is an (N, 3) array.
    
    Returns
    -------
    callable
        A function that returns the computed neighbourhood radius.
    """
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
    
    print(f"Starting KDTree query_ball_point execution with radius {R}...")
    idx = tree.query_ball_point(stitched, r=R) # This operation can be slow for large point clouds
    
    for i, (point, neighbours) in tqdm(enumerate(zip(stitched, idx)), total=n, desc="Computing deltas", colour='cyan'):            
        neighbourhood = stitched[neighbours]          # always contains point itself
        mean_vec = neighbourhood.mean(axis=0)
        deltas[i] = point - mean_vec

    print("\n")
    return deltas

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
        # sig[abs(sig) > threshold] = np.nan # zero out values below threshold for better visualization
        ax.plot(x, sig, lw=0.8, color=color, alpha=0.85)
        ax.set_title(f"{row_label}  —  {col_title}", fontsize=8, pad=3)
        ax.set_xlabel("point index", fontsize=7)
        ax.tick_params(labelsize=7)
        _annotate_rmse(ax, sig, threshold)
        ax.legend(fontsize=6, loc="upper left")

def _plot_hist_row(
    axes: np.ndarray,          # shape (4,)
    deltas: np.ndarray,        # (N, 3)
    threshold: float,
    row_label: str,
    color: str,
) -> None:
    """Fill one row of four subplots for a given *deltas* array."""
    mod = _modulus(deltas)

    signals = [mod, deltas[:, 0], deltas[:, 1], deltas[:, 2]]

    for ax, sig, col_title in zip(axes, signals, _COL_TITLES):
        ax.hist(sig, bins='rice', color=color, alpha=0.85)
        ax.set_title(f"{row_label}  —  {col_title}", fontsize=8, pad=3)
        ax.set_xlabel("value", fontsize=7)
        ax.tick_params(labelsize=7)
        _annotate_rmse(ax, sig, threshold)
        ax.legend(fontsize=6, loc="upper right")

def _plot_comparison_figure(
    stitched_results: dict,
    deltas: dict,
    plot_function: Callable,
    title_prefix: str,
    noise_threshold: float,
    labels: Sequence[str] | None = None,
    figsize_scale: float = 5.0
) -> plt.Figure:
    """
    Generic helper to generate a comparison figure for stitching methods.
    This function contains the common logic for plot_deltas and plot_histograms.
    """
    if not deltas:
        print("[WARN COMPARATOR] Deltas not computed yet. Run compute_all_deltas() first.")
        return None

    n_methods = len(stitched_results)
    if labels is None:
        labels = [f"Method {i}" for i in range(n_methods)]

    # --- pair-wise combinations ----------------------------------------
    pairs = list(itertools.combinations(range(n_methods), 2))
    n_rows = n_methods + len(pairs)
    n_cols = 4

    palette = plt.cm.tab10.colors  # up to 10 distinct colours

    fig = plt.figure(figsize=(figsize_scale * n_cols, figsize_scale * 0.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)
    
    deltas_all = list(deltas.values())

    # --- per-method rows -----------------------------------------------
    for row_idx, (delta_data, label) in enumerate(zip(deltas_all, labels)):
        axes = np.array([fig.add_subplot(gs[row_idx, col]) for col in range(n_cols)])
        color = palette[row_idx % len(palette)]
        plot_function(axes, delta_data, noise_threshold, label, color)

    # --- pair-wise difference rows -------------------------------------
    for pair_idx, (i, j) in enumerate(pairs):
        row_idx = n_methods + pair_idx
        diff = deltas_all[i] - deltas_all[j]
        pair_label = f"{labels[i]} − {labels[j]}"
        axes = np.array([fig.add_subplot(gs[row_idx, col]) for col in range(n_cols)])
        color = palette[(n_methods + pair_idx) % len(palette)]
        plot_function(axes, diff, noise_threshold, pair_label, color)

    fig.suptitle(
        f"{title_prefix}\n"
        f"(noise threshold = {noise_threshold:.3g})",
        fontsize=11, y=1.01,
    )
    return fig
    

class BallQuery:
    """
    A class to encapsulate the comparison of stitching results.
    """
    def __init__(self, stitched_results: dict[str, tuple[np.ndarray, list[np.ndarray]]]):
        """Initialize the Comparator with stitching results.
        
        Parameters
        ----------
        stitched_results : dict[str, tuple[ndarray, list[ndarray]]]
            Dictionary mapping method names to tuples of (stitched_surface, point_clouds_T)
            where stitched_surface is the final stitched point cloud and
            point_clouds_T is the list of individual (transformed) point clouds.
        """
        self.stitched_results = stitched_results
        self.deltas = {}

    def compute_all_deltas(self):
        """Compute local-mean residuals for all stitching methods.
        
        Calculates deltas for each stitched result and stores them in self.deltas.
        Must be called before plotting or colormapping.
        """
        for method_name, restuple in self.stitched_results.items():
            stitched, stitched_T = restuple
            cR = make_compute_R(stitched_T)
            self.deltas[method_name] = compute_deltas(stitched, cR)
        self._print_summary()

    def _print_summary(self):
        """
        Print a summary of delta statistics for all methods.
        For each method, prints the scipy stats describe of |Δ|, Δx, Δy, Δz in a pretty format.
        """
        print("\n" + "=" * 70)
        print("Ball Query Statistics Summary")
        print("=" * 70)
        for method_name, delta_data in self.deltas.items():
            mod = _modulus(delta_data)
            stats_mod = scipy.stats.describe(mod, nan_policy='omit')
            stats_x = scipy.stats.describe(delta_data[:, 0], nan_policy='omit')
            stats_y = scipy.stats.describe(delta_data[:, 1], nan_policy='omit')
            stats_z = scipy.stats.describe(delta_data[:, 2], nan_policy='omit')

            print(f"\nMethod: {method_name}")
            print(f"{'Component':>10s} | {'Count':>10s} | {'Mean':>12s} | {'StdDev':>12s} | {'Min':>12s} | {'Max':>12s}")
            print("-" * 120)
            for comp_label, stats in zip(_COL_TITLES, [stats_mod, stats_x, stats_y, stats_z]):
                count, (minn, maxx), mean, stddev, skew, kurt = stats
                print(f"{comp_label:>10s} | {count:10d} | {mean:12.6g} | {stddev:12.6g} | {minn:12.6g} | {maxx:12.6g}")

    def plot_deltas(self, noise_threshold: float, labels: Sequence[str] | None = None, figsize_scale: float = 5.0) -> plt.Figure:
        """Generate diagnostic comparison figure for stitching methods.
        
        Produces a figure with one row per stitching method and one row per
        pair of methods, showing |Δ|, Δx, Δy, Δz with RMSE annotations.
        
        Parameters
        ----------
        noise_threshold : float
            Threshold value for RMSE annotation of outliers.
        labels : sequence[str], optional
            Labels for each stitching method. If None, uses "Method 0", "Method 1", etc.
        figsize_scale : float, optional
            Scaling factor for figure size (default: 5.0).
        
        Returns
        -------
        Figure
            The matplotlib figure object, or None if deltas haven't been computed.
        """
        return _plot_comparison_figure(
            self.stitched_results,
            self.deltas,
            _plot_row,
            "Stitching algorithm comparison",
            noise_threshold,
            labels,
            figsize_scale
        )
    
    def plot_histograms(self, noise_threshold: float, labels: Sequence[str] | None = None, figsize_scale: float = 5.0) -> plt.Figure:
        """Generate histogram comparison figure for stitching methods.
        
        Similar to plot_deltas but uses histograms instead of line plots.
        """
        if labels is None: labels = list(self.stitched_results.keys())
        return _plot_comparison_figure(
            self.stitched_results,
            self.deltas,
            _plot_hist_row,
            "Stitching algorithm comparison (histograms)",
            noise_threshold,
            labels,
            figsize_scale
        )

    def colormap_deltas(self, mode='modulus'):
        """Visualize deltas using colormapped point clouds.
        
        Parameters
        ----------
        mode : str, optional
            Coloring mode:
            - 'modulus' (default): color by |Δ|
            - 'x', 'y', 'z': color by component
            - 'xyz': color by 3D RGB weights
        """
        if self.deltas == {}:
            print("[WARN COMPARATOR] Deltas not computed yet. Run compute_all_deltas() first.")
            return

        make_plot = lambda factor: splotter.compare_point_clouds(
            [[pc] for _, (pc, _) in self.stitched_results.items()],
            [[splotter.get_colors_from_weights('plasma', factor(self.deltas[method]))] for method in self.stitched_results.keys()],
            names=[f"{method} - {mode}" for method in self.stitched_results.keys()]
        )
        
        if mode == 'modulus':
            make_plot(lambda deltas: _modulus(deltas))
        elif mode == 'x':
            make_plot(lambda deltas: deltas[:, 0])
        elif mode == 'y':
            make_plot(lambda deltas: deltas[:, 1])
        elif mode == 'z':
            make_plot(lambda deltas: deltas[:, 2])
        elif mode == 'xyz':
            splotter.compare_point_clouds(
                [[pc] for _, (pc, _) in self.stitched_results.items()],
                [[splotter.get_rgb_from_3d_weights(self.deltas[method])] for method in self.stitched_results.keys()],
                names=[f"{method} - XYZ" for method in self.stitched_results.keys()]
            )
        else:
            print(f"[WARN COMPARATOR] Unknown colormap mode '{mode}'. Supported modes: 'modulus', 'x', 'y', 'z', 'xyz'.")

    def plot(self, cmap="plasma"):
        """Plot all stitched point clouds side-by-side for visual comparison.
        
        Parameters
        ----------
        cmap : str, optional
            Colormap name for point cloud visualization (default: "plasma").
        """
        splotter.compare_point_clouds([[pc] for _, (pc, _) in self.stitched_results.items()], cmap)


def compute_dmp_error(point_clouds_T: list[np.ndarray]) -> dict[str, float]:
    """
    Compute the Dense Map Posterior (DMP) error for a list of transformed point clouds.
    
    The DMP method evaluates stitching quality by:
    1. For each point cloud i, finding nearest neighbours in all other point clouds
    2. Computing squared distances to these nearest neighbours
    3. Summing squared distances per point cloud
    4. Summing all point cloud errors to get total error
    
    Parameters
    ----------
    point_clouds_T : list[ndarray]
        List of transformed point clouds, where each element is an (N, 3) array.
    
    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'total_error': Sum of all errors across all point clouds
        - 'per_cloud_errors': List of errors for each point cloud
        - 'mean_error_per_cloud': Mean error per point cloud
    """
    point_clouds_T = [np.asarray(pc, dtype=float) for pc in point_clouds_T]
    n_clouds = len(point_clouds_T)
    
    if n_clouds < 2:
        raise ValueError("At least 2 point clouds are required for DMP evaluation")
    
    mean_total_error = 0.0
    squared_total_error = 0.0
    per_cloud_mean_errors = []
    per_cloud_squared_errors = []
    
    # For each point cloud
    for i in range(n_clouds):
        current_cloud = point_clouds_T[i]
        
        # Combine all other point clouds
        other_clouds = np.vstack([point_clouds_T[j] for j in range(n_clouds) if j != i])
        
        # Build KDTree for efficient nearest neighbour search
        tree = KDTree(other_clouds)
        
        # Find nearest neighbour distances for each point in current cloud
        distances, _ = tree.query(current_cloud, k=1)
        
        # Average of errors for this point cloud
        mean_error = np.mean(distances)
        per_cloud_mean_errors.append(mean_error)
        mean_total_error += mean_error

        # Squared sum of errors of distances
        squared_error = np.sqrt(np.sum(distances ** 2))
        per_cloud_squared_errors.append(squared_error)
        squared_total_error += squared_error

        print(f"Point cloud {i}: error = {mean_error:.6f}, mean distance = {mean_error:.6f}, squared error = {squared_error:.6f}")
    
    mean_error_per_cloud = np.mean(per_cloud_mean_errors)
    squared_error_per_cloud = np.mean(per_cloud_squared_errors)
    
    print(f"\nTotal DMP error: {mean_total_error / n_clouds:.6f}")
    print(f"Mean error per cloud: {mean_error_per_cloud:.6f}")
    print(f"Squared error per cloud: {squared_error_per_cloud:.6f}")
    
    return {
        'total_error': float(mean_total_error / n_clouds),
        'per_cloud_errors': per_cloud_mean_errors,
        'per_cloud_squared_errors': per_cloud_squared_errors,
        'mean_error_per_cloud': float(mean_error_per_cloud),
        'squared_error_per_cloud': float(squared_error_per_cloud),
    }


class DensityMapPosterior:
    """
    A class to evaluate stitching quality using the Dense Map Posterior method.
    """
    def __init__(self, stitched_results: dict[str, tuple[np.ndarray, list[np.ndarray]]]):
        """
        Initialize the DensityMapPosterior evaluator with stitching results.
        
        Parameters
        ----------
        stitched_results : dict[str, tuple[ndarray, list[ndarray]]]
            Dictionary mapping method names to tuples of (stitched_surface, point_clouds_T)
            where stitched_surface is the final stitched point cloud and
            point_clouds_T is the list of individual (transformed) point clouds.
        """
        self.stitched_results = stitched_results
        self.dmp_errors = {}
    
    def compute_all_dmp_errors(self) -> None:
        """
        Compute DMP errors for all stitching methods.
        
        Results are stored in self.dmp_errors as a dictionary mapping method names
        to error dictionaries containing 'total_error', 'per_cloud_errors', and
        'mean_error_per_cloud'.
        """
        print("=" * 70)
        print("Computing DMP errors for all stitching methods")
        print("=" * 70)
        
        for method_name, (stitched, point_clouds_T) in self.stitched_results.items():
            print(f"\n--- Method: {method_name} ---")
            errors = compute_dmp_error(point_clouds_T)
            self.dmp_errors[method_name] = errors
        
        print("\n" + "=" * 70)
        print("DMP Evaluation Summary")
        print("=" * 70)
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print a summary of DMP errors for all methods."""
        if not self.dmp_errors:
            print("[WARN DMP] No errors computed yet. Run compute_all_dmp_errors() first.")
            return
        
        # Sort methods by total error for ranking
        sorted_methods = sorted(
            self.dmp_errors.items(),
            key=lambda x: x[1]['total_error']
        )
        
        print("\nRanking by total DMP error (lower is better):")
        for rank, (method_name, errors) in enumerate(sorted_methods, 1):
            summary_items = []
            for key, value in errors.items():
                if key == 'per_cloud_errors' or key == 'per_cloud_squared_errors':
                    pass
                else:
                    label = key.replace('_', ' ').title()

                if isinstance(value, float):
                    summary_items.append(f"{label}: {value:12.6f}")

            print(f"  {rank}. {method_name:<25s} | {' | '.join(summary_items)}")
    
    def get_dmp_errors(self) -> dict[str, dict]:
        """
        Get the computed DMP errors for all methods.
        
        Returns
        -------
        dict[str, dict]
            Dictionary mapping method names to error dictionaries.
        """
        if not self.dmp_errors:
            print("[WARN DMP] No errors computed yet. Run compute_all_dmp_errors() first.")
            return {}
        return self.dmp_errors
    
    def get_best_method(self) -> tuple[str, float] | None:
        """
        Get the best performing method (lowest total DMP error).
        
        Returns
        -------
        tuple[str, float] or None
            Tuple of (method_name, total_error), or None if no errors computed.
        """
        if not self.dmp_errors:
            print("[WARN DMP] No errors computed yet. Run compute_all_dmp_errors() first.")
            return None
        
        best_method = min(
            self.dmp_errors.items(),
            key=lambda x: x[1]['total_error']
        )
        return (best_method[0], best_method[1]['total_error'])

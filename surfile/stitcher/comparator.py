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
      • one row per stitching method   – |$Delta$|, $Delta$x, $Delta$y, $Delta$z
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
import pickle
import os
import pathlib

from surfile.stitcher import stitcher as sstitcher
from surfile.stitcher import plotter as splotter
from surfile.stitcher import pipeline as spipe
from surfile.stitcher import utils as sutils

import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial import cKDTree


_COMPONENT_LABELS = ("x", "y", "z")
_COL_TITLES = (fr"$\left|\Delta\right|$", r"$\Delta$x", r"$\Delta$y", r"$\Delta$z")

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
    figsize_scale: float = 5.0,
    show_differences: bool = True
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

    # --- Axis Sharing Setup ---
    share_x_ax = None
    share_y_ax = None

    # --- per-method rows -----------------------------------------------
    for row_idx, (delta_data, label) in enumerate(zip(deltas_all, labels)):
        axes_list = []
        for col in range(n_cols):
            # Create subplot, sharing axes with the very first subplot created
            ax = fig.add_subplot(gs[row_idx, col], sharex=share_x_ax, sharey=share_y_ax)
            
            # Set the baseline reference if it's the first plot
            if share_x_ax is None:
                share_x_ax = ax
                share_y_ax = ax
                
            axes_list.append(ax)
            
        axes = np.array(axes_list)        
        color = palette[row_idx % len(palette)]
        plot_function(axes, delta_data, noise_threshold, label, color)

        # --- pair-wise difference rows -------------------------------------
    if show_differences:
        for pair_idx, (i, j) in enumerate(pairs):
            row_idx = n_methods + pair_idx
            diff = deltas_all[i] - deltas_all[j]
            pair_label = f"{labels[i]} − {labels[j]}"

            axes_list = []
            for col in range(n_cols):
                ax = fig.add_subplot(gs[row_idx, col], sharex=share_x_ax, sharey=share_y_ax)
                axes_list.append(ax)

            axes = np.array(axes_list)
            color = palette[(n_methods + pair_idx) % len(palette)]
            plot_function(axes, diff, noise_threshold, pair_label, color)

    fig.suptitle(
        f"{title_prefix}\n"
        f"(noise threshold = {noise_threshold:.3g})",
        fontsize=11, y=1.01,
    )

    return fig


class Comparator:
    """
    Main class to compare stitching results using various metrics and visualizations.
    """
    stitched_results: dict[str, tuple[np.ndarray, list[np.ndarray]]]
    deltas: dict[str, np.ndarray]
    
    def __init__(self, stitched_results: dict[str, tuple[np.ndarray, list[np.ndarray]]], bplt=True):
        """Initialize the Comparator with stitching results.
        
        Parameters
        ----------
        stitched_results : dict[str, tuple[ndarray, list[ndarray]]]
            Dictionary mapping method names to tuples of (stitched_surface, point_clouds_T)
            where stitched_surface is the final stitched point cloud and
            point_clouds_T is the list of individual (transformed) point clouds.
        names : sequence[str], optional
            Labels for each stitching method. If None, uses "Method 0", "Method 1", etc.
        """
        self.stitched_results = stitched_results
        self.deltas = {}

        self.compute()
        self.save_deltas()
        if bplt:
            self.print_summary()
            self.plot_deltas(noise_threshold=0.01)
            self.plot_histograms(noise_threshold=0.01)
            self.colormap_deltas(mode='xyz')
            plt.show()

    def compute(self):
        print("[Comparator] Base compute() method called. Override this method in subclasses to compute specific metrics.")
        pass

    def print_summary(self, precision=3):
        """
        Print a summary of delta statistics for all methods.
        For each method, prints the scipy stats describe of |$Delta$|, $Delta$x, $Delta$y, $Delta$z in a pretty format.
        """
        results = {}
        
        if self.deltas:
            print("\n" + "=" * 70)
            print("Ball Query Statistics Summary")
            print("=" * 70)
        for method_name, delta_data in self.deltas.items():
            
            results[method_name] = {
                "Component" :  [],
                "Count" : [],
                r"Mean [$\mu$m]" : [],
                r"StdDev [$\mu$m]" : [],
                r"Min [$\mu$m]" : [],
                r"Max [$\mu$m]" : []
            }

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

                results[method_name]["Component"].append(comp_label)
                results[method_name]["Count"].append(count)
                results[method_name][r"Mean [$\mu$m]"].append(f"{mean:.{precision}f}")
                results[method_name][r"StdDev [$\mu$m]"].append(f"{stddev:.{precision}f}")
                results[method_name][r"Min [$\mu$m]"].append(f"{minn:.{precision}f}")
                results[method_name][r"Max [$\mu$m]"].append(f"{maxx:.{precision}f}")

                # Dynamically build the format string based on your precision choice
                fmt = f"{{:>10s}} | {{:10d}} | {{:12.{precision}f}} | {{:12.{precision}f}} | {{:12.{precision}f}} | {{:12.{precision}f}}"
                
                # Inside your loop:
                print(fmt.format(comp_label, count, mean, stddev, minn, maxx))

                # print(f"{comp_label:>10s} | {count:10d} | {mean:12.6g} | {stddev:12.6g} | {minn:12.6g} | {maxx:12.6g}")


        return results

    def plot_deltas(self, noise_threshold: float, labels: Sequence[str] | None = None, figsize_scale: float = 5.0) -> plt.Figure:
        """Generate diagnostic comparison figure for stitching methods.
        
        Produces a figure with one row per stitching method and one row per
        pair of methods, showing |$Delta$|, $Delta$x, $Delta$y, $Delta$z with RMSE annotations.
        
        Parameters
        ----------
        noise_threshold : float
            Threshold value for RMSE annotation of outliers.
        labels : sequence[str], optional
            Labels for each stitching method. If None, it uses self.stitched_results.keys()
        figsize_scale : float, optional
            Scaling factor for figure size (default: 5.0).
        
        Returns
        -------
        Figure
            The matplotlib figure object, or None if deltas haven't been computed.
        """
        if labels is None: labels = list(self.stitched_results.keys())
        fig = _plot_comparison_figure(
            self.stitched_results,
            self.deltas,
            _plot_row,
            "Stitching algorithm comparison",
            noise_threshold,
            labels,
            figsize_scale,
            show_differences=False
        )
        return fig
    
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
            figsize_scale,
            show_differences=False
        )

    def colormap_deltas(self, mode='xyz'):
        """Visualize deltas using colormapped point clouds.
        
        Parameters
        ----------
        mode : str, optional
            Coloring mode:
            - 'modulus' (default): color by |$Delta$|
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
                [[splotter.get_rgb_from_3d_weights(self.deltas[method])] for method in self.stitched_results.keys()], names=[f"{method} - XYZ" for method in self.stitched_results.keys()]
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
    
    @staticmethod
    def _ensure_stitched_result_dict(stitched_results, names=None):
        if isinstance(stitched_results, dict):
            return stitched_results

        if names is None:
            raise ValueError("names must be provided when stitched_results is not a dict")

        if isinstance(stitched_results, list):
            if not isinstance(names, (list, tuple)):
                raise ValueError("names must be a list or tuple of the same length as stitched_results")
            if len(names) != len(stitched_results):
                raise ValueError("Length of names must match number of point clouds in stitched_results")
            return {
                name: (pc, [pc])
                for name, pc in zip(names, stitched_results)
            }

        if isinstance(names, (list, tuple)):
            if len(names) != 1:
                raise ValueError("names must be a single string or a single-element list/tuple for a single point cloud")
            name = names[0]
        else:
            name = names

        return {
            name: (stitched_results, [stitched_results])
        }
    
    def save_deltas(self):
        pass

    @staticmethod
    def load_ball_query(folder, bq_filename, bplt=True):
        """
        Loads pickled BallQuery results from a folder and re-hydrates the instance.
        
        Parameters
        ----------
        folder : str or Path
            Path to folder containing the pickled file.
        bq_filename : str
            Filename of the pickled BallQuery data (e.g., "bq_data.pkl").
        
        Returns
        -------
        BallQuery
            Re-hydrated BallQuery instance with deltas loaded.
        """
        path = pathlib.Path(folder)
        with open(path / bq_filename, "rb") as f:
            ball_query_results = pickle.load(f)
        
        ball_query = BallQuery.return_empty_instance()
        ball_query.deltas = ball_query_results["deltas"]
        ball_query.stitched_results = {key: None for key in ball_query_results["deltas"]}

        if bplt:
            summary = ball_query.print_summary()
            fig1 = ball_query.plot_deltas(noise_threshold=0.01)
            fig2 = ball_query.plot_histograms(noise_threshold=0.01)
            fig3 = ball_query.colormap_deltas()
            plt.show()
        return ball_query, summary, fig1, fig2, fig3
        

    @staticmethod
    def load_density_map_posterior(folder, dmp_filename, bplt=True):
        """
        Loads pickled DensityMapPosterior results from a folder and re-hydrates the instance.
        
        Parameters
        ----------
        folder : str or Path
            Path to folder containing the pickled file.
        dmp_filename : str
            Filename of the pickled DMP data (e.g., "dmp_data.pkl").
        
        Returns
        -------
        DensityMapPosterior
            Re-hydrated DensityMapPosterior instance with deltas and errors loaded.
        """
        path = pathlib.Path(folder)
        with open(path / dmp_filename, "rb") as f:
            dmp_results = pickle.load(f)
        
        dmp = DensityMapPosterior.return_empty_instance()
        dmp.deltas = dmp_results["deltas"]
        dmp.dmp_errors = dmp_results.get("errors") or {}
        dmp.stitched_results = {key: None for key in dmp_results["deltas"]}
        if bplt:
            summary = dmp.print_summary()
            fig1 = dmp.plot_deltas()
            fig2 = dmp.plot_histograms()
            plt.show()
        return dmp, summary, fig1, fig2
    

def make_compute_R(point_clouds_T):
    """Create a function that computes the neighbourhood radius for deltas computation.
    
    Calculates a dynamic radius based on the internal point density of the clouds
    and the typical gap (median offset) between consecutive point clouds.

    "Accurate 3D comparison of complex topography with terrestrial laser scanner: Application to the Rangitikei canyon (N-Z)",
    Lague et al., ISPRS Journal of Photogrammetry and Remote Sensing, 2013.
    https://doi.org/10.1016/j.isprsjprs.2013.04.009
    """
    R_vals = []
    g_vals = []

    for i in range(len(point_clouds_T) - 1):
        fixed_pts = point_clouds_T[i]
        aligned = point_clouds_T[i + 1]

        fppc = sutils.pcd_to_o3d_pcd(fixed_pts)
        apc = sutils.pcd_to_o3d_pcd(aligned)

        internal_distances = np.asarray(fppc.compute_nearest_neighbor_distance())
        cloud_resolution = np.mean(internal_distances)

        r_patch = cloud_resolution * 5.0 
        R_vals.append(r_patch)

        distances = np.asarray(apc.compute_point_cloud_distance(fppc))
        g = np.percentile(distances, 50) 
        g_vals.append(g)

        if False:
            fig, ax = plt.subplots()

            ax.hist(distances, bins='auto', alpha=0.5, label="fixed → moving")
            ax.vlines([g], 0, plt.ylim()[1], colors='r', linestyles='dashed', label='Median Gap (g)')
            ax.vlines([R], 0, plt.ylim()[1], colors='g', linewidth=2, label=f'Final Radius R ({R:.4f})')

            ax.set_xlabel("Distance")
            ax.set_ylabel("Count")
            ax.set_title(f"Distance distribution (Pair {i} to {i+1})")
            
            plt.legend()
            plt.show()

    R_value = float(np.mean(R_vals))
    g_value = float(np.mean(g_vals))

    R = np.sqrt(g_value**2 + R_value**2)
    print(f"Computed neighbourhood radius R = {R_value:.6f}")
    print(f"Based on patch radius = {r_patch:.6f} and typical gap = {g_value:.6f}")

    def compute_R(*args, **kwargs):
        return R

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
    tree = cKDTree(stitched)

    deltas = np.empty((n, 3), dtype=float)
    R = float(compute_R())
    
    idx = tree.query_ball_point(stitched, r=R) # This operation can be slow for large point clouds
    # _, idx = tree.query(stitched, k=30)
    
    for i, (point, neighbours) in tqdm(enumerate(zip(stitched, idx)), total=n, desc="Computing deltas", colour='cyan'):            
        neighbourhood = stitched[neighbours]  # always contains point itself
        mean_vec = neighbourhood.mean(axis=0)
        deltas[i] = point - mean_vec

    print("\n")
    return deltas

class BallQuery(Comparator):
    """
    Evaluates stitching quality via the ball-query local-mean residual method.

    For every point in the stitched surface, the residual is defined as the
    vector from the point to the centroid of all neighbours within radius R.
    A well-stitched surface should produce small, zero-mean residuals.

    Parameters
        ----------
        save_path : str
            Filesystem path where the computed deltas will be pickled.
        stitched_results : dict
            Maps method name -> (stitched_surface, list_of_transformed_clouds).
        bplt : bool
            If True, the parent Comparator will call print_summary, plot_deltas,
            plot_histograms, and colormap_deltas immediately after construction.
            Keep False when the caller controls the plotting flow explicitly.

    """
    def __init__(
            self, 
            save_path: str, 
            stitched_results: dict[str, tuple[np.ndarray, list[np.ndarray]]],
            bplt: bool = False):
        self.save_path = save_path
        super().__init__(stitched_results, bplt=bplt)

    @staticmethod
    def return_empty_instance():
        return BallQuery.__new__(BallQuery)

    def compute(self):
        """Compute local-mean residuals for all stitching methods.
        
        Calculates deltas for each stitched result and stores them in self.deltas.
        Must be called before plotting or colormapping.
        """
        for method_name, restuple in self.stitched_results.items():
            stitched, stitched_T = restuple
            cR = make_compute_R(stitched_T)
            print(f"[INFO COMPARATOR] Starting KDTree query_ball_point execution for method {method_name}")
            self.deltas[method_name] = compute_deltas(stitched, cR)

    def save_deltas(self):
        if not self.save_path is None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "wb") as f:
                pickle.dump({
                    "deltas": self.deltas,
                }, f)

 
def compute_distances_T(
    point_clouds_T: list[np.ndarray],
    distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> list[np.ndarray]:
    """
    Given a list of transformed point clouds, compute the distances from each point in each cloud
    using the provided distance function (or a default nearest-neighbour distance if None) and return
    
    Parameters
    ----------
    point_clouds_T : list[ndarray]
        List of transformed point clouds, where each element is an (N, 3) array.
    distance_function : callable, optional
        Function that takes ``fixed_points`` and ``moving_points`` and returns
        either an array of distances or an array of difference vectors.
    
    Returns
    -------
    list[np.ndarray]
        List of distance arrays for each point cloud
    """
    point_clouds_T = [np.asarray(pc, dtype=float) for pc in point_clouds_T]
    n_clouds = len(point_clouds_T)
    
    if n_clouds < 2:
        raise ValueError("At least 2 point clouds are required for DMP evaluation")
    
    distance_T = []
    
    # For each point cloud
    for i in tqdm(range(n_clouds), desc="Computing DMP distances", colour='magenta'):
        current_cloud = point_clouds_T[i]
        
        # Combine all other point clouds
        other_clouds = np.vstack([point_clouds_T[j] for j in range(n_clouds) if j != i])
        
        distances = distance_function(current_cloud, other_clouds)
        distances = np.asarray(distances, dtype=float)
        if distances.ndim == 2 and distances.shape[1] == 3:
            distances = np.linalg.norm(distances, axis=1)
        elif distances.ndim != 1:
            raise ValueError(
                "distance_function must return either a 1D distance array or an (N, 3) array of difference vectors"
            )
        
        if distances.size == 0:
            raise ValueError("distance_function returned an empty distance array")
        
        distance_T.append(distances)

    return distance_T

class DensityMapPosterior():
    """
    A class to evaluate stitching quality using the Dense Map Posterior method.
    """
    def __init__(self, save_path: str, stitched_results: dict[str, tuple[np.ndarray, list[np.ndarray]]], 
                 distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, bplt=False): 
        
        self.save_path = save_path
        self.bplt = bplt

        if distance_function is None:
            self.distance_function = sstitcher.KDTree_mutual_diffs
        else:
            self.distance_function = distance_function
        
        self.stitched_results = stitched_results
        self.deltas = {}

        self.compute()

        self.dmp_errors = {}
        self.compute_all_dmp_errors()
        
        self.save_deltas()

    @staticmethod
    def return_empty_instance():
        return DensityMapPosterior.__new__(DensityMapPosterior)

    def compute(self) -> None:
        """
        Compute DMP distances for all stitching methods and store in self.deltas.
        
        Parameters
        ----------
        distance_function : callable, optional
            Function that takes ``fixed_points`` and ``moving_points`` and
            returns either an array of distances or an array of difference
            vectors. If ``None``, a standard KDTree nearest-neighbour distance
            is used.
        KDTreeMutual : bool, optional
            If True and ``distance_function`` is None, uses
            ``sstitcher.KDTree_mutual_diffs`` to compute only mutual nearest
            neighbour differences. This parameter is kept for compatibility.
        """
        print("=" * 70)
        print("Computing DMP distances for all stitching methods")
        print("=" * 70)
        
        for method_name, (stitched, point_clouds_T) in self.stitched_results.items():
            print(f"\n--- Method: {method_name} ---")
            distances = compute_distances_T(point_clouds_T, distance_function=self.distance_function)
            self.deltas[method_name] = distances
    
    def compute_all_dmp_errors(self) -> None:
        """
        Compute DMP errors for all stitching methods.
        
        Results are stored in self.dmp_errors as a dictionary mapping method names
        to error dictionaries containing 'total_error', 'per_cloud_errors', and
        'mean_error_per_cloud'.
        """
        self.dmp_errors = {}

        for method_name, distance_list in self.deltas.items():
            mean_total_error = 0.0
            squared_total_error = 0.0
            per_cloud_mean_errors = []
            per_cloud_squared_errors = []
            n_clouds = len(distance_list)

            for i, distances in enumerate(distance_list):
                distances = np.asarray(distances, dtype=float)
                if distances.ndim == 0:
                    distances = distances.reshape(1)
                elif distances.ndim == 2 and distances.shape[1] == 3:
                    distances = np.linalg.norm(distances, axis=1)
                elif distances.ndim != 1:
                    raise ValueError(
                        "Stored distances must be a 1D distance array or an (N, 3) array of difference vectors"
                    )

                mean_error = np.mean(distances)
                squared_error = float(np.sum(distances ** 2))

                per_cloud_mean_errors.append(mean_error)
                per_cloud_squared_errors.append(squared_error / len(distances))
                mean_total_error += mean_error   # $\mu$m
                squared_total_error += squared_error / len(distances) # $\mu$m^2

            mean_error_per_cloud = np.mean(per_cloud_mean_errors)
            squared_error_per_cloud = np.mean(per_cloud_squared_errors)

            self.dmp_errors[method_name] = {
                r'total_error [$\mu$m]': float(mean_total_error / n_clouds),  # $\mu$m
                r'per_cloud_errors [$\mu$m]': per_cloud_mean_errors,  # $\mu$m
                r'per_cloud_squared_errors [$\mu$m^2]': per_cloud_squared_errors, # $\mu$m^2
                # r'mean_error_per_cloud [$\mu$m]': float(mean_error_per_cloud),# $\mu$m  equivale a total error
                r'squared_error_per_cloud [$\mu$m^2]': float(squared_error_per_cloud), # $\mu$m^2
                r'DMP metric [$\mu$m^2]': float(squared_total_error) # $\mu$m^2
            }

    def save_deltas(self):
        if not self.save_path is None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "wb") as f:
                pickle.dump({
                    "deltas": self.deltas,
                    "errors": self.dmp_errors if hasattr(self, 'dmp_errors') else None,
                }, f)
    
    def print_summary(self, precision=3) -> None:
        """Print a summary of DMP errors for all methods."""
        if not self.dmp_errors:
            print("[WARN DMP] No errors computed yet. Run compute_all_dmp_errors() first.")
            return

        results = {
            "general_data" : {
                " " : []
            },
            "per_cloud_data" : {
                " " : []
            }
        }

        print("\n" + "=" * 70)
        print("DMP Evaluation Summary")
        print("=" * 70)
        
        # Temporary debug line to see the exact keys
        # for method, errors in self.dmp_errors.items():
        #     print(f"Method {method} has keys: {list(errors.keys())}")

        # Sort methods by total error for ranking
        sorted_methods = sorted(
            self.dmp_errors.items(),
            key=lambda x: x[1][r'total_error [$\mu$m]']
        )
        
        print("\nRanking by total DMP error (lower is better):")
        for rank, (method_name, errors) in enumerate(sorted_methods, 1):
            summary_items = []
            per_cloud_errors = None
            
            results["per_cloud_data"][" "].append(method_name.replace('_', '\\_'))

            for key, value in errors.items():
                if key == r'per_cloud_errors [$\mu$m]':  
                    per_cloud_errors = value

                    for i, error in enumerate(per_cloud_errors):
                        results["per_cloud_data"].setdefault(f"Cloud {i}", []).append(error)

                elif key == r'per_cloud_squared_errors [$\mu$m^2]':
                    pass
                else:
                    if isinstance(value, float):
                        print_label = key.split(' [')[0].replace('_', ' ').title()
                        summary_items.append(f"{print_label}: {value:.{precision}f}")

                        dict_key = key.replace('_', '\\_')
                        results["general_data"].setdefault(dict_key, []).append(value)        


            results["general_data"][" "].append(method_name.replace('_', '\\_')) 

            print(f"  {rank}. {method_name:<25s} | {' | '.join(summary_items)}")

            # Print per-cloud errors in columns below
            if per_cloud_errors:
                print(f"      Per-cloud errors:")
                for i, error in enumerate(per_cloud_errors):
                    print(f"        Cloud {i}: {error:.{precision}f}")

        return results

    def plot_deltas(self, method_name: str | None = None, figsize: tuple[int, int] = (8, 4), ax: plt.Axes | None = None) -> plt.Figure | None:
        """
        Plot per-cloud DMP squared errors for one or all methods.

        X axis: point cloud index `i` (i.e. point_clouds_T[i]).
        Y axis: per-cloud squared DMP error (the contributions used to form the
        'DMP metric (squared_total_error)' for each method).

        If `method_name` is provided, only that method is plotted. Otherwise
        every computed method is plotted on the same axes for comparison.
        Each series is connected with lines and has circle markers on every point.
        """
        if not self.dmp_errors:
            print("[WARN DMP] No errors computed yet. Run compute_all_dmp_errors() first.")
            return None

        if method_name is not None and method_name not in self.dmp_errors:
            print(f"[WARN DMP] Method '{method_name}' not found in computed errors.")
            return None

        methods = [method_name] if method_name else list(self.dmp_errors.keys())

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # collect tick range across all methods to produce consistent M_i labels
        lengths = []
        for method in methods:
            errors = self.dmp_errors[method].get('per_cloud_squared_errors [$\mu$m^2]', [])
            if not errors:
                print(f"[WARN DMP] No per-cloud squared errors for method '{method}'. Skipping.")
                continue

            y = np.asarray(errors, dtype=float)
            x = np.arange(len(y))

            total = self.dmp_errors[method].get('DMP metric [$\mu$m^2]', None)
            label = method if total is None else f"{method} (total={total:.6g})"

            ax.plot(x, y, marker='o', linestyle='None', markersize=6, label=label)
            lengths.append(len(y))

        if not lengths:
            print("[WARN DMP] No valid per-cloud errors found to plot.")
            return fig

        max_n = max(lengths)
        xticks = np.arange(max_n)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"$M_{i}$" for i in xticks])

        ax.set_xlabel("point cloud ($M_i$)")
        ax.set_ylabel("per-cloud squared DMP error [$\mu m^2$]")
        title_suffix = f" - {method_name}" if method_name else ""
        ax.set_title("DMP per-cloud contributions" + title_suffix)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

        return fig

    def plot_histograms(
        self,
        method_name: str | None = None,
        bins: int | str = 'auto',
        figsize: tuple[int, int] | None = None,
    ) -> dict[str, plt.Figure] | None:
        """
        Plot histograms for stored DMP distances per point cloud.

        If `method_name` is None, produces one figure per method in
        ``self.deltas`` (each figure contains m histograms for m point
        clouds). Returns a dict mapping method->Figure.
        """
        if not self.deltas:
            print("[WARN DMP] No distances computed yet. Run compute_all_deltas() first.")
            return None

        methods = [method_name] if method_name else list(self.deltas.keys())
        figs: dict[str, plt.Figure] = {}

        for method in methods:
            if method not in self.deltas:
                print(f"[WARN DMP] Method '{method}' not found. Skipping.")
                continue

            distance_list = self.deltas[method]
            m = len(distance_list)
            if m == 0:
                print(f"[WARN DMP] No distance arrays for method '{method}'. Skipping.")
                continue

            # layout: try to make a grid close to square
            cols = int(np.ceil(np.sqrt(m)))
            rows = int(np.ceil(m / cols))
            if figsize is None:
                fig_w = max(6, cols * 3)
                fig_h = max(3, rows * 2.5)
                fig = plt.figure(figsize=(fig_w, fig_h))
            else:
                fig = plt.figure(figsize=figsize)

            # --- Axis Sharing Setup (Per Figure) ---
            share_x_ax = None
            share_y_ax = None                

            for i, distances in enumerate(distance_list):
                d = np.asarray(distances, dtype=float)
                if d.ndim == 2 and d.shape[1] == 3:
                    d = np.linalg.norm(d, axis=1)

                # Create the subplot sharing axes with our reference subplots
                ax = fig.add_subplot(rows, cols, i + 1, sharex=share_x_ax, sharey=share_y_ax)
                
                # Establish the reference axes using the very first subplot of this figure
                if share_x_ax is None:
                    share_x_ax = ax
                    share_y_ax = ax

                if d.size == 0:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center')
                else:
                    ax.hist(d, bins=bins, color='C0', alpha=0.85)
                    
                ax.set_title(f"{method} — $M_{i}$", fontsize=8)
                ax.set_xlabel('distance')
                ax.set_ylabel('count')
                ax.grid(True, linestyle='--', alpha=0.4)

            fig.suptitle(f"DMP distance histograms — {method}")
            fig.tight_layout()    # helps prevent overlapping titles/labels
            figs[method] = fig

        return figs


class CAD(Comparator):
    """
    Comparison class between stitched_results and a cad file (pointcloud of mesh)

    allows for:
    - compute_deltas_cad(stitched, cad, compute_R)
        For every point in a stitched surface, find the local neighbourhood
        inside a radius R, compute the mean of that neighbourhood, and return
        the vector difference between the point and that mean.
    """
    def __init__(self, 
                 stitched_results: dict[str, tuple[np.ndarray, list[np.ndarray]]],
                 cad_points: np.ndarray, 
                 pipeline: spipe.TreePipeline, 
                 pipeline_path: str, 
                 names=None,
                 bplt: bool = False
                 ):
        self.cad_points = cad_points
        self.pipeline = pipeline
        self.pipeline_path = pipeline_path

        # self.stitched_aligned_to_cad = {}
        self.aligned_cad_per_method: dict[str, np.ndarray] = {}

        super().__init__(self._ensure_stitched_result_dict(stitched_results, names), bplt=bplt)

    def align_cad_to_stitched(self, bplt: bool = False):
        """
        Align the CAD point cloud to the stitched surface using a registration pipeline
        """
        base_pipe_name = self.pipeline.name

        for method_name, restuple in self.stitched_results.items():
            stitched, _ = restuple
            self.pipeline.name = f'{method_name}_{base_pipe_name}'
            aligned_cad = self.pipeline.run([self.cad_points, stitched], save_transforms_root=self.pipeline_path, bplt=bplt)

            if not aligned_cad:
                print(f"[WARN CAD] First run of pipeline {self.pipeline.name} rerun with saved pipe to continue")
                continue
            
            self.stitched_aligned_to_cad[method_name] = aligned_cad
        
    def compute(self):
        self.align_cad_to_stitched(bplt=False)

        for key in self.stitched_aligned_to_cad:
            bq = BallQuery(stitched_results=self.stitched_aligned_to_cad[key])
            bq.plot_deltas(noise_threshold=0.5)
            bq.plot_histograms(noise_threshold=0.5)
            bq.colormap_deltas(mode='xyz')
            bq.print_summary()
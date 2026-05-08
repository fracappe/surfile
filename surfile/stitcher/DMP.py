"""
DMP.py
======
Dense Map Posterior (DMP) method for evaluating stitching quality.

Public API
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

from typing import Callable, Sequence
import numpy as np
from scipy.spatial import KDTree


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
    
    total_error = 0.0
    per_cloud_errors = []
    
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
        per_cloud_errors.append(mean_error)
        total_error += mean_error / len(current_cloud)  # Normalize by number of points in cloud
        
        print(f"Point cloud {i}: error = {mean_error:.6f}, mean squared distance = {mean_error / len(current_cloud):.6f}")
    
    mean_error_per_cloud = np.mean(per_cloud_errors)
    
    print(f"\nTotal DMP error: {total_error:.6f}")
    print(f"Mean error per cloud: {mean_error_per_cloud:.6f}")
    
    return {
        'total_error': float(total_error),
        'per_cloud_errors': per_cloud_errors,
        'mean_error_per_cloud': float(mean_error_per_cloud),
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
            total_error = errors['total_error']
            mean_per_cloud = errors['mean_error_per_cloud']
            n_clouds = len(errors['per_cloud_errors'])
            print(f"  {rank}. {method_name:<25s} | Total: {total_error:12.6f} | "
                  f"Mean/cloud: {mean_per_cloud:10.6f} | N_clouds: {n_clouds}")
    
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

"""
'surfile.stitcher'
- implementation of surface stitching methods

@author: Andrea Giura
"""
from surfile import funct
import surfile.stitcher.utils as sutils
import surfile.stitcher.plotter as splotter

import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d
from skopt import gp_minimize
from skopt.space import Real
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation

import copy
import multiprocessing as mp
import os
import pickle
from pathlib import Path

class TransformParams:
    """
    Manages a 6-DOF rigid body transformation.

    This class encapsulates a 3D transformation, defined by three translation
    components (tx, ty, tz) and three rotation components (rx, ry, rz) as
    Euler angles in an 'xyz' sequence.

    It provides class methods to instantiate from various data formats (e.g.,
    lists, files, Kabsch algorithm output) and methods to convert the
    parameters into a 4x4 homogeneous transformation matrix.

    Attributes
    ----------
    tx, ty, tz : float
        Translation along the x, y, and z axes.
    rx, ry, rz : float
        Rotation around the x, y, and z axes in degrees.

    """
    rx: float
    ry: float
    rz: float
    tx: float
    ty: float
    tz: float

    def __init__(self):
        """Initializes an empty TransformParams object."""
        pass

    @classmethod
    def from_numbers(cls, tx=0, ty=0, tz=0, rx=0, ry=0, rz=0):
        """
        Create a TransformParams instance from individual numbers.

        Parameters
        ----------
        tx, ty, tz : float, optional
            Translation values. Defaults to 0.
        rx, ry, rz : float, optional
            Rotation values in degrees. Defaults to 0.

        Returns
        -------
        TransformParams
            A new instance of the class.
        """
        instance = cls()
        instance.rx, instance.ry, instance.rz, instance.tx, instance.ty, instance.tz = rx, ry, rz, tx, ty, tz
        return instance

    @classmethod
    def from_list(cls, params: list[float]):
        """
        Create a TransformParams instance from a list.

        Parameters
        ----------
        params : list[float]
            A list of 6 floats in the order [tx, ty, tz, rx, ry, rz].

        Returns
        -------
        TransformParams
            A new instance of the class.
        """
        instance = cls()
        instance.tx, instance.ty, instance.tz, instance.rx, instance.ry, instance.rz = params
        return instance

    @classmethod
    def from_tuples(cls, rot: R, trasl: np.ndarray):
        """
        Create a TransformParams instance from a SciPy Rotation object and a
        translation vector.

        Parameters
        ----------
        rot : scipy.spatial.transform.Rotation
            The rotation object.
        trasl : np.ndarray
            The translation vector of shape (3,).

        Returns
        -------
        TransformParams
            A new instance of the class.
        """
        instance = cls()
        eul = rot.as_euler('xyz')
        instance.rx, instance.ry, instance.rz = eul[0], eul[1], eul[2]
        instance.tx, instance.ty, instance.tz = trasl[0], trasl[1], trasl[2]
        return instance

    @classmethod
    def from_file(cls, filename, tr_n, header=1):
        """
        Create a TransformParams instance from a specific line in a text file.

        The file is expected to contain transformation data where each line
        represents a transformation.

        Parameters
        ----------
        filename : str
            Path to the text file.
        tr_n : int
            The transformation number (line index) to read, starting from 0
            after the header.
        header : int, optional
            Number of header lines to skip. Defaults to 1.

        Returns
        -------
        TransformParams
            A new instance of the class.
        """
        instance = cls()
        with open(filename, "r") as f:
            riga = list(f)[header + tr_n]
        riga = riga[3:].strip().replace(',', ' ')
        values = np.array(list(map(float, riga.split())))
        instance.tx, instance.ty, instance.tz, instance.rx, instance.ry, instance.rz = values
        return instance

    @classmethod
    def from_kabsch(cls, m: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the least-squares rigid transformation that maps points m to f.

        This method implements the Kabsch algorithm (1976) to find the optimal
        rotation and translation that minimizes the Root Mean Square Deviation
        (RMSD) between two sets of corresponding points.

        Parameters
        ----------
        m : np.ndarray
            The moving points, as an (N, 3) array.
        f : np.ndarray
            The fixed (target) points, as an (N, 3) array.

        Returns
        -------
        TransformParams
            A new instance of the class representing the optimal transformation.

        Notes
        -----
        The algorithm performs the following steps:
        1.  Subtract the centroids from both point sets.
        2.  Compute the cross-covariance matrix $$H = m_{centered}^T \cdot f_{centered}$$.
        3.  Perform Singular Value Decomposition (SVD) on H.
        4.  Calculate the optimal rotation matrix $$R = V \cdot D \cdot U^T$$, where D is a
            correction matrix to ensure a proper rotation (determinant = +1).
        5.  Calculate the optimal translation vector $$t = c_f - R \cdot c_m$$, where
            $c_f$ and $c_m$ are the centroids of the fixed and moving points.
        """
        instance = cls()

        m = np.asarray(m, dtype=float)
        f = np.asarray(f, dtype=float)
        assert m.shape == f.shape and m.ndim == 2 and m.shape[1] == 3

        # 1. Centroid subtraction
        cm = m.mean(axis=0)
        cf = f.mean(axis=0)
        m_c = m - cm
        f_c = f - cf

        # 2. Cross-covariance matrix  H = m_centred^T · f_centred
        H = m_c.T @ f_c          # (3, 3)

        # 3. SVD
        U, S, Vt = np.linalg.svd(H)

        # 4. Correct for reflection (det check ensures a proper rotation, det = +1)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])   # d = sign(det); flips last singular vector if needed

        # 5. Rotation and translation
        R_m = Vt.T @ D @ U.T

        t = cf - R_m @ cm

        # 6. Homogeneous 4×4 matrix
        T = np.eye(4)
        T[:3, :3] = R_m
        T[:3,  3] = t

        # from rotation matrix to angles
        r = R.from_matrix(R_m)
        instance.rx, instance.ry, instance.rz  = r.as_euler('xyz', degrees=True)
        instance.tx, instance.ty, instance.tz = t

        return instance

    @classmethod
    def from_pickle(cls, filepath: str):
        """
        Load a TransformParams instance from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the .pkl file.

        Returns
        -------
        TransformParams
            The loaded instance.
        """
        instance = cls()
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        return instance

    def to_pickle(self,  filepath: str):
        """
        Save the current TransformParams instance to a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the destination .pkl file. The directory will be created
            if it does not exist.
        """
        filepath: Path = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
            print(f'[INFO TRANSFORM PICKLE] Saved {filepath.name}')

    def get_params(self):
        """
        Get the transformation parameters as a list.

        Returns
        -------
        list[float]
            A list of 6 floats in the order [rx, ry, rz, tx, ty, tz].
        """
        return [self.rx, self.ry, self.rz, self.tx, self.ty, self.tz]

    def get_matrix(self):
        """
        Compute and return the 4x4 homogeneous transformation matrix.

        Returns
        -------
        np.ndarray
            The 4x4 transformation matrix.
        """
        Rmat = R.from_euler('xyz', np.radians([self.rx, self.ry, self.rz])).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = [self.tx, self.ty, self.tz]
        return T

    def rescale(self, factor):
        """
        Scale the translation components by a given factor.

        This is useful for converting units (e.g., from meters to millimeters).
        Note that rotation is unaffected.

        Parameters
        ----------
        factor : float
            The scaling factor to apply to tx, ty, and tz.
        """
        self.tx *= factor
        self.ty *= factor
        self.tz *= factor

    def __str__(self):
        """Return a string representation of the transformation matrix."""
        return f"TransformParams: {self.get_matrix()}"

@sutils.ensure_numpy_pcd
def apply_transform(points: np.ndarray, params: TransformParams | np.ndarray, params0: TransformParams | np.ndarray=None):
    """
    Apply a rigid transformation to a set of points.

    Parameters
    ----------
    points : np.ndarray
        The (N, 3) array of points to transform.
    params : TransformParams or np.ndarray
        The transformation to apply. Can be a `TransformParams` object or a
        4x4 homogeneous transformation matrix.
    params0 : TransformParams or np.ndarray, optional
        An initial transformation to make `params` relative to. If provided,
        the effective transformation applied is `inv(params0) @ params`.
        This is useful for aligning all point clouds to the coordinate system
        of the first one. Defaults to None.

    Returns
    -------
    np.ndarray
        The transformed (N, 3) array of points.
    """
    if hasattr(params, 'get_matrix'):
        T = params.get_matrix()
    else: T = params

    if params0 is not None:
        if hasattr(params0, 'get_matrix'):
            T0 = params0.get_matrix()
        else: T0 = params0

        T0_inv = np.linalg.inv(T0)
        
        T = T0_inv @ T

    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (T @ pts_h.T).T[:, :3]


def KDTree_mutual_diffs(fixed_points, moving_points):
    """
    Calculate the difference vectors between mutual nearest neighbors.

    This function finds pairs of points (one from `fixed_points`, one from
    `moving_points`) that are each other's closest neighbor. It then returns
    the difference vectors for these mutual pairs.

    Parameters
    ----------
    fixed_points : np.ndarray
        The (N, 3) fixed point cloud.
    moving_points : np.ndarray
        The (M, 3) moving point cloud.

    Returns
    -------
    np.ndarray
        An (K, 3) array of difference vectors for the K mutual pairs found.
        Returns `float('inf')` if no mutual pairs are found.
    """
    fixed_tree = cKDTree(fixed_points)
    moving_tree = cKDTree(moving_points)

    dist_f2m, idx_f2m = moving_tree.query(fixed_points, k=1, workers= -1)
    dist_m2f, idx_m2f = fixed_tree.query(moving_points, k=1, workers= -1)

    selected_points = (np.arange(len(fixed_points)) == idx_m2f[idx_f2m])
    if not np.any(selected_points):
        return float('inf')

    diffs = fixed_points[selected_points] - moving_points[idx_f2m[selected_points]]
    return diffs


class Isolator():
    """
    A factory class for isolating overlapping regions between two point clouds.

    This class provides different strategies to select subsets of points that
    are likely to be in the overlapping area of two point clouds. This is a
    crucial pre-processing step for many registration algorithms, as it
    focuses the algorithm on the relevant data and improves robustness.

    Attributes
    ----------
    geometrical : str
        Identifier for the geometrical isolation method.
    maxmin : str
        Identifier for the max-min bounding box isolation method.
    KDTree : str
        Identifier for the KD-Tree distance-based isolation method.
    manual : str
        Identifier for the manual point selection isolation method.
    type : str
        The selected isolation strategy.
    stitchprc : int
        Parameter for the 'geometrical' method.
    max_distance : float or None
        Parameter for the 'KDTree' method.
    axes : str
        Parameter for the 'maxmin' method.

    """
    geometrical: str = 'geometrical'
    maxmin: str = 'maxmin'
    KDTree: str = 'KDTree'
    manual: str = 'manual'

    type: str

    def __init__(self, type: str, stitchprc=80, max_distance=None, axes='xyz'):
        """
        Initializes the Isolator with a specific strategy and its parameters.

        Parameters
        ----------
        type : str
            The isolation strategy to use. Must be one of 'geometrical',
            'maxmin', 'KDTree', 'manual', or 'convex_hull'.
        stitchprc : int, optional
            Percentage of overlap for the 'geometrical' method. Defaults to 80.
        max_distance : float or None, optional
            Maximum distance threshold for the 'KDTree' method. If None, it is
            estimated automatically. Defaults to None.
        axes : str, optional
            Axes to consider for the 'maxmin' bounding box. Defaults to 'xyz'.
        """
        self.type = type
        self.stitchprc = stitchprc
        self.max_distance = max_distance
        self.axes = axes

    def apply_isolator(self, fixed_pts: np.ndarray, moving_pts: np.ndarray, bplt=False):
        if self.type == 'geometrical': return self.isolate_common_points_geometrical(fixed_pts, moving_pts, self.stitchprc, bplt=bplt)
        elif self.type == 'maxmin': return self.isolate_common_points_max_min(fixed_pts, moving_pts, self.axes, bplt=bplt)
        elif self.type == 'KDTree': return self.isolate_common_points_kdtree(fixed_pts, moving_pts, self.max_distance, bplt=bplt)
        elif self.type == 'manual': return self.isolate_manual(fixed_pts, moving_pts)
        elif self.type == 'convex_hull': return self.isolate_convex_hull(fixed_pts, moving_pts, bplt=bplt)

        else:
            raise ValueError('Unknown isolator type')
        
    @staticmethod
    def plot_colored_distances(fixed_pts, moving_pts, log=False, cmap=plt.cm.plasma):
        """Helper function to visualize the distance-based isolation."""
        dist_f2m, _ = cKDTree(moving_pts).query(fixed_pts, k=1)
        dist_m2f, _ = cKDTree(fixed_pts).query(moving_pts, k=1)

        fixed_colors = splotter.get_colors_from_weights(cmap, dist_f2m, log=log)
        moving_colors = splotter.get_colors_from_weights(cmap, dist_m2f, log=log)
        splotter.show_point_clouds([fixed_pts, moving_pts], colors=[fixed_colors, moving_colors])

    @staticmethod
    def plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts):
        """Helper function to visualize the isolated point cloud subsets."""
        splotter.show_point_clouds([fixed_subset, moving_subset], colors='uniform')

    @staticmethod
    def isolate_common_points_geometrical(fixed_pts: np.ndarray, moving_pts: np.ndarray, stitchprc=80, bplt=False):
        """
        Isolate points based on their projection along the vector connecting
        the point cloud centers.

        This method assumes the primary direction of movement between the two
        scans is along the line connecting their centroids. It keeps only the
        points in the `stitchprc` percentage of each cloud closest to the other.

        Parameters
        ----------
        fixed_pts : np.ndarray
            The (N, 3) fixed point cloud.
        moving_pts : np.ndarray
            The (M, 3) moving point cloud.
        stitchprc : int, optional
            The percentage of the point cloud to consider as overlapping,
            measured from the edge closest to the other cloud. Defaults to 80.
        bplt : bool, optional
            If True, visualize the isolated subsets. Defaults to False.

        Returns
        -------
        fixed_subset : np.ndarray
            The isolated subset of the fixed point cloud.
        moving_subset : np.ndarray
            The isolated subset of the moving point cloud.
        """
        fixed_center = np.mean(fixed_pts, axis=0)
        moving_center = np.mean(moving_pts, axis=0)

        # direzione movimento
        moving_dir = moving_center - fixed_center
        norm = np.linalg.norm(moving_dir)
        if norm == 0:
            moving_dir = np.array([1.0, 0.0, 0.0])
        else:
            moving_dir /= norm

        # filtra punti entro stitchprc
        dist_fixed = (fixed_pts - fixed_center) @ moving_dir
        fixed_subset = fixed_pts[dist_fixed >= norm * (1 - stitchprc / 100)]

        dist_moving = (moving_pts - moving_center) @ moving_dir
        moving_subset = moving_pts[norm * (1 - stitchprc / 100) <= -dist_moving]

        if bplt: 
            Isolator.plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts)
            Isolator.plot_colored_distances(fixed_subset, moving_subset, log=True)

        return fixed_subset, moving_subset

    @staticmethod
    def isolate_common_points_max_min(fixed_pts: np.ndarray, moving_pts: np.ndarray, axes: str, bplt=False):
        """
        Isolate points based on the intersection of their bounding boxes.

        This method calculates the common volume shared by the axis-aligned
        bounding boxes of the two point clouds and returns all points that
        fall within this volume.

        Parameters
        ----------
        fixed_pts : np.ndarray
            The (N, 3) fixed point cloud.
        moving_pts : np.ndarray
            The (M, 3) moving point cloud.
        axes : str
            A string containing the axes to consider for intersection, e.g.,
            'xy', 'z'.
        bplt : bool, optional
            If True, visualize the isolated subsets. Defaults to False.

        Returns
        -------
        fixed_subset : np.ndarray
            The isolated subset of the fixed point cloud.
        moving_subset : np.ndarray
            The isolated subset of the moving point cloud.
        """
        def get_common_boundries(pts_a, pts_b):
            min_a = np.min(pts_a, axis=0)
            max_a = np.max(pts_a, axis=0)

            min_b = np.min(pts_b, axis=0)
            max_b = np.max(pts_b, axis=0)

            x_common = (max(min_a[0], min_b[0]), min(max_a[0], max_b[0]))
            y_common = (max(min_a[1], min_b[1]), min(max_a[1], max_b[1]))
            z_common = (max(min_a[2], min_b[2]), min(max_a[2], max_b[2]))

            return x_common, y_common, z_common

        x_mM, y_mM, z_mM = get_common_boundries(fixed_pts, moving_pts)

        if 'x' not in axes: x_mM = [-np.inf, +np.inf]
        if 'y' not in axes: y_mM = [-np.inf, +np.inf]
        if 'z' not in axes: z_mM = [-np.inf, +np.inf]

        make_selected_points = lambda pts: (
            (pts[:, 0] >= x_mM[0]) & (pts[:, 0] <= x_mM[1]) &
            (pts[:, 1] >= y_mM[0]) & (pts[:, 1] <= y_mM[1]) &
            (pts[:, 2] >= z_mM[0]) & (pts[:, 2] <= z_mM[1])
        )

        fixed_subset = fixed_pts[make_selected_points(fixed_pts)]
        moving_subset = moving_pts[make_selected_points(moving_pts)]

        if bplt: 
            Isolator.plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts)
            Isolator.plot_colored_distances(fixed_subset, moving_subset, log=True)
        return fixed_subset, moving_subset

    @staticmethod
    def isolate_common_points_kdtree(fixed_pts: np.ndarray, moving_pts: np.ndarray, max_distance: float=None, bins_after_max=1, bplt=False):
        """
        Isolate points based on nearest neighbor distances.

        This method keeps points from each cloud that are "close" to the other
        cloud. For each point in cloud A, it finds its nearest neighbor in
        cloud B. If the distance is below `max_distance`, the point is kept.

        If `max_distance` is not provided, it is estimated from the histogram
        of all nearest-neighbor distances.

        Parameters
        ----------
        fixed_pts : np.ndarray
            The (N, 3) fixed point cloud.
        moving_pts : np.ndarray
            The (M, 3) moving point cloud.
        max_distance : float or None, optional
            The distance threshold. Points further than this from the other
            cloud are discarded. If None, it's estimated. Defaults to None.
        bins_after_max : int, optional
            When estimating `max_distance`, this is the number of bins after
            the histogram's peak to set the threshold. Defaults to 1.
        bplt : bool, optional
            If True, visualize the isolated subsets and the distance histogram.
            Defaults to False.

        Returns
        -------
        fixed_subset, moving_subset : tuple[np.ndarray, np.ndarray]
            The isolated subsets of the fixed and moving point clouds.
        """
        A_to_B = lambda A, B: cKDTree(B).query(A, k=1)

        dist_f2m, _ = A_to_B(fixed_pts, moving_pts)
        dist_m2f, _ = A_to_B(moving_pts, fixed_pts)

        if max_distance is None:
            dist_hist, dist_bins = np.histogram(np.hstack((dist_f2m, dist_m2f)), 50)

            max_distance = dist_bins[np.nanargmax(dist_hist) + bins_after_max]

            if False:
                fig, ax = plt.subplots()

                ax.hist(dist_f2m, bins=50, alpha=0.5, label="fixed → moving")
                ax.hist(dist_m2f, bins=50, alpha=0.5, label="moving → fixed")

                ax.hist(np.hstack((dist_f2m, dist_m2f)), bins=50, alpha=0.5, label="all")
                ax.vlines([max_distance], 0, np.nanmax(dist_hist), label='max distance')

                ax.set_xlabel("Distance")
                ax.set_ylabel("Count")
                
                plt.show()

        fixed_subset = fixed_pts[dist_f2m <= max_distance]
        moving_subset = moving_pts[dist_m2f <= max_distance]

        if bplt:
            Isolator.plot_isolated_areas(fixed_subset, moving_subset, fixed_pts, moving_pts)
            Isolator.plot_colored_distances(fixed_subset, moving_subset, log=True)
        return fixed_subset, moving_subset

    @staticmethod
    def isolate_manual(left_pcd: np.ndarray, right_pcd: np.ndarray):
        """
        Isolate points by manual selection in an interactive window.

        This function opens two separate windows, one for each point cloud,
        allowing the user to manually pick corresponding points. It is typically
        used to provide initial correspondences for algorithms like Kabsch.

        Parameters
        ----------
        left_pcd : np.ndarray
            The first point cloud.
        right_pcd : np.ndarray
            The second point cloud.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the selected points from the left and right clouds.
        """
        queue_l = mp.Queue()
        queue_r = mp.Queue()

        p_l = mp.Process(target=Isolator._pick_point, args=(left_pcd, 'Select points L', queue_l, 0))
        p_r = mp.Process(target=Isolator._pick_point, args=(right_pcd, 'Select points R', queue_r, 1000))

        p_l.start()
        p_r.start()

        p_l.join()
        p_r.join()

        fixed_subset = queue_l.get()
        moving_subset = queue_r.get()

        print(f"\033[95mSelected point left =\033[0m",
               np.array2string(fixed_subset, formatter={'float_kind': lambda x: f"{x:.8f}"}))
        print()
        print(f"\033[96mSelected point right =\033[0m",
               np.array2string(moving_subset, formatter={'float_kind': lambda x: f"{x:.8f}"}))

        return fixed_subset, moving_subset
    
    @staticmethod
    def isolate_convex_hull(left_pcd, right_pcd, bplt=False):
        """
        Isolate points by selecting the points that lie within the convex hull of the other cloud.
        
        Parameters
        ----------
        left_pcd : np.ndarray
            The first point cloud.
        right_pcd : np.ndarray
            The second point cloud.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the selected points from the left and right clouds.
        """
        from scipy.spatial import ConvexHull, Delaunay

        def points_in_hull(points, hull):
            delaunay = Delaunay(hull.points[hull.vertices])
            return delaunay.find_simplex(points) >= 0

        hull_left = ConvexHull(left_pcd)
        hull_right = ConvexHull(right_pcd)

        left_in_right = points_in_hull(left_pcd, hull_right)
        right_in_left = points_in_hull(right_pcd, hull_left)

        fixed_subset = left_pcd[left_in_right]
        moving_subset = right_pcd[right_in_left]
        
        if bplt: 
            Isolator.plot_isolated_areas(fixed_subset, moving_subset, left_pcd, right_pcd)
            Isolator.plot_colored_distances(fixed_subset, moving_subset, log=True)

        return fixed_subset, moving_subset

    @staticmethod
    def _pick_point(points, window_name, queue, left=0):
            """
            Internal helper to run a single point picking window.

            This function is executed in a separate process for `isolate_manual`.
            It displays a point cloud and waits for the user to select points,
            then places the selected points into a multiprocessing queue.
            """
            pcd_o3d = sutils.pcd_to_o3d_pcd(points)

            print(f"\n{window_name}")
            print("Select the point with Shift + left click")
            print("Remove the last selected point with Shift + right")

            pcd_o3d = splotter.assign_defined_colors_to_point_clouds([pcd_o3d], colors="normal")[0]

            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name=window_name, width=900, height=900, left=left, top=50)
            vis.add_geometry(pcd_o3d)

            render_option = vis.get_render_option()
            render_option.point_size = 2

            vis.run()
            vis.destroy_window()

            picked = vis.get_picked_points()
            queue.put(points[picked])

class Thresholder():
    """
    A factory class for determining the distance threshold for registration.

    Registration algorithms like ICP and FGR require a `max_correspondence_distance`
    threshold. This class provides different strategies for calculating an
    appropriate value for this threshold.

    Attributes
    ----------
    type : str
        The selected thresholding strategy.
    value : str
        Identifier for using a fixed, user-provided value.
    KDTree : str
        Identifier for estimating the threshold from nearest-neighbor distances.

    """
    type: str

    value: str = 'value'
    KDTree: str = 'KDTree'

    def __init__(self, type: str, val=20, threshold_expansion=1.2):
        """
        Initializes the Thresholder with a specific strategy.

        Parameters
        ----------
        type : str
            The thresholding strategy to use. Must be 'value' or 'KDTree'.
        val : float, optional
            The fixed threshold value to use if `type` is 'value'.
            Defaults to 20.
        threshold_expansion : float, optional
            A multiplier applied to the estimated threshold if `type` is
            'KDTree'. Defaults to 1.2.
        """
        self.type = type
        self.threshold_expansion = threshold_expansion
        self.val = val

    def apply_thresholder(self, fixed_subset, moving_subset):
        """
        Apply the selected thresholding strategy.

        Parameters
        ----------
        fixed_subset : np.ndarray
            The fixed point cloud subset.
        moving_subset : np.ndarray
            The moving point cloud subset.

        Returns
        -------
        float
            The calculated distance threshold.
        """
        if self.type == 'value': return self.threshold_value(self.val)
        elif self.type == 'KDTree': return self.threshold_KDTree(fixed_subset, moving_subset, self.threshold_expansion)
        else:
            raise ValueError('Unknown thresholder type')

    @staticmethod
    def threshold_value(x): return x

    @staticmethod
    def threshold_KDTree(fixed_subset, moving_subset, threshold_expansion=1.2):
        """
        Estimate threshold from the mean distance of mutual nearest neighbors.
        """
        diffs = KDTree_mutual_diffs(fixed_subset, moving_subset)
        threshold = np.mean(np.abs(diffs))
        threshold_dev = np.std(np.abs(diffs))

        return threshold * threshold_expansion


class SurfaceStitcher:
    """
    A collection of static methods for stitching multiple point clouds.

    This class provides a suite of different algorithms for registering and
    merging a sequence of point clouds. Each method takes a list of point
    clouds and returns a single, merged point cloud along with the list of
    individually transformed clouds.

    The general workflow for most methods is to iteratively align each
    point cloud (`moving`) to the accumulated result of the previous alignments
    (`fixed`).
    """
    @staticmethod
    @sutils.ensure_numpy_pcd
    def stitchSavedTransforms(point_clouds: list[np.ndarray], transforms_folder, bplt=True):
        """
        Stitch point clouds by applying pre-computed, saved transformations.

        Parameters
        ----------
        point_clouds : list[np.ndarray]
            List of point clouds to stitch.
        transforms_folder : str or Path
            Path to the folder containing saved `TransformParams` .pkl files.
            Files should be named like '0.pkl', '1.pkl', etc.
        bplt : bool, optional
            If True, visualize the results. Defaults to True.
        """

        transforms_folder = Path(transforms_folder)

        fixed = np.asarray(point_clouds[0])
        point_clouds_T = [point_clouds[0]]

        transform_files = sorted(
            transforms_folder.glob("*.pkl"),
            key=lambda p: int(p.stem)
        )

        if len(transform_files) < len(point_clouds) - 1:
            raise ValueError(
                f"[ERROR SAVED_TRANSFORMS] Not enough transform files. Expected at least {len(point_clouds) - 1}, found {len(transform_files)}"
            )

        loaded_transforms_params = []
        for f in transform_files:
            with open(f, "rb") as pkl_file:
                loaded_obj = pickle.load(pkl_file)
                if isinstance(loaded_obj, TransformParams):
                    loaded_transforms_params.append(loaded_obj)
                elif isinstance(loaded_obj, np.ndarray) and loaded_obj.shape == (4, 4):
                    loaded_transforms_params.append(loaded_obj)
                else:
                    raise TypeError(f"Unexpected type in pickle file: {type(loaded_obj)}. Expected TransformParams or 4x4 np.ndarray.")

        # The first point cloud is the reference, no transform applied to it.
        # The loop starts from the second point cloud (index 1).
        for i, pc in enumerate(point_clouds[1:]): # i will be 0 for point_clouds[1], 1 for point_clouds[2], etc.
            moving = np.asarray(pc)

            # Apply the i-th loaded transform (which corresponds to point_clouds[i+1])
            moved = apply_transform(moving, loaded_transforms_params[i])
            point_clouds_T.append(moved)

            fixed = np.vstack([fixed, moved])

        if bplt:
            # show_point_clouds([fixed], colors="uniform")
            # show_point_clouds(point_clouds_T, colors="normal")
            splotter.compare_point_clouds([[fixed], point_clouds_T], ["normal", "uniform"])

        return fixed, point_clouds_T

    @staticmethod
    @sutils.ensure_numpy_pcd
    def stitchManual(point_clouds: list[np.ndarray], points_in_sphere=100,  bplt=False, save_transform=None):
        """
        Stitch point clouds using manual correspondence selection.

        This method iteratively prompts the user to select at least 3
        corresponding points between the fixed and moving clouds. It then uses
        the Kabsch algorithm (`TransformParams.from_kabsch`) to compute the
        optimal rigid transformation and applies it.

        Parameters
        ----------
        point_clouds : list[np.ndarray]
            List of point clouds to stitch.
        points_in_sphere : int, optional
            Instead of using the single selected point, a patch of this many
            points around the selection is used, and their centroid is taken
            as the correspondence. This can improve robustness to selection
            inaccuracy. Defaults to 100.
        bplt : bool, optional
            If True, visualize the results. Defaults to False.
        save_transform : str or None, optional
            If provided, path to a folder to save the computed transformations.
        """

        def get_n_closest_points(pcd, center):
            # n_points = pcd.shape[0]
            # auto_radius = (n_points / 100) * 0.05
            # print(f"RADIUS: {auto_radius}")
            tmp = pcd - center
            modules = np.linalg.norm(tmp, axis=1)
            indices = np.argsort(modules)[:points_in_sphere]

            return indices

        def get_patch(cloud, center):
            mean_patch_point = np.mean(cloud[get_n_closest_points(cloud, center)], axis=0)
            return mean_patch_point

        def optimize(fixed, moving):

            voxel_size = 100

            fixed_o3d = sutils.pcd_to_o3d_pcd(fixed)
            moving_o3d = sutils.pcd_to_o3d_pcd(moving)

            fp_d = np.asarray(fixed_o3d.points)
            mp_d = np.asarray(moving_o3d.points)

            fp, mp = Isolator.isolate_manual(fp_d, mp_d)

            if len(fp) != len(mp):
                raise RuntimeError("[ERROR MANUAL] Select same amount of points from left and right")

            if len(fp) < 3:
                print("[ERROR MANUAL] You need to select at least 3 points!")

            fixed_patches = []
            moving_patches = []

            for pf, pm in zip(fp, mp):
                fixed_patches.append(get_patch(fixed, pf))
                moving_patches.append(get_patch(moving, pm))

            fp = np.vstack(fixed_patches)
            mp = np.vstack(moving_patches)

            RTM: TransformParams = TransformParams.from_kabsch(mp, fp)
            if save_transform is not None:
                save_folder = os.path.join(save_transform, "man")
                
                os.makedirs(save_folder, exist_ok=True)
                RTM.to_pickle(os.path.join(save_folder, f"{i}.pkl"))

            moved = apply_transform(moving, RTM)

            # apply transformatıon
            return moved

        fixed = np.asarray(point_clouds[0])
        point_clouds_T = [point_clouds[0]]

        for i, pc in enumerate(point_clouds[1:]):
            moving = np.asarray(pc)
            moved = optimize(fixed, moving)
            point_clouds_T.append(moved)

            fixed = np.vstack([fixed, moved])

        if bplt:
            splotter.show_point_clouds([fixed], colors=None)
            splotter.show_point_clouds(point_clouds_T, colors=None)

        return fixed, point_clouds_T

    @staticmethod
    @sutils.ensure_numpy_pcd
    def stitchRobot(point_clouds: list[np.ndarray], robotTfile, save_transform=None, bplt=False):
        """
        Stitch point clouds using transformation data from a robot's kinematics.

        This method reads transformation parameters directly from a file that
        logs the robot's pose for each scan. It assumes the file format is
        compatible with `TransformParams.from_file`.

        Parameters
        ----------
        point_clouds : list[np.ndarray]
            List of point clouds to stitch.
        robotTfile : str
            Path to the file containing robot transformation data.
        save_transform : str or None, optional
            If provided, path to a folder to save the transformations.
        bplt : bool, optional
            If True, visualize the results. Defaults to False.
        """
        robot_trans = []
        point_clouds_clean = []

        for i, pc in enumerate(point_clouds):
            pc = sutils.remove_outliers_from_point_cloud(pc)
            point_clouds_clean.append(pc)

            tr = TransformParams.from_file(robotTfile, i, header=1)
            tr.rescale(1000)
            if save_transform is not None:
                save_folder = os.path.join(save_transform, "rob")
                
                os.makedirs(save_folder, exist_ok=True)
                tr.to_pickle(os.path.join(save_folder, f"{i}.pkl"))
            
            robot_trans.append(tr)

        print(f"[INFO ROBOT STITCH] Loaded {len(robot_trans)} robot transformations for {len(point_clouds_clean)} surfaces")

        point_clouds_T = []
        fixed = point_clouds_clean[0]
        for pc, trasf in zip(point_clouds_clean, robot_trans):
            pts = pc
            pts_T = apply_transform(pts, trasf, params0=robot_trans[0])
            point_clouds_T.append(pts_T)
            fixed = sutils.merge_and_downsample_point_cloud(fixed, pts_T)  # TODO: check why we downsaple

        if bplt:
            splotter.compare_point_clouds([[fixed], point_clouds_T], ["afmhot", "uniform"])

        return fixed, point_clouds_T

    @staticmethod
    @sutils.ensure_numpy_pcd
    def stitchRMSE(point_clouds: list[np.ndarray], n_calls, isolator: Isolator, save_transform=None, bplt=False):
        """
        Stitch point clouds by minimizing RMSE using Bayesian optimization.

        This method searches for the optimal 6-DOF transformation by minimizing
        the Root Mean Square Error between the overlapping regions of the
        point clouds. The search is performed using Gaussian Process regression
        (`skopt.gp_minimize`).

        Parameters
        ----------
        point_clouds : list[np.ndarray]
            List of point clouds to stitch.
        n_calls : int
            The number of iterations (evaluations of the objective function)
            for the Bayesian optimization.
        isolator : Isolator
            An `Isolator` instance to define the overlapping region where RMSE
            is calculated.
        save_transform : str or None, optional
            If provided, path to a folder to save the computed transformations.
        bplt : bool,optional
            If True, visualize the results and the optimization progress.
        """
        def optimize(fixed_pts, moving_pts):
            U_tx, U_ty, U_tz = 56, 56, 56
            U_theta = 0.5

            # tx ty tz rx ry rz
            t0 = [0, 0, 0, 0, 0, 0]

            def objective(x):
                p = TransformParams.from_list(x)
                moved = apply_transform(moving_pts, p)

                fixed_sub, moved_sub = isolator.apply_isolator(fixed_pts, moved, bplt=False)
                plt.show()
                diffs = KDTree_mutual_diffs(fixed_sub, moved_sub)  # just use the dıfference, not the mutual kdtree

                rmse = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))

                npoints.append(len(diffs))
                rmses.append(rmse)
                return rmse

            space = [
                Real(t0[0] - U_tx, t0[0] + U_tx),
                Real(t0[1] - U_ty, t0[1] + U_ty),
                Real(t0[2] - U_tz, t0[2] + U_tz),
                Real(t0[3] - U_theta, t0[3] + U_theta),
                Real(t0[4] - U_theta, t0[4] + U_theta),
                Real(t0[5] - U_theta, t0[5] + U_theta),
            ]

            res = gp_minimize(objective, space, x0=t0, n_calls=n_calls, random_state=42, n_jobs=4, verbose=True)

            best = res.x
            best_p = TransformParams.from_list(best)
            if save_transform is not None:
                save_folder = os.path.join(save_transform, "rmse")
                
                os.makedirs(save_folder, exist_ok=True)
                best_p.to_pickle(os.path.join(save_folder, f"{i}.pkl"))

            aligned = apply_transform(moving_pts, best_p)

            return aligned, res

        fixed = np.asarray(point_clouds[0])
        point_clouds_T = [point_clouds[0]]

        for i, pc in enumerate(point_clouds[1:]):
            rmses = []
            npoints = []

            moving = np.asarray(pc)
            print(f'[INFO RMSE] Optimizing image {i}')
            optimized_moving, _ = optimize(fixed, moving)
            point_clouds_T.append(optimized_moving)

            fixed = np.vstack([fixed, optimized_moving])

            if bplt:
                fig, (ax, bx) = plt.subplots(2, 1)
                ax.set_title(f'Image {i} opt')

                ax.plot(range(1, n_calls+1), rmses)
                ax.set_xlabel("Number of call")
                ax.set_ylabel("RMSE")
                ax.grid(True)

                bx.plot(range(1, n_calls+1), npoints)
                bx.set_xlabel("Number of call")
                bx.set_ylabel("npoints")
                bx.grid(True)

        if bplt:
            splotter.compare_point_clouds([[fixed], point_clouds_T], ["viridis", "uniform"])

        return fixed, point_clouds_T

    @staticmethod
    @sutils.ensure_numpy_pcd
    def stitchICP(point_clouds: list[np.ndarray], thresholder: Thresholder, isolator: None | Isolator, save_transform, bplt=False):
        """
        Stitch point clouds using the Iterative Closest Point (ICP) algorithm.

        This is a classic fine registration algorithm that iteratively refines
        the alignment by minimizing the distance between corresponding points.

        Parameters
        ----------
        point_clouds : list[np.ndarray]
            List of point clouds to stitch.
        thresholder : Thresholder
            A `Thresholder` instance to determine the max correspondence distance.
        isolator : Isolator or None
            An `Isolator` instance to select the overlapping region. If None,
            the entire point clouds are used.
        save_transform : str or None
            If provided, path to a folder to save the computed transformations.
        bplt : bool, optional
            If True, visualize the results. Defaults to False.
        """
        def optimize(fixed_pts, moving_pts):

            if isolator != None:
                fixed_subset, moving_subset = isolator.apply_isolator(fixed_pts, moving_pts)
            else:
                fixed_subset, moving_subset = fixed_pts, moving_pts

            pc_fixed = sutils.pcd_to_o3d_pcd(fixed_subset)
            pc_moving = sutils.pcd_to_o3d_pcd(moving_subset)

            trans_init = np.eye(4)
            threshold = thresholder.apply_thresholder(fixed_subset, moving_subset)
            if threshold == 0:
                print("[WARNING ICP] Threshold is 0, setting to 1 to avoid errors")
                threshold = 1
            print(f'{threshold=}')

            reg_p2p = o3d.pipelines.registration.registration_icp(
                pc_moving, pc_fixed, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

            if save_transform is not None:
                save_folder = os.path.join(save_transform, "icp")
                
                os.makedirs(save_folder, exist_ok=True)
                with open(os.path.join(save_folder, f"{i}.pkl"), "wb") as f:
                    pickle.dump(reg_p2p.transformation, f)

            aligned = apply_transform(moving_pts, reg_p2p.transformation)

            return aligned, reg_p2p
        
        fixed = np.asarray(point_clouds[0])
        point_clouds_T = [point_clouds[0]]

        for i, pc in enumerate(point_clouds[1:]):
            moving = np.asarray(pc)

            print(f"[INFO ICP] Optimizing image {i}")

            optimized_moving, _ = optimize(fixed, moving)
            point_clouds_T.append(optimized_moving)

            fixed = np.vstack([fixed, optimized_moving])

        if bplt:
            splotter.compare_point_clouds([[fixed], point_clouds_T], ["viridis", "uniform"])

        return fixed, point_clouds_T

    @staticmethod
    @sutils.ensure_numpy_pcd
    def stitchFGR(point_clouds: list[np.ndarray], voxel_size: float, thresholder: Thresholder, isolator: None | Isolator, save_transform, bplt=False):
        """
        Stitch point clouds using Fast Global Registration (FGR).

        FGR is a global registration algorithm that does not require a close
        initial alignment. It works by matching FPFH (Fast Point Feature
        Histograms) features between the two clouds.

        Parameters
        ----------
        point_clouds : list[np.ndarray]
            List of point clouds to stitch.
        voxel_size : float
            The voxel size for downsampling the point clouds before feature
            computation. A smaller value preserves more detail but increases
            computation time.
        thresholder : Thresholder
            A `Thresholder` instance to determine the correspondence distance.
        isolator : Isolator or None
            An `Isolator` instance to select the overlapping region.
        save_transform : str or None
            If provided, path to a folder to save the computed transformations.
        bplt : bool, optional
            If True, visualize the results. Defaults to False.
        """
        def optimize(fixed_pts, moving_pts):
            [fixed_scaled, moving_scaled], scales = sutils.rescale_point_cloud([fixed_pts, moving_pts])

            if isolator != None:
                fixed_subset, moving_subset = isolator.apply_isolator(fixed_scaled, moving_scaled, bplt=False)
            else:
                fixed_subset, moving_subset = fixed_scaled, moving_scaled

            pc_fixed = sutils.pcd_to_o3d_pcd(fixed_subset)
            pc_moving = sutils.pcd_to_o3d_pcd(moving_subset)

            pc_fixed_down = pc_fixed.voxel_down_sample(voxel_size)
            pc_moving_down = pc_moving.voxel_down_sample(voxel_size)

            radius_normal = voxel_size * 2
            pc_fixed_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )
            pc_moving_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )

            radius_feature = voxel_size * 5
            fixed_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pc_fixed_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )

            moving_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pc_moving_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )

            threshold = thresholder.apply_thresholder(fixed_subset, moving_subset)
            print(f'{threshold=}')
            reg_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                pc_moving_down,
                pc_fixed_down,
                moving_fpfh,
                fixed_fpfh,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=threshold
                )
            )

            if save_transform is not None:
                save_folder = os.path.join(save_transform, "fgr")
                
                os.makedirs(save_folder, exist_ok=True)
                with open(os.path.join(save_folder, f"{i}.pkl"), "wb") as f:
                    pickle.dump(reg_fgr.transformation, f)

            aligned_scaled = apply_transform(moving_scaled, reg_fgr.transformation)

            [aligned], _ = sutils.rescale_point_cloud(
                [aligned_scaled],
                scales=scales,
                revert=True
            )

            return aligned

        fixed = np.asarray(point_clouds[0])
        point_clouds_T = [point_clouds[0]]

        for i, pc in enumerate(point_clouds[1:]):
            moving = np.asarray(pc)
            print(f"[INFO FGR] Optimizing image {i}")
            optimized_moving = optimize(fixed, moving)
            point_clouds_T.append(optimized_moving)
            fixed = np.vstack([fixed, optimized_moving])

        if bplt:
            splotter.compare_point_clouds([[fixed], point_clouds_T], ["viridis", "uniform"])

        return fixed, point_clouds_T

    @staticmethod
    @sutils.ensure_numpy_pcd
    def stitchCorrelation(point_clouds: list[np.ndarray], dx: float, dy: float, isolator: None | Isolator, correlateDer=True, save_transform=None, bplt=False):
        """
        Stitch point clouds using 2D phase cross-correlation.

        This method is suitable for data that can be represented as 2D height
        maps (surfaces). It converts the overlapping regions of the point clouds
        to surfaces, then uses `skimage.registration.phase_cross_correlation`
        to find the translational shift (tx, ty) between them. The vertical
        shift (tz) is then estimated by comparing the mean heights in the
        aligned overlap region.

        Parameters
        ----------
        point_clouds : list[np.ndarray]
            List of point clouds to stitch.
        dx, dy : float
            Grid spacing to use when converting point clouds to surfaces.
        isolator : Isolator or None
            An `Isolator` instance to select the overlapping region.
        correlateDer : bool, optional
            If True, perform correlation on the derivative of the surfaces,
            which can make the algorithm more robust to height differences.
        save_transform : str or None
            If provided, path to a folder to save the computed transformations.
        bplt : bool, optional
            If True, visualize the results. Defaults to False.
        """
        def optimize(fixed_pts, moving_pts):

            if isolator != None:
                fixed_subset, moving_subset = isolator.apply_isolator(fixed_pts, moving_pts, bplt=False)
            else:
                fixed_subset, moving_subset = fixed_pts, moving_pts

            fixed_surf, moving_surf = sutils.pcd_to_surface([fixed_subset, moving_subset], dx, dy, force_same_size=True, bplt=False)

            print('shapes:', fixed_surf.Z.shape, moving_surf.Z.shape)

            if fixed_surf.Z.shape != moving_surf.Z.shape:
                raise ValueError("[ERROR COR] fixed_surf and moving_surf must have the same shape.")

            lzone = copy.deepcopy(fixed_surf.Z)
            rzone = copy.deepcopy(moving_surf.Z)

            print(f"[INFO COR] {lzone.shape=}, {rzone.shape=}")

            if correlateDer:
                lzone = np.diff(lzone)
                rzone = np.diff(rzone)

            mask = np.logical_or(lzone.mask, rzone.mask)
            plt.figure()
            plt.imshow(mask, origin='lower')

            shift, _, _ = phase_cross_correlation(
                lzone,
                rzone,
                upsample_factor=10,
                reference_mask=np.logical_not(mask)
            )

            shift_y, shift_x = shift

            tx = shift_x * dx
            ty = shift_y * dy
            if bplt:
                fig, (ax, bx) = plt.subplots(nrows=1, ncols=2)
                ax.imshow(lzone)
                ax.set_title("lzone")
                bx.imshow(rzone)
                bx.set_title("rzone")
                funct.persFig([ax, bx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')
                plt.show()

            aligned = moving_pts.copy()
            aligned[:, 0] += tx
            aligned[:, 1] += ty

            # temp = moving_pts.copy()
            # temp[:, :2] += [tx, ty]

            # selected_points_f = (fixed_pts[:, 0] >= temp[:, 0].min()) & (fixed_pts[:, 0] <= temp[:, 0].max()) & (fixed_pts[:, 1] >= temp[:, 1].min()) & (fixed_pts[:, 1] <= temp[:, 1].max())
            # selected_points_m = (temp[:, 0] >= fixed_pts[:, 0].min()) & (temp[:, 0] <= fixed_pts[:, 0].max()) & (temp[:, 1] >= fixed_pts[:, 1].min()) & (temp[:, 1] <= fixed_pts[:, 1].max())
            # tz = np.median(fixed_pts[selected_points_f, 2]) - np.median(temp[selected_points_m, 2])

            fix_sub, temp_sub = isolator.isolate_common_points_kdtree(fixed_pts, aligned, bins_after_max=1, bplt=True)
            tz = np.nanmean(fix_sub[:, 2]) - np.nanmean(temp_sub[:, 2])

            aligned[:, 2] += tz

            tr = TransformParams.from_numbers(tx, ty, tz, 0, 0, 0)
            if save_transform is not None:
                save_folder = os.path.join(save_transform, "cc")
                
                os.makedirs(save_folder, exist_ok=True)
                tr.to_pickle(os.path.join(save_folder, f"{i}.pkl"))

            print(f"before mean z = {np.mean(moving_pts[:, 2])}")
            print(f"after mean z  = {np.mean(aligned[:, 2])}")

            # if bplt:
            #     fig, ((ax, bx, cx), (gx, ex, fx)) = plt.subplots(nrows=2, ncols=3)
            #     ax.imshow(ccL)
            #     ax.set_title('ccL')
            #     ax.plot(xML, yML, 'ro', ms=5)
            #     bx.imshow(lzone)
            #     bx.set_title('lzone')
            #     cx.imshow(sampleR)
            #     cx.set_title('rsample')

            #     gx.imshow(ccR)
            #     gx.set_title('ccR')
            #     gx.plot(xMR, yMR, 'ro', ms=5)
            #     ex.imshow(rzone)
            #     ex.set_title('rzone')
            #     fx.imshow(sampleL)
            #     fx.set_title('lsample')
            #     funct.persFig([ax, bx, cx, gx, ex, fx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')

            #     fig2, (lx, mx, nx) = plt.subplots(nrows=1, ncols=3)
            #     lx.imshow(ccL)
            #     lx.set_title('ccL')
            #     mx.imshow(np.flip(ccR))
            #     mx.set_title('ccR rot')
            #     nx.imshow(ccL * np.flip(ccR))
            #     nx.set_title('cc mult')
            #     funct.persFig([lx, mx, nx], xlab='x [pixels]', ylab='y [pixels]', gridcol='none')
            #     nx.plot(xM, yM, 'r.', ms=5)

            #     plt.show()

            return aligned, [tx, ty]

        fixed = np.asarray(point_clouds[0])
        point_clouds_T = [point_clouds[0]]

        for i, pc in enumerate(point_clouds[1:]):
            moving = np.asarray(pc)
            print(f"[INFO COR] Optimizing image {i}")
            optimized_moving, _ = optimize(fixed, moving)
            point_clouds_T.append(optimized_moving)
            fixed = np.vstack([fixed, optimized_moving])

        if bplt:
            splotter.compare_point_clouds([[fixed], point_clouds_T], ["viridis", "uniform"])

        return fixed, point_clouds_T
from datetime import datetime

import open3d as o3d
import numpy as np
from scipy import ndimage
import multiprocessing as mp
from scipy.spatial import ConvexHull, Delaunay


import os, pickle
from functools import wraps
from surfile import surface

def to_numpy(item):
    """
    Convert an item to a NumPy array representation of a point cloud.

    This helper function attempts to convert various data types into a
    standard (N, 3) NumPy array. It handles Open3D PointClouds, objects
    with a `getPoints` method (like `surfile.surface.Surface`), and
    existing NumPy arrays.

    Parameters
    ----------
    item : object
        The item to convert. Supported types are `np.ndarray`,
        `o3d.geometry.PointCloud`, and objects with a `getPoints` method.

    Returns
    -------
    np.ndarray
        The converted (N, 3) NumPy array. If conversion is not possible,
        the original item is returned with a warning.
    
    Notes
    -----
    For `surfile.surface.Surface` objects, `getPoints(exclude_nan=True)` is called.
    """
    if isinstance(item, np.ndarray):
        return item
    
    if isinstance(item, o3d.geometry.PointCloud):
        print(f'[INFO STITCH] Auto-Converted {type(item)} to ndarray')
        return np.asarray(item.points)
    
    if hasattr(item, 'getPoints'):
        print(f'[INFO STITCH] Auto-Converted {type(item)} to ndarray')
        return np.asarray(item.getPoints(exclude_nan=True))
    
    print(f'[WARN STITCH] Could not ensure ndarray from type {type(item)}')
    return item

def ensure_numpy_pcd(func):
    """
    Decorator to ensure the input data is a NumPy array point cloud.

    This decorator automatically converts the first argument of the decorated
    function into a NumPy array of shape (N, 3). If the first argument is a
    list, it attempts to convert each element of the list.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function, which will receive the converted data.

    Examples
    --------
    >>> @ensure_numpy_pcd
    ... def process_points(points: np.ndarray):
    ...     print(points.shape)
    ...
    >>> o3d_pc = o3d.geometry.PointCloud()
    >>> o3d_pc.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    >>> process_points(o3d_pc)
    [INFO STITCH] Auto-Converted <class 'open3d.cpu.pybind.geometry.PointCloud'> to ndarray
    (100, 3)
    """
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, list):
            processed_data = [to_numpy(x) for x in data]
        else:
            processed_data = to_numpy(data)
            
        return func(processed_data, *args, **kwargs)
        
    return wrapper

def ensure_o3d_pc(func):
    """
    Decorator to ensure the input data is an Open3D PointCloud object.

    This decorator automatically converts the first argument of the decorated
    function into an `o3d.geometry.PointCloud`. If the first argument is a
    list, it attempts to convert each element of the list.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function, which will receive the converted data.

    Examples
    --------
    >>> @ensure_o3d_pc
    ... def visualize_points(pcd: o3d.geometry.PointCloud):
    ...     o3d.visualization.draw_geometries([pcd])
    ...
    >>> numpy_pc = np.random.rand(100, 3)
    >>> visualize_points(numpy_pc)
    [INFO STITCH] Auto-Converted <class 'numpy.ndarray'> to o3d_pcd
    # This will open an Open3D visualization window.
    """
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        def to_o3d(item):
            if isinstance(item, o3d.geometry.PointCloud):
                return item
            
            if isinstance(item, np.ndarray):
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(item)
                print(f'[INFO STITCH] Auto-Converted {type(item)} to o3d_pcd')
                return pc
            
            if hasattr(item, 'getPoints'):
                points = item.getPoints(exclude_nan=True)
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(np.asarray(points))
                print(f'[INFO STITCH] Auto-Converted {type(item)} to o3d_pcd')
                return pc
            
            print(f'[WARN STITCH] Could not ensure o3d PC from type {type(item)}')
            return item

        if isinstance(data, list):
            processed_data = [to_o3d(x) for x in data]
        else:
            processed_data = to_o3d(data)
            
        return func(processed_data, *args, **kwargs)
        
    return wrapper

@ensure_numpy_pcd
def pcd_least_squared_plane(pcds: list[np.ndarray]):
    """
    Calculate the normal vector of the best-fit plane for each point cloud.

    This function fits a plane to each point cloud using the method of least
    squares. The plane is defined by the equation $$z = ax + by + d$$.
    The normal vector to this surface is then computed and normalized.

    Parameters
    ----------
    pcds : list[np.ndarray]
        A list of point clouds, where each point cloud is a NumPy array of
        shape (N, 3).

    Returns
    -------
    list[np.ndarray]
        A list of normalized normal vectors, one for each input point cloud.
        Each normal vector is a NumPy array of shape (3,).

    Notes
    -----
    The least squares problem solves for the coefficients `C = [a, b, d]` in
    the system of equations `A @ C = z`, where `A` is a matrix with columns
    `[x, y, 1]` and `z` is the vector of z-coordinates.

    The plane equation can be written as $$ax + by - z + d = 0$$. The function returns
    the equivalent normal vector $$(-a, -b, 1)$$ after normalization, which
    points "upwards" in the z-direction relative to the plane's slope.
    """
    normals = []
    for pcd in pcds:
        points = pcd 
        A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
        C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
        a, b, d = C 
        
        normal = np.array([-a, -b, 1.0])
        normal /= np.linalg.norm(normal)

        normals.append(normal)
        
    return normals

@ensure_numpy_pcd
def pcd_to_surface(pcds: list[np.ndarray], dx, dy, force_same_size=True, bplt=False) -> list[surface.Surface]:
    """
    Convert point clouds to gridded `surfile.surface.Surface` objects.

    This function transforms one or more 3D point clouds into 2D height maps
    (surfaces). It performs the following steps for each point cloud:
    1.  (Optionally) Rotates the point cloud so its best-fit plane is aligned
        with the XY plane. Currently, this rotation is disabled.
    2.  Projects the points onto the XY plane.
    3.  Creates a 2D grid with spacing `dx` and `dy`.
    4.  Averages the Z-values of all points falling into each grid cell.
    5.  Interpolates to fill in any empty cells.
    6.  Constructs and returns a `surfile.surface.Surface` object.

    Parameters
    ----------
    pcds : list[np.ndarray]
        A list of point clouds to convert, each as an (N, 3) NumPy array.
    dx : float
        The desired grid spacing along the x-axis.
    dy : float
        The desired grid spacing along the y-axis.
    force_same_size : bool, optional
        If True, all output surfaces are cropped to the smallest dimensions
        among them to ensure they have the same size. Defaults to True.
    bplt : bool, optional
        If True, plots the resulting surface using `surface.setValues`.
        Defaults to False.

    Returns
    -------
    list[surface.Surface]
        A list of `surfile.surface.Surface` objects corresponding to the
        input point clouds.

    Notes
    -----
    The rotation logic to align the point cloud with the XY plane is currently
    bypassed (`if True: R = np.eye(3)`). The implementation uses Rodrigues'
    rotation formula to find the rotation matrix `R` that would align the
    plane's normal with the z-axis, but this is not applied.

    The binning and averaging is performed using `numpy.histogram2d`.
    Empty grid cells are filled using `scipy.ndimage.map_coordinates` with
    linear interpolation (`order=1`) and nearest-neighbor extrapolation
    for points outside the original data boundary.
    """
    z_maps = []

    for pcd in pcds:
        points = pcd

        normal = pcd_least_squared_plane(pcds)

        z_axis = np.array([0, 0, 1])
        v = np.cross(normal, z_axis)
        c = np.dot(normal, z_axis)
        s = np.linalg.norm(v)
        
        if True:  # Already aligned
            R = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        
        centroid = np.mean(points, axis=0)
        centered_pts = points - centroid
        rotated_pts = centered_pts @ R.T
        
        x_pts = rotated_pts[:, 0]
        y_pts = rotated_pts[:, 1]
        z_pts = rotated_pts[:, 2] # These are now distances from the plane
        
        x_min, x_max = x_pts.min(), x_pts.max()
        y_min, y_max = y_pts.min(), y_pts.max()
        n_x = int(np.ceil((x_max - x_min) / dx))
        n_y = int(np.ceil((y_max - y_min) / dy))
        grid_x = np.linspace(x_min, x_min + n_x * dx, n_x)
        grid_y = np.linspace(y_min, y_min + n_y * dy, n_y)
        gx, gy = np.meshgrid(grid_x, grid_y)

        print("\n--- GRID INFO ---")
        print(f"x_min: {x_min:.3f}, x_max: {x_max:.3f}, range: {x_max - x_min:.3f}")
        print(f"y_min: {y_min:.3f}, y_max: {y_max:.3f}, range: {y_max - y_min:.3f}")

        print(f"n_x: {n_x}, n_y: {n_y}")
        print(f"dx: {dx}, dy: {dy}")

        print(f"grid_x shape: {grid_x.shape}, grid_y shape: {grid_y.shape}")
        print(f"meshgrid shape: gx: {gx.shape}, gy: {gy.shape}")
        print("------------------\n")
        
        z_sum, _, _ = np.histogram2d(y_pts, x_pts, bins=[grid_y, grid_x], weights=z_pts)
        
        # 3. Calculate the count of points in each bin
        counts, _, _ = np.histogram2d(y_pts, x_pts, bins=[grid_y, grid_x])
                
        # Average the bins and fill NaNs
        z_sum = np.divide(z_sum, counts, out=np.zeros_like(z_sum), where=counts!=0)
        
        mean_val = np.nanmean(z_pts)
        z_sum[counts == 0] = mean_val
        
        print(f'[INFO PCD_TO_SUR] Could not bin {z_sum[counts == 0].size} elements')

        # Create the coordinate map (the "query" points in index space)
        # Since gx and gy are already spaced by dx/dy, their index-space is just a ramp
        coords = np.array([
            (gy - y_min) / dy, 
            (gx - x_min) / dx
        ])[:, 0:-1, 0:-1]
        
        print(f'[INFO PCD_TO_SUR] Converting pc using spacings dx: {dx:.3f} um, dy: {dy:.3f} um, coords shape: {coords.shape}, mask shape: {counts.shape}')
        # order=3 is equivalent to cubic interpolation
        z_map = ndimage.map_coordinates(z_sum, coords, order=1, mode='nearest')
        z_map = np.ma.array(z_map, mask=(counts == 0))

        z_maps.append(z_map)

    if force_same_size == True:
        min_rows = min(z_map.shape[0] for z_map in z_maps)
        min_cols = min(z_map.shape[1] for z_map in z_maps)

        for i in range(len(z_maps)):
            z_maps[i] = z_maps[i][:min_rows, :min_cols]

    surfs = []

    for z_map in z_maps:
        surf = surface.Surface()
        surf.setValues(dx, dy, z_map, bplt=bplt)
        surfs.append(surf)

    return surfs

@ensure_numpy_pcd
def pcd_to_o3d_pcd(pcd: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert a NumPy array point cloud to an Open3D PointCloud object.

    Parameters
    ----------
    pcd : np.ndarray
        The input point cloud as a NumPy array of shape (N, 3).

    Returns
    -------
    o3d.geometry.PointCloud
        The converted Open3D PointCloud object.
    
    Notes
    -----
    This is a convenience wrapper that handles the conversion.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pcd)
    
    return pc

@ensure_numpy_pcd
def rescale_point_cloud(point_clouds: list[np.ndarray], scales=None, revert=False):
    """
    Rescale a list of point clouds along x, y, and z axes.

    If `scales` is not provided, the scale factors are computed from the
    first point cloud in the list as:

        scales = maxs - mins

    where `mins` and `maxs` are the per-axis minimum and maximum values of
    the reference point cloud. The same scale factors are then applied to
    all point clouds in the list.

    If `revert` is True, the inverse scaling is applied by using `1 / scales`.

    Parameters
    ----------
    point_clouds : list[np.ndarray]
        List of point clouds, each of shape (N, 3).
    scales : np.ndarray or None, optional
        Array of shape (3,) containing the scale factors for x, y, and z.
        If None, the factors are computed from the first point cloud.
    revert : bool, optional
        If True, apply the inverse scaling factors instead of the direct
        scaling factors. Default is False.

    Returns
    -------
    scaled_list : list[np.ndarray]
        List of rescaled point clouds.
    scales : np.ndarray
        The scale factors that were used.

    Notes
    -----
    This function only divides coordinates by the scale factors.
    It does not shift the point clouds by subtracting their minimum values.
    Therefore, the output is not guaranteed to lie in the [0, 1] range.
    """
    if scales is None:
        ref = point_clouds[0]

        mins = ref.min(axis=0)
        maxs = ref.max(axis=0)
        scales = maxs - mins

    scaled_list = []
    scales = 1 / scales if revert else scales

    for pts in point_clouds:
        pts_scaled = pts / scales
        scaled_list.append(pts_scaled)

    return scaled_list, scales

@ensure_o3d_pc
def remove_outliers_from_point_cloud(point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Remove statistical outliers from a point cloud.

    This function uses Open3D's statistical outlier removal filter. For each
    point, it computes the average distance to its `k` nearest neighbors.
    Points whose average distance is outside a threshold (defined by the
    global average distance plus a number of standard deviations) are
    considered outliers and removed.

    Parameters
    ----------
    point_cloud : o3d.geometry.PointCloud
        The input point cloud.

    Returns
    -------
    np.ndarray
        A NumPy array of the point cloud with outliers removed.

    Notes
    -----
    The current implementation uses a fixed number of neighbors (`nb_neighbors=40`)
    and a standard deviation ratio (`std_ratio=3.0`). These parameters might
    need to be adjusted for different point cloud densities and noise levels.
    `nb_neighbors` high makes open_pc_cloud_from_file extremely heavy, `std_ratio` low gives a more aggressive filtration.
    """
    # TODO: maybe write your own outlier function?
    pc, _ = point_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    # pc, _ = point_cloud.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.1) # works quite well but not perfectly
    # pc, _ = point_cloud.remove_radius_outlier(nb_points=10, radius=0.05) # it doesn't work yet
    
    return np.asarray(pc.points)

def merge_and_downsample_point_cloud(pc1: np.ndarray, pc2: np.ndarray, voxel_size=0.001):
    """
    Merge two point clouds and then downsample the result using a voxel grid.

    This function combines two point clouds, creates a voxel grid with the
    specified `voxel_size`, and then averages all points within each voxel
    to a single point, effectively reducing the point cloud density.
    """
    combined = np.vstack([pc1, pc2])
    pc = pcd_to_o3d_pcd(combined)
    pc_down = pc.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pc_down.points)

@ensure_numpy_pcd
def cut_point_cloud(pc):
    """
    Cut a point cloud by selecting points using ``Isolator._pick_point``.

    The selected points are used to fit a least-squares plane, and the
    original point cloud is split into two subsets:
    ``pc_top`` on the normal side of the plane and ``pc_bottom`` on the
    opposite side.

    Parameters
    ----------
    pc : np.ndarray
        The input point cloud as an (N, 3) NumPy array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``pc_top`` and ``pc_bottom``.
    """
    from surfile.stitcher.stitcher import Isolator

    queue = mp.Queue()
    p = mp.Process(target=Isolator._pick_point, args=(pc, 'Select points of cut plane', queue, 0))

    print("[INFO CUT] Starting point picking process...")
    p.start()
    p.join()

    if queue.empty():
        raise RuntimeError("No points were selected for the cut plane.")

    selected_points = queue.get()
    print(f"[INFO CUT] Selected points: {selected_points}")

    if selected_points.ndim != 2 or selected_points.shape[1] != 3:
        raise ValueError("Selected points must be an (M, 3) array.")
    if selected_points.shape[0] < 3:
        raise ValueError("At least 3 points are required to define a cut plane.")

    normal = pcd_least_squared_plane([selected_points])[0]
    plane_point = np.mean(selected_points, axis=0)

    distances = (pc - plane_point) @ normal
    pc_top = pc[distances > 0]
    pc_bottom = pc[distances <= 0]
    print(f"[INFO CUT] Points above plane: {pc_top.shape[0]}, points below plane: {pc_bottom.shape[0]}")

    #  After asking the user which side to keep (top or bottom), the chosen subset is cleaned by keeping only
    # the points that lie within a distance from the cut plane. The distance threshold is half of the
    # maximum width of the selected points, where the maximum width is the maximum pairwise distance
    # between the selected points.

    return pc_top, pc_bottom


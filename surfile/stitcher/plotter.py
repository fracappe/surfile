import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import multiprocessing as mp

import surfile.stitcher.utils as sutils


def get_colors_from_weights(cmap_name, weights, log=False):
    """
    Map a set of weights to RGB colors using a specified Matplotlib colormap.

    Parameters
    ----------
    cmap_name : str
        The name of the Matplotlib colormap to use (e.g., "viridis", "plasma").
    weights : np.ndarray
        A 1D array of scalar values that will be mapped to colors. These should
        be normalized to the range [0, 1] for best results.

    Returns
    -------
    np.ndarray
        An (N, 3) array of RGB colors corresponding to the input weights.
    """    
    if log:
        weights = np.log(weights)  # log scaling to enhance contrast
        
    weights = np.asarray(weights)
    weights = weights / np.max(weights) if np.max(weights) > 0 else weights
    
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(weights)[:, :3]  # Get RGB values, ignore alpha channel
    return colors

@sutils.ensure_o3d_pc
def show_point_clouds(point_clouds: list[o3d.geometry.PointCloud], colors="normal"):
    """
    Visualize a list of point clouds in a single Open3D window.

    Parameters
    ----------
    point_clouds : list[o3d.geometry.PointCloud]
        A list of Open3D PointCloud objects to be visualized together.
    colors : str or None, optional
        The coloring scheme to apply to the point clouds. See
        `assign_defined_colors_to_point_clouds` for available options.
        If None, the original colors of the point clouds are used.
        Defaults to "normal".

    """
    if colors is not None:
        point_clouds = assign_defined_colors_to_point_clouds(point_clouds, colors=colors)
    o3d.visualization.draw_geometries(point_clouds, point_show_normal=False)

@sutils.ensure_o3d_pc
def compare_point_clouds(pc_lists: list[list[o3d.geometry.PointCloud]], colors: str | list[str]="normal"):
    """
    Compare multiple lists of point clouds, each in a separate window.

    This function uses multiprocessing to launch a separate visualization
    window for each list of point clouds provided.

    Parameters
    ----------
    pc_lists : list[list[o3d.geometry.PointCloud]]
        A list where each element is another list of point clouds to be
        displayed together in one window.
    colors : str or list[str], optional
        The coloring scheme(s) to apply. If a single string, the same scheme
        is applied to all windows. If a list of strings, each window gets
        the corresponding color scheme from the list. Defaults to "normal".

    """
    procs = []
    if type(colors) == str: colors = [colors for _ in range(len(pc_lists))]
    print(f"[INFO COMPARE PLOT] Plotting comparisons in multiple processes with colors: {colors}")

    for i, pc_list in enumerate(pc_lists):
        p = mp.Process(target=show_point_clouds, args=(pc_list, colors[i]))
        p.start()

        procs.append(p)
    
    for p in procs: p.join()

@sutils.ensure_o3d_pc
def color_points_from_closest_triangle_normal(pcd: o3d.geometry.PointCloud, method: str = "poisson", depth: int = 8, alpha: float = 1.0, orient_k: int = 30, remove_low_density: bool = True, density_quantile: float = 0.02):
    """
    Generate colors for a point cloud based on surface normals from a reconstructed mesh.

    This method provides a more robust way to color a point cloud by its
    surface orientation. Instead of using per-point normal estimates (which can
    be noisy), it first reconstructs a 3D mesh from the point cloud. Then, for
    each point in the original cloud, it finds the closest triangle on the mesh
    and assigns a color based on that triangle's normal vector.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The input point cloud.
    method : {'poisson', 'alpha'}, optional
        The mesh reconstruction algorithm to use. 'poisson' for Screened
        Poisson Surface Reconstruction, 'alpha' for Alpha Shapes.
        Defaults to "poisson".
    depth : int, optional
        The depth of the octree used for Poisson reconstruction. Higher values
        result in a more detailed mesh. Defaults to 8.
    alpha : float, optional
        The alpha parameter for the Alpha Shape algorithm. It controls the
        level of detail. Defaults to 1.0.
    orient_k : int, optional
        Number of nearest neighbors to use for consistent normal orientation.
        Defaults to 30.
    remove_low_density : bool, optional
        If True (for Poisson), removes vertices from areas of low point
        density, which often correspond to artifacts. Defaults to True.
    density_quantile : float, optional
        The quantile of densities to use as a threshold for removing
        low-density vertices. Defaults to 0.02.

    Returns
    -------
    colors : np.ndarray
        An (N, 3) array of RGB colors, where N is the number of points in `pcd`.
    mesh : o3d.geometry.TriangleMesh
        The reconstructed triangle mesh.
    tri_ids : np.ndarray
        An (N,) array where each element is the index of the closest triangle
        in `mesh` for the corresponding point in `pcd`.
    tri_normals : np.ndarray
        An (M, 3) array of the normal vectors for all M triangles in `mesh`.

    Raises
    ------
    ValueError
        If the input point cloud is empty or the method is not supported.
    RuntimeError
        If mesh reconstruction fails or if a closest triangle cannot be found
        for some points.

    """
    if len(pcd.points) == 0:
        raise ValueError("Input point cloud is empty.")

    pcd_work = o3d.geometry.PointCloud(pcd)

    # Normals are needed for most mesh reconstruction methods
    if len(pcd_work.normals) != len(pcd_work.points):
        pcd_work.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )

    try:
        pcd_work.orient_normals_consistent_tangent_plane(orient_k)
    except Exception:
        # If orientation fails, continue anyway
        pass

    # --- mesh reconstruction ---
    if method.lower() == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_work, depth=depth
        )

        if remove_low_density:
            densities = np.asarray(densities)
            keep_mask = densities > np.quantile(densities, density_quantile)
            mesh.remove_vertices_by_mask(~keep_mask)

    elif method.lower() == "alpha":
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_work)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd_work, alpha, tetra_mesh, pt_map
        )
    else:
        raise ValueError("method must be either 'poisson' or 'alpha'")

    if len(mesh.triangles) == 0:
        raise RuntimeError("Mesh reconstruction failed: mesh has no triangles.")

    mesh.compute_triangle_normals()
    tri_normals = np.asarray(mesh.triangle_normals)

    # --- convert mesh to tensor mesh for closest-triangle queries ---
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    points = np.asarray(pcd_work.points, dtype=np.float32)
    query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    # closest_points gives the primitive_ids (triangle indices)
    ans = scene.compute_closest_points(query_points)
    tri_ids = ans["primitive_ids"].numpy()

    # Some points may return invalid ids in edge cases
    if np.any(tri_ids < 0):
        raise RuntimeError("Some points could not be assigned to a closest triangle.")

    point_normals = tri_normals[tri_ids]

    # Map normals from [-1, 1] to [0, 1] for RGB coloring
    colors = (point_normals + 1.0) / 2.0

    return colors, mesh, tri_ids, tri_normals

@sutils.ensure_o3d_pc
def assign_defined_colors_to_point_clouds(point_clouds: list[o3d.geometry.PointCloud], colors: None | str | list[np.ndarray] = None):
    """Assign colors to a list of point clouds based on a defined scheme.

    Parameters
    ----------
    point_clouds : list[o3d.geometry.PointCloud]
        A list of PointCloud objects to be colored.
    colors : str, list or None, optional
        The coloring scheme to apply. The options are:
        - "uniform": Assigns a unique, uniform color to each point cloud in
          the list from a predefined colormap.
        - "normal": Colors each point based on its estimated normal vector.
          The XYZ components of the normal are mapped to RGB values.
        - "betternormal": Colors each point based on the normal of the
          closest triangle in a reconstructed mesh, providing a smoother
          and more robust coloring. See
          `color_points_from_closest_triangle_normal`.
        - A Matplotlib colormap name (e.g., "viridis", "plasma", "afmhot"):
          Colors each point based on its Z-coordinate, normalized across the
          point cloud's height.
        - A list of (N, 3) numpy arrays of RGB colors, where each array
          corresponds to the colors for the points in the respective point cloud.
        - None: No coloring is applied.
        Defaults to None.

    Returns
    -------
    list[o3d.geometry.PointCloud]
        The list of point clouds with updated colors.
    """
    # init for uniform
    num_pcs = len(point_clouds)
    cmap = plt.get_cmap("tab10")  # pastel1, pastel2, Accent
    unicolors = [cmap(j % 10) for j in range(num_pcs)]
    
    if isinstance(colors, str):
        colors = colors.lower()
        for i, pc in enumerate(point_clouds):
            if colors == "normal":
                pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
                # pc.orient_normals_consistent_tangent_plane(10)
                # pc.orient_normals_to_align_with_direction([1, 0, 0])
                normals = np.asarray(pc.normals)

                ncolors = (normals + 1) / 2  # da [-1,1] a [0,1]
                pc.colors = o3d.utility.Vector3dVector(ncolors)

            elif colors == "betternormal":
                ncolors, _, _, _ = color_points_from_closest_triangle_normal(pc)
                pc.colors = o3d.utility.Vector3dVector(ncolors)

            elif colors == "uniform":
                raw_color = unicolors[i % len(unicolors)]
                rgb_color = np.asarray(mcolors.to_rgb(raw_color))
                
                pc.paint_uniform_color(rgb_color)

            else:
                pts = np.asarray(pc.points)
                z = pts[:, 2]

                z_min = z.min()
                z_max = z.max()

                if z_max - z_min == 0:
                    z_norm = np.zeros_like(z)
                else:
                    z_norm = (z - z_min) / (z_max - z_min)

                cmap = plt.get_cmap(colors)  # you can also try "turbo"
                rgb = cmap(z_norm)[:, :3]
                pc.colors = o3d.utility.Vector3dVector(rgb)
                
    elif isinstance(colors, list):
        if len(colors) != num_pcs:
            raise ValueError("Length of colors list must match number of point clouds.")
        
        for pc, col in zip(point_clouds, colors):
            assert col.shape[0] == len(pc.points), "[WARNING COLORED] Color array length must match number of points in the point cloud."
            pc.colors = o3d.utility.Vector3dVector(col)
        
    return point_clouds

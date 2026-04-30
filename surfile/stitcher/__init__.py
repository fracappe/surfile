"""
This submodule provides classes and functions for stitching together multiple 3D point clouds or surfaces.

It includes methods for:
- `TransformParams`: Handling 3D rigid body transformations (translation and rotation).
- `Isolator`: Identifying and extracting overlapping regions between point clouds using various strategies (geometrical, max-min, KDTree, manual selection).
- `Thresholder`: Determining appropriate distance thresholds for registration algorithms.
- `SurfaceStitcher`: Implementing different stitching algorithms:
    - `stitchSavedTransforms`: Applying pre-saved transformations.
    - `stitchManual`: Manual point selection for alignment.
    - `stitchRobot`: Using robot pose data for initial alignment.
    - `stitchRMSE`: Optimizing alignment by minimizing Root Mean Square Error (RMSE) using Bayesian optimization.
    - `stitchICP`: Refining alignment using the Iterative Closest Point (ICP) algorithm.
    - `stitchFGR`: Global registration using Fast Global Registration (FGR).
    - `stitchCorrelation`: Aligning surfaces using phase cross-correlation.

The submodule also provides utility functions for point cloud manipulation, conversion between data types (NumPy arrays, Open3D point clouds, `surfile.surface` objects), and visualization.
"""

__docformat__ = 'numpy'
__name__ = 'surfile.stitcher'
"""
evaluation_pipeline.py
======================

Builds and runs a full stitching evaluation pipeline that:

  1. Runs ``stitchManual`` as the mandatory coarse alignment root step.
  2. Attaches one child ``PipelineStep`` per valid (stitcher, isolator) combination,
     which refines the manual alignment.
  3. Collects all leaf results and evaluates them with both ``BallQuery`` and
     ``DensityMapPosterior`` comparators.

Valid combinations
------------------
- ``stitchRMSE``        : maxmin, KDTree, manual, convex_hull
- ``stitchICP``         : maxmin, KDTree, manual, convex_hull
- ``stitchFGR``         : maxmin, KDTree, manual, convex_hull
- ``stitchCorrelation`` : maxmin, KDTree, manual, convex_hull  (2D point clouds only)
- ``stitchManual``      : manual isolator only (root step, not a refinement leaf)
- ``stitchRobot``       : no isolator, requires a robot file (excluded here)

Tunable constants are grouped at the top of ``build_evaluation_pipeline`` so they
are easy to find and adjust.
"""

from datetime import datetime

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from surfile.stitcher import stitcher as sst
from surfile.stitcher import comparator
import surfile.stitcher.pipeline as spipe

import numpy as np


# ---------------------------------------------------------------------------
# Isolator factory helpers
# ---------------------------------------------------------------------------

def _make_maxmin_isolator() -> sst.Isolator:
    """Isolator based on axis-aligned bounding box overlap (x and y axes)."""
    return sst.Isolator(type='maxmin', axes='xy')


def _make_kdtree_isolator() -> sst.Isolator:
    """Isolator based on mutual KDTree nearest-neighbour distance."""
    return sst.Isolator(type='KDTree', max_distance=[0.5])


def _make_manual_isolator() -> sst.Isolator:
    """Isolator that asks the user to select the overlapping region manually."""
    return sst.Isolator(type='manual')


def _make_convex_hull_isolator() -> sst.Isolator:
    """Isolator based on the convex hull of the overlapping region."""
    return sst.Isolator(type='convex_hull')


# Isolators available for all refinement stitchers (excludes 'geometrical').
_REFINEMENT_ISOLATOR_FACTORIES = {
    'maxmin':       _make_maxmin_isolator,
    'KDTree':       _make_kdtree_isolator,
    'convex_hull':  _make_convex_hull_isolator,
}


# ---------------------------------------------------------------------------
# Refinement step factories  (one per stitching method)
# ---------------------------------------------------------------------------

def _build_rmse_steps(n_calls: int, bplt: bool) -> list[spipe.PipelineStep]:
    """
    One ``stitchRMSE`` child step per available isolator.

    Parameters
    ----------
    n_calls : int
        Number of Bayesian optimisation calls.
    bplt : bool
        Whether to show plots during execution.
    """
    steps = []
    for isolator_name, isolator_factory in _REFINEMENT_ISOLATOR_FACTORIES.items():
        leaf_name = f"rmse_{isolator_name}"
        step = spipe.PipelineStep(
            sst.SurfaceStitcher.stitchRMSE,
            name=leaf_name,
            n_calls=n_calls,
            isolator=isolator_factory(),
            bplt=bplt,
        )
        steps.append(step)
    return steps


def _build_icp_steps(thresholder: sst.Thresholder, bplt: bool) -> list[spipe.PipelineStep]:
    """
    One ``stitchICP`` child step per available isolator.

    Parameters
    ----------
    thresholder : sst.Thresholder
        Shared thresholder instance used to determine max correspondence distance.
    bplt : bool
        Whether to show plots during execution.
    """
    steps = []
    for isolator_name, isolator_factory in _REFINEMENT_ISOLATOR_FACTORIES.items():
        leaf_name = f"icp_{isolator_name}"
        step = spipe.PipelineStep(
            sst.SurfaceStitcher.stitchICP,
            name=leaf_name,
            thresholder=thresholder,
            isolator=isolator_factory(),
            bplt=bplt,
        )
        steps.append(step)
    return steps


def _build_fgr_steps(voxel_size: float, thresholder: sst.Thresholder, bplt: bool) -> list[spipe.PipelineStep]:
    """
    One ``stitchFGR`` child step per available isolator.

    Parameters
    ----------
    voxel_size : float
        Voxel size for downsampling before FPFH feature computation.
    thresholder : sst.Thresholder
        Shared thresholder instance used to determine correspondence distance.
    bplt : bool
        Whether to show plots during execution.
    """
    steps = []
    for isolator_name, isolator_factory in _REFINEMENT_ISOLATOR_FACTORIES.items():
        leaf_name = f"fgr_{isolator_name}"
        step = spipe.PipelineStep(
            sst.SurfaceStitcher.stitchFGR,
            name=leaf_name,
            voxel_size=voxel_size,
            thresholder=thresholder,
            isolator=isolator_factory(),
            bplt=bplt,
        )
        steps.append(step)
    return steps


def _build_correlation_steps(dx: float, dy: float, bplt: bool) -> list[spipe.PipelineStep]:
    """
    One ``stitchCorrelation`` child step per available isolator.

    Only use this when the point clouds are 2D height maps.

    Parameters
    ----------
    dx, dy : float
        Grid spacing used when converting point clouds to surfaces.
    bplt : bool
        Whether to show plots during execution.
    """
    steps = []
    for isolator_name, isolator_factory in _REFINEMENT_ISOLATOR_FACTORIES.items():
        leaf_name = f"correlation_{isolator_name}"
        step = spipe.PipelineStep(
            sst.SurfaceStitcher.stitchCorrelation,
            name=leaf_name,
            dx=dx,
            dy=dy,
            isolator=isolator_factory(),
            bplt=bplt,
        )
        steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_evaluation_pipeline(
    bplt: bool = False,
    include_correlation: bool = False,
) -> spipe.TreePipeline:
    """
    Build the full evaluation pipeline.

    The pipeline has the following structure::

        stitchManual (root, no name)
        ├── stitchRMSE       + {maxmin, KDTree, manual, convex_hull}
        ├── stitchICP        + {maxmin, KDTree, manual, convex_hull}
        ├── stitchFGR        + {maxmin, KDTree, manual, convex_hull}
        └── stitchCorrelation + {maxmin, KDTree, manual, convex_hull}  [optional]

    ``stitchManual`` is intentionally unnamed because it is an internal node:
    its results are not saved, only forwarded to the refinement children.

    Parameters
    ----------
    bplt : bool
        If True, each step shows its visualisation during execution. Set to
        False for unattended batch runs.
    include_correlation : bool
        Set to True only when point clouds are 2D height maps. Defaults to
        False to avoid errors on 3D data.

    Returns
    -------
    spipe.TreePipeline
        A configured pipeline ready to call ``.run(pcs, folder_path)``.
    """

    # ------------------------------------------------------------------
    # Tunable constants — adjust these before running
    # ------------------------------------------------------------------
    RMSE_N_CALLS    = 10
    ICP_THRESHOLD_TYPE = 'KDTree'
    FGR_VOXEL_SIZE  = 0.05
    FGR_THRESHOLD_TYPE = 'KDTree'
    CORRELATION_DX  = 0.01
    CORRELATION_DY  = 0.01
    # ------------------------------------------------------------------

    manual_root = spipe.PipelineStep(
        sst.SurfaceStitcher.stitchManual,
        # No name: this is an internal node, its result is not saved to disk.
        bplt=bplt,
    )

    refinement_steps = [
        # *_build_rmse_steps(
        #     n_calls=RMSE_N_CALLS,
        #     bplt=bplt,
        # ),
        *_build_icp_steps(
            thresholder=sst.Thresholder(type=ICP_THRESHOLD_TYPE),
            bplt=bplt,
        ),
        *_build_fgr_steps(
            voxel_size=FGR_VOXEL_SIZE,
            thresholder=sst.Thresholder(type=FGR_THRESHOLD_TYPE),
            bplt=bplt,
        ),
    ]

    if include_correlation:
        refinement_steps.extend(
            _build_correlation_steps(
                dx=CORRELATION_DX,
                dy=CORRELATION_DY,
                bplt=bplt,
            )
        )

    for step in refinement_steps:
        manual_root.add_child(step)

    return spipe.TreePipeline(
        root_steps=[manual_root],
        name="full_evaluation_pipeline",
    )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    pcs: list[np.ndarray],
    folder_path: str,
    bplt_pipeline: bool = False,
    bplt_comparators: bool = False,
    noise_threshold: float = 0.05,
    include_correlation: bool = False,
) -> None:
    """
    Build the pipeline, run it, and evaluate results with both comparators.

    Parameters
    ----------
    pcs : list[np.ndarray]
        List of raw point clouds to stitch.
    folder_path : str
        Root folder for saving and loading pipeline results.
    bplt_pipeline : bool
        Whether to show plots during each stitching step.
    bplt_comparators : bool
        Whether to show comparator plots after evaluation.
    noise_threshold : float
        Threshold passed to ``plot_deltas`` and ``plot_histograms`` for RMSE
        outlier annotation.
    include_correlation : bool
        Forward to ``build_evaluation_pipeline``; enable only for 2D data.
    """
    pipeline = build_evaluation_pipeline(
        bplt=bplt_pipeline,
        include_correlation=include_correlation,
    )

    stitching_results = pipeline.run(pcs, folder_path, bplt=bplt_pipeline)

    # use the pipeline timestamp so all outputs go inside the same pipeline folder
    timestamp = pipeline.timestamp

    # Keep the original selected folder as the comparator output root
    original_root = os.path.dirname(folder_path) if os.path.isfile(folder_path) else folder_path

    # Create a separate pipeline folder where pipeline artifacts are stored
    pipeline_folder = os.path.join(original_root, 'pipelines', f"{pipeline.name}_{pipeline.timestamp}")
    os.makedirs(pipeline_folder, exist_ok=True)

    # Place comparator PKL files directly in the selected folder under 'comparator_deltas'
    base_save_path_ball = os.path.join(
        original_root,
        'comparator_deltas',
        f"ball_query_{timestamp}.pkl"
    )
    base_save_path_DMP = os.path.join(
        original_root,
        'comparator_deltas',
        f"density_map_{timestamp}.pkl"
    )

    _run_ball_query_evaluation(stitching_results, noise_threshold, base_save_path_ball, bplt_comparators)
    _run_density_map_evaluation(stitching_results, noise_threshold, base_save_path_DMP, bplt_comparators)


def _run_ball_query_evaluation(
    stitching_results: dict,
    noise_threshold: float,
    save_path: str,
    bplt: bool,
) -> None:
    """Evaluate stitching results using the BallQuery comparator."""
    save_folder = os.path.dirname(save_path)
    result_names = ", ".join(str(k) for k in stitching_results.keys()) if stitching_results else "<no results>"
    print("\n[INFO EVAL] Running BallQuery comparator...")
    print(f"[INFO EVAL] Comparator output folder: {save_folder}")
    print(f"[INFO EVAL] Stitcher result keys: {result_names}")
    bq = comparator.BallQuery(save_path, stitching_results)
    bq.print_summary()
    if bplt:
        bq.plot_deltas(noise_threshold=noise_threshold)
        bq.plot_histograms(noise_threshold=noise_threshold)


def _run_density_map_evaluation(
    stitching_results: dict,
    noise_threshold: float,
    save_path: str,
    bplt: bool,
) -> None:
    """Evaluate stitching results using the DensityMapPosterior comparator."""
    save_folder = os.path.dirname(save_path)
    result_names = ", ".join(str(k) for k in stitching_results.keys()) if stitching_results else "<no results>"
    print("\n[INFO EVAL] Running DensityMapPosterior comparator...")
    print(f"[INFO EVAL] Comparator output folder: {save_folder}")
    print(f"[INFO EVAL] Stitcher result keys: {result_names}")
    dmp = comparator.DensityMapPosterior(save_path, stitching_results)
    dmp.print_summary()
    if bplt:
        dmp.plot_deltas(noise_threshold=noise_threshold)
        dmp.plot_histograms(noise_threshold=noise_threshold)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from stitch_pipeline import open_files  # adjust import to your project

    folder = 'G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth'
    pcs, folder_path = open_files(folder=folder, downsample=3, userscales=[1, 1, 1])

    run_evaluation(
        pcs=pcs,
        folder_path=folder_path,
        bplt_pipeline=False,
        bplt_comparators=False,
        noise_threshold=0.05,
        include_correlation=False,
    )
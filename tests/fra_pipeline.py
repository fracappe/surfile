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
- ``stitchRMSE``        : maxmin, KDTree, convex_hull
- ``stitchICP``         : maxmin, KDTree, convex_hull
- ``stitchFGR``         : maxmin, KDTree, convex_hull
- ``stitchCorrelation`` : maxmin, KDTree, convex_hull  (2D point clouds only)
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
from pathlib import Path


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
            bplt=bplt
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
    manual_step_name: str = "pt_manual",
) -> spipe.TreePipeline:
    """
    Build the full evaluation pipeline.

    The pipeline has the following structure::

        stitchManual (root, no name)
        ├── stitchRMSE       + {maxmin, KDTree, convex_hull}
        ├── stitchICP        + {maxmin, KDTree, convex_hull}
        ├── stitchFGR        + {maxmin, KDTree, convex_hull}
        └── stitchCorrelation + {maxmin, KDTree, convex_hull}  [optional]

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
    manual_step_name : str
        Name used to identify and recall the manual alignment step. Only used
        when use_cached_manual is True.

    Returns
    -------
    spipe.TreePipeline
        A configured pipeline ready to call ``.run(pcs, folder_path)``.
    """

    # ------------------------------------------------------------------
    # Tunable constants — adjust these before running
    # ------------------------------------------------------------------
    RMSE_N_CALLS    = 30
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
        recall_from_passtrough=manual_step_name
    )

    refinement_steps = [
        *_build_rmse_steps(
            n_calls=RMSE_N_CALLS,
            bplt=bplt,
        ),
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

def _find_comparator_results_folder(root_folder: str) -> list[str]:
    """
    Search for all folders starting with 'comparator_results' in the given root folder.
    
    Parameters
    ----------
    root_folder : str
        Root folder to search in.
    
    Returns
    -------
    list[str]
        List of paths to comparator_results folders, sorted by name (most recent first).
    """
    root = Path(root_folder)
    if not root.exists():
        return []
    
    folders = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith('comparator_results'):
            folders.append(str(item))
    
    # Sort in reverse order to show most recent first (if timestamps are in folder names)
    return sorted(folders, reverse=True)


def _ask_select_comparator_folder(folders: list[str]) -> str | None:
    """
    Let user select from a list of comparator_results folders.
    
    Parameters
    ----------
    folders : list[str]
        List of folder paths to choose from.
    
    Returns
    -------
    str or None
        Selected folder path, or None if user cancels.
    """
    print("\n[INFO] Found multiple comparator results folders:")
    for idx, folder in enumerate(folders, 1):
        print(f"  {idx}. {Path(folder).name}")
    
    while True:
        try:
            choice = input(f"Select a folder (1-{len(folders)}) or press 'n' to run new evaluation: ").strip().lower()
            if choice == 'n':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(folders):
                return folders[choice_idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(folders)}, or 'n'.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(folders)}, or 'n'.")


def _ask_use_existing_results(comparator_folder: str) -> bool:
    """
    Ask user if they want to use existing comparator results or run new evaluation.
    
    Parameters
    ----------
    comparator_folder : str
        Path to the existing comparator_results folder.
    
    Returns
    -------
    bool
        True if user wants to use existing results, False to run new evaluation.
    """
    print(f"\n[INFO] Found existing comparator results folder: {Path(comparator_folder).name}")
    response = input("Do you want to use this folder (y) or run new evaluation (n)? [y/n]: ").strip().lower()
    return response in ('y', 'yes')

def run_evaluation(
    pcs: list[np.ndarray],
    folder_path: str,
    bplt_pipeline: bool = False,
    bplt_comparators: bool = False,
    noise_threshold: float = 0.05,
    include_correlation: bool = False,
    manual_step_name: str = "pt_manual",
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
    manual_step_name : str
        Name used to identify and recall the manual alignment step.
    """
    # Check if comparator_results folder already exists
    original_root = os.path.dirname(folder_path) if os.path.isfile(folder_path) else folder_path
    existing_comparator_folders = _find_comparator_results_folder(original_root)
    comparator_folder = None
    use_existing = False
    
    if existing_comparator_folders:
        if len(existing_comparator_folders) == 1:
            # Only one folder found, ask if user wants to use it
            comparator_folder = existing_comparator_folders[0]
            use_existing = _ask_use_existing_results(comparator_folder)
        else:
            # Multiple folders found, let user choose
            comparator_folder = _ask_select_comparator_folder(existing_comparator_folders)
            use_existing = comparator_folder is not None
    
    # If not using existing results, run the pipeline
    if not use_existing:
        pipeline = build_evaluation_pipeline(
            bplt=bplt_pipeline,
            include_correlation=include_correlation,
            manual_step_name=manual_step_name,
        )

        stitching_results = pipeline.run(pcs, folder_path, bplt=bplt_pipeline)
        timestamp = pipeline.timestamp
        
    else:
        # If using existing results, we don't need to run the pipeline
        stitching_results = None
        timestamp = None
    
    # Create folder paths
    if not use_existing:
        pipeline_folder = os.path.join(original_root, 'pipelines', f"{pipeline.name}_{pipeline.timestamp}")
        if not getattr(pipeline, 'used_past_run', False):
            os.makedirs(pipeline_folder, exist_ok=True)

        comparator_folder = os.path.join(original_root, 'comparator_results')
        os.makedirs(comparator_folder, exist_ok=True)

        # Create new save paths
        base_save_path_ball = os.path.join(
            comparator_folder,
            f"ball_query_{timestamp}.pkl"
        )
        base_save_path_DMP = os.path.join(
            comparator_folder,
            f"density_map_{timestamp}.pkl"
        )
    else:
        base_save_path_ball = None
        base_save_path_DMP = None
    
    summary_ball, fig1_ball, fig2_ball, fig3_ball = _run_ball_query_evaluation(
        stitching_results=stitching_results if not use_existing else {},
        noise_threshold=noise_threshold,
        save_path=base_save_path_ball,
        bplt=bplt_comparators,
        comparator_folder=comparator_folder
    )
    summary_DMP, fig1_DMP, fig2_DMP, _ = _run_density_map_evaluation(
        stitching_results=stitching_results if not use_existing else {},
        save_path=base_save_path_DMP,
        bplt=bplt_comparators,
        comparator_folder=comparator_folder
    )

    return {
        "ball_query": {
            "summary": summary_ball,
            "fig_deltas": fig1_ball,
            "fig_histograms": fig2_ball,
            "fig_colormap": fig3_ball,
        },
        "density_map_posterior": {
            "summary": summary_DMP,
            "fig_deltas": fig1_DMP,
            "fig_histograms": fig2_DMP,
        }
    }


def _run_ball_query_evaluation(
    stitching_results: dict,
    noise_threshold: float,
    save_path: str,
    bplt: bool,
    comparator_folder: str | None = None,
) -> tuple:
    """Evaluate stitching results using the BallQuery comparator.
    
    If ``stitching_results`` is empty and ``comparator_folder`` is provided,
    the most recent pickled BallQuery file found in that folder is loaded
    instead of running a new evaluation.

    Parameters
    ----------
    stitching_results : dict
        Dictionary of stitching results from pipeline.run().
    noise_threshold : float
        Threshold for RMSE annotation.
    save_path : str
        Path where to save BallQuery pickle file.
    bplt : bool
        Whether to show plots.
    comparator_folder : str, optional
        If provided, try to load BallQuery from this folder instead of computing.
    
    Returns
    -------
    tuple[dict | None, Figure | None, Figure | None, Figure | None]
        (summary, fig_deltas, fig_histograms, fig_colormap)
    """
    
    summary, fig1, fig2, fig3 = None, None, None, None

    
    bq_files = []
    if not stitching_results and comparator_folder:
        print("\n[INFO EVAL] Attempting to load BallQuery from existing folder...")
        results_folder = Path(comparator_folder)
        bq_files = list(results_folder.glob("ball_query_*.pkl"))
        
        if bq_files:
            bq_file = sorted(bq_files)[-1]
            try:
                bq, summary, fig1, fig2, fig3 = comparator.Comparator.load_ball_query(comparator_folder, bq_file.name, bplt=bplt)
                print(f"[INFO EVAL] Loaded BallQuery from: {bq_file.name}")
                return summary, fig1, fig2, fig3
            except Exception as e:
                print(f"[WARN EVAL] Failed to load BallQuery: {e}")
                print("[INFO EVAL] Proceeding with new evaluation...")
    
    if not stitching_results:
        print("[INFO EVAL] No stitching results available and no existing BallQuery results found.")
        return summary, fig1, fig2, fig3
    
    if not bq_files:
        save_folder = os.path.dirname(save_path)
        result_names = ", ".join(str(k) for k in stitching_results.keys())
        print("\n[INFO EVAL] Running BallQuery comparator...")
        print(f"[INFO EVAL] Comparator output folder: {save_folder}")
        print(f"[INFO EVAL] Stitcher result keys: {result_names}")

        bq = comparator.BallQuery(save_path, stitching_results, bplt=False)
        summary = bq.print_summary()
        if bplt:
            fig1 = bq.plot_deltas(noise_threshold=noise_threshold)
            fig2 = bq.plot_histograms(noise_threshold=noise_threshold)
            fig3 = bq.colormap_deltas()
        return summary, fig1, fig2, fig3


def _run_density_map_evaluation(
    stitching_results: dict,
    save_path: str | None,
    bplt: bool,
    comparator_folder: str | None = None,
) -> tuple:
    """Evaluate stitching results using the DensityMapPosterior comparator.
    
    If ``stitching_results`` is empty and ``comparator_folder`` is provided,
    the most recent pickled DMP file found in that folder is loaded instead of
    running a new evaluation.

    Parameters
    ----------
    stitching_results : dict
        Dictionary of stitching results from ``TreePipeline.run()``.
        Pass an empty dict to trigger loading from disk.
    save_path : str or None
        Filesystem path where the freshly computed DMP pickle is written.
        Pass None only when loading from disk.
    bplt : bool
        Whether to show plots.
    comparator_folder : str or None
        Folder to search for existing pickled results when ``stitching_results``
        is empty.

    Returns
    -------
    tuple[dict | None, Figure | None, dict[str, Figure] | None, None]
        (summary, fig_deltas, fig_histograms, None)
        The last element is always None and exists only to keep the return
        signature uniform with _run_ball_query_evaluation.
    """
    
    summary, fig1, fig2 = None, None, None

    if not stitching_results and comparator_folder:
        print("\n[INFO EVAL] Attempting to load DensityMapPosterior from existing folder...")
        results_folder = Path(comparator_folder)
        dmp_files = list(results_folder.glob("density_map_*.pkl"))
        
        if dmp_files:
            dmp_file = sorted(dmp_files)[-1]
            try:
                dmp, summary, fig1, fig2 = comparator.Comparator.load_density_map_posterior(comparator_folder, dmp_file.name, bplt=bplt)
                print(f"[INFO EVAL] Loaded DensityMapPosterior from: {dmp_file.name}")
                return summary, fig1, fig2, None
            except Exception as e:
                print(f"[WARN EVAL] Failed to load DensityMapPosterior: {e}")
                print("[INFO EVAL] Proceeding with new evaluation...")
    
    if not stitching_results:
        print("[INFO EVAL] No stitching results available and no existing DMP results found.")
        return summary, fig1, fig2, None

    save_folder = os.path.dirname(save_path)
    result_names = ", ".join(str(k) for k in stitching_results.keys())
    print("\n[INFO EVAL] Running DensityMapPosterior comparator...")
    print(f"[INFO EVAL] Comparator output folder: {save_folder}")
    print(f"[INFO EVAL] Stitcher result keys: {result_names}")

    dmp = comparator.DensityMapPosterior(save_path, stitching_results, bplt=False)
    summary = dmp.print_summary()
    if bplt:
        fig1 = dmp.plot_deltas()
        fig2 = dmp.plot_histograms()

    return summary, fig1, fig2, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from stitch_pipeline import open_files  # adjust import to your project

    folder = 'G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth'
    pcs, folder_path = open_files(folder=folder, downsample=3, userscales=[1, 1, 1])

    results = run_evaluation(
        pcs=pcs,
        folder_path=folder_path,
        bplt_pipeline=False,
        bplt_comparators=True,
        include_correlation=False,
        manual_step_name="pt_manual"
    )
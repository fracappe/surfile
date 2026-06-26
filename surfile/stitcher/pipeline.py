"""
pipeline.py
===========

Implements TreePipeline and Complete Pipeline builders and runners

Tree Pipeline
-------------
TODO

MultilevelPipelineBuilder
-------------------------
    Builds and runs a multi-level stitching evaluation pipeline in two modes:

    Mode 1 — Progressive Methods (fixed isolator)
    ----------------------------------------------
    Each branch of the tree fixes one isolator type and chains stitching methods
    in sequence.  The root fans out one branch per permutation of (starting
    stitcher, isolator).  With ``max_depth=2`` and three stitchers
    {RMSE, ICP, FGR} and three isolators {maxmin, KDTree, convex_hull}, the root
    spawns 3 × 3 × 2 = 18 branches (all ordered pairs of distinct stitchers ×
    all isolators).

    Example with max_depth=2::

        stitchManual (root)
        ├── stitchRMSE+maxmin  → stitchICP+maxmin
        ├── stitchRMSE+KDTree  → stitchICP+KDTree
        ├── stitchRMSE+maxmin  → stitchFGR+maxmin
        ...
        ├── stitchICP+maxmin   → stitchRMSE+maxmin
        ...

    Mode 2 — Progressive Isolators (fixed method)
    ----------------------------------------------
    Each branch fixes one stitching method and chains isolators in sequence.
    The root fans out one branch per permutation of (starting isolator, stitcher).

    Example with max_depth=2:

        stitchManual (root)
        ├── stitchRMSE+maxmin   → stitchRMSE+KDTree
        ├── stitchRMSE+maxmin   → stitchRMSE+convex_hull
        ├── stitchRMSE+KDTree   → stitchRMSE+maxmin
        ...
        ├── stitchICP+maxmin    → stitchICP+KDTree

    Permutation semantics
    ---------------------
    ``max_depth`` sets the length of the chain *after* the manual root step, i.e.
    the number of refinement steps.  All ordered sequences of length ``max_depth``
    drawn *without replacement* from the available pool are enumerated.  In Mode 1
    the pool is the set of stitcher functions; in Mode 2 it is the set of isolator
    factories.

MultilevelEvaluationRunner
--------------------------
    Contructs a MultilevelPipelineBuilder and executes it to apply comparators

    ``BallQuery`` and ``DensityMapPosterior`` comparators can be evaluated either
    only at leaf nodes (``evaluate_intermediate=False``) or at every node that
    produces a stitching result (``evaluate_intermediate=True``).

Reuse levels
------------
When ``MultilevelEvaluationRunner.run()`` is called, the user is interactively
asked which level of reuse they want:

  1. **Full rerun** — re-stitch everything from scratch and rerun both
     comparators.  ``TreePipeline``'s own past-run prompt is intentionally
     bypassed; the pipeline always starts fresh.
  2. **Reuse stitching, rerun comparators** — select a past pipeline run
     folder; its saved transforms are loaded via
     ``TreePipeline._run_past_pipeline()`` to reconstruct the stitched clouds,
     then BallQuery + DMP are rerun on them.
  3. **Reuse everything** — load existing comparator pickles and show plots
     only, no computation performed.
"""

from __future__ import annotations
import itertools
from typing import Callable, Sequence
import pathlib
import inspect
import os
import pickle

import surfile.stitcher.stitcher as sstitcher
import surfile.stitcher.comparator as scomp

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def _prompt_user_for_run_choice(past_runs):
        for i, run in enumerate(past_runs):
            print(f"{i}: {run}")
        print(f"{len(past_runs)}: Run new")
        
        while True:
            choice = input(f"Select a past run to apply or run new (0-{len(past_runs)}): ")
            if choice.isdigit():
                choice_idx = int(choice)
                if 0 <= choice_idx <= len(past_runs):
                    return past_runs[choice_idx] if choice_idx < len(past_runs) else None
            print("Invalid input. Please enter a number corresponding to the options above.")


class PipelineStep:
    f: callable
    
    def __init__(self, f, name, recall_from_passtrough=None, **kwargs):
        self.f = f
        self.name = name
        self.kwargs = kwargs
        self.pt_name = recall_from_passtrough
        self.children = []
        
    def add_child(self, step):
        if not isinstance(step, PipelineStep):
            raise ValueError("Il figlio deve essere un'istanza di PipelineStep")
        self.children.append(step)
        return step
    
    def _check_arguments(self):
        """
        Check if the provided arguments match the function's signature.
        If kwargs has parameters that are not in the function signature gives a warning and continues
        """
        if not callable(self.f):
            raise ValueError(f"The provided function {self.f} is not callable.")
        
        signature = inspect.signature(self.f)
        for param in signature.parameters.values():
            if param.name == 'point_clouds':
                continue  # This is the expected input argument for stitching functions
            if param.name not in self.kwargs and param.default is param.empty:
                raise ValueError(f"Missing required argument '{param.name}' for function '{self.f.__name__}'")
            
        for kwarg in self.kwargs:
            if kwarg not in signature.parameters:
                print(f'[WARN PIPELINE] Argument "{kwarg}" is not in the signature of function "{self.f.__name__}". It will be ignored.')
                
    def _check_all_leaves_have_name(self):
        if not self.children:
            if not self.name:
                raise ValueError("All leaf nodes must have a name for saving results.")
        else:
            for child in self.children:
                child._check_all_leaves_have_name()

    def attach_pt(self):
        if len(self.children) > 0:
            if self.name is not None:
                self.children.append(PipelineStep.pass_through('intermediate_' + self.name))
            else:
                print('[INFO PIPESTEP] Could not add passtrough to step with no name')

        for child in self.children:
            child.attach_pt()
    
    def run(self, pcs, current_transforms, base_save_path=None, bplt_override=False) -> dict[str, tuple[np.ndarray, list[np.ndarray]]]:
        """
        Execute this step and recursively all its children, collecting leaf results.

        At each node the local transform returned by the stitching function is
        composed with the accumulated global transform passed down from the
        parent: T_global = T_local @ T_parent. This ensures that each leaf
        stores the full transform chain from the original coordinate frame.

        Parameters
        ----------
        pcs : list[np.ndarray]
            Point clouds to process at this node.
        current_transforms : list[np.ndarray]
            Accumulated 4x4 global transforms from all ancestor steps.
        base_save_path : str or None
            Root folder under which leaf results are saved. If None, nothing
            is written to disk.
        bplt_override : bool
            If True, forces bplt=True on this step's function call.

        Returns
        -------
        dict[str, tuple[np.ndarray, list[np.ndarray]]]
            Mapping of leaf name -> (stitched_fixed, transformed_point_clouds),
            collected recursively from all reachable leaves.
        """
        self._check_arguments()
        self._check_all_leaves_have_name()
        if bplt_override: self.kwargs['bplt'] = bplt_override
        
        if self.pt_name is not None:
            fixed, next_pcs, local_transforms = self._recall_from_passtrough(pcs, base_save_path, self.kwargs['bplt'])
        else:
            fixed, next_pcs, local_transforms = self.f(pcs, **self.kwargs)
        
        new_global_transforms = [T_local @ T for T, T_local in zip(current_transforms, local_transforms)]

        if not self.children:
            if base_save_path is not None:
                self._save(new_global_transforms, base_save_path, self.name)
                print(f'[INFO PIPESTEP] Saving leaf result: {self.name}')
            return {self.name: (fixed, next_pcs)}

        leaf_results = {}
        for child in self.children:
            child_results = child.run(next_pcs, new_global_transforms, base_save_path, bplt_override)
            leaf_results.update(child_results)
        return leaf_results

    def _save(self, transforms, base_path, leaf_path_name):
        save_folder = os.path.join(base_path, leaf_path_name)
            
        os.makedirs(save_folder, exist_ok=False)
        for i, T in enumerate(transforms):
            with open(os.path.join(save_folder, f"{i}.pkl"), "wb") as f:
                pickle.dump(T, f)
                
    def _recall_from_passtrough(self, pcs, base_save_path, bplt=False):
        if base_save_path is None: raise ValueError("[ERROR PIPELINE] Base save path must be set for search por previous passtrough calls")
        # walk all subdirs and ask user which past passtrough to use instead
        past_pt = []
        # remove from base_save_path all after folder pipelines: ...\\pipelines\\{remove part}
        marker = 'pipelines\\'
        base_pipelines_path = base_save_path.split(marker)[0] + marker

        for dirpath, dirname, filenames in os.walk(base_pipelines_path):
            if self.pt_name in dirname:
                dir_name_past_base = dirpath.replace(base_pipelines_path, '')
                past_pt.append(dir_name_past_base)
        
        print(f'[INFO PIPELINE] Found {len(past_pt)} past runs for passtrough {self.pt_name}.')
        if past_pt:
            choice = _prompt_user_for_run_choice(past_pt)
            
            if choice is not None:
                subdir = os.path.join(base_pipelines_path, choice, self.pt_name)
                print(base_pipelines_path, choice, self.pt_name)
                print(f'[INFO PIPELINE] Running past passtrough from: {subdir}')
                return sstitcher.SurfaceStitcher.stitchSavedTransforms(pcs, subdir, bplt=bplt)
        
        print(f'[INFO PIPELINE] Starting new step: {self.name}')  
        return self.f(pcs, **self.kwargs)

    @classmethod
    def pass_through(cls, name):
        def f(point_clouds: list[np.ndarray]):
            fixed = np.vstack(point_clouds)
            return fixed, point_clouds, [np.eye(4) for _ in point_clouds]
        return cls(f, name=name)


class TreePipeline:
    def __init__(self, root_steps: list[PipelineStep], name: str):
        self.name = name
        self.root_steps = root_steps
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def run(self, pcs, save_transforms_root, bplt=True):
        # prepare result dict[str, tuple[np.ndarray, list[np.ndarray]]]
        stitching_results = {}
        self.used_past_run = False

        # if save_transforms_root is not a folder but a file only consider the base path
        save_transforms_root = os.path.dirname(save_transforms_root) if os.path.isfile(save_transforms_root) else save_transforms_root
        base_save_path = os.path.join(
            save_transforms_root,
            'pipelines',
            f"{self.name}_{self.timestamp}"
        )
        
        past_runs = self._check_for_past_runs(save_transforms_root)
        print(f'[INFO PIPELINE] Found {len(past_runs)} past runs for pipeline "{self.name}".')
        if past_runs:
            choice = _prompt_user_for_run_choice(past_runs)
            if choice is None: # make a new run
                pass

            else: # apply existing
                self.used_past_run = True
                print(f'[INFO PIPELINE] Applying past pipeline results from: {choice}')
                stitching_results = self._run_past_pipeline(pcs, os.path.join(save_transforms_root, 'pipelines', choice), bplt=bplt)
                return stitching_results

        initial_transforms = [np.eye(4) for _ in pcs]
        print(f'[INFO PIPELINE] Starting new Tree Pipeline: {self.name}')
        for root in self.root_steps:
            root_results = root.run(pcs, initial_transforms, base_save_path, bplt_override=bplt)
            stitching_results.update(root_results)
        return stitching_results
            
    def _check_for_past_runs(self, save_transforms_root):        
        past_runs = []
        if not os.path.exists(os.path.join(save_transforms_root, 'pipelines')): 
            return past_runs
        for subdir in os.listdir(os.path.join(save_transforms_root, 'pipelines')):
            if subdir.startswith(self.name):
                past_runs.append(subdir)
                
        return past_runs
            
    def _run_past_pipeline(self, pcs, past_run_folder, bplt=False):
        stitching_results = {}

        for subdir in _find_subfolders_with_prefix(past_run_folder, prefix=''):
            fixed, next_pcs, _ = sstitcher.SurfaceStitcher.stitchSavedTransforms(pcs, os.path.join(past_run_folder, subdir), bplt=bplt)
            name = os.path.basename(subdir)
            stitching_results[name] = (fixed, next_pcs)
        return stitching_results


class PipelineMode:
    """Symbolic names for the two pipeline modes."""
    PROGRESSIVE_METHODS   = "progressive_methods"
    PROGRESSIVE_ISOLATORS = "progressive_isolators"


class ReuseLevel:
    """
    Symbolic names for the three interactive reuse choices.

    FULL_RERUN
        Re-stitch everything and rerun both comparators.
    REUSE_STITCHING
        Load a past pipeline run from disk; rerun comparators only.
    REUSE_EVERYTHING
        Load existing comparator pickles and show plots only.
    """
    FULL_RERUN        = "full_rerun"
    REUSE_STITCHING   = "reuse_stitching"
    REUSE_EVERYTHING  = "reuse_everything"


class IsolatorSpec:
    def __init__(self, type, kwargs: dict):
        self.type = type
        self.kwargs = kwargs

    def build_isol(self) -> PipelineStep:
        return sstitcher.Isolator(
            type = self.type,
            **self.kwargs
        )


def build_default_isolator_specs(
    build_methods: list[str] = ['maxmin', 'KDTree', 'convex_hull'],
    KDTree_distance = [0],
    max_min_axes = 'xyz'
) -> dict[str, IsolatorSpec]:
    ispecs = {}

    if 'maxmin' in build_methods:
        ispecs.update({
            'maxmin': IsolatorSpec('maxmin', kwargs=dict(axes=max_min_axes))
        })
    if 'KDTree' in build_methods:
        ispecs.update({
            'KDTree': IsolatorSpec('KDTree', kwargs=dict(max_distance=KDTree_distance))
        })
    if 'convex_hull' in build_methods:
        ispecs.update({
            'convex_hull': IsolatorSpec('convex_hull', kwargs={})
        })

    return ispecs


class StitcherSpec:
    """
    Bundles a stitcher method reference with its fixed keyword arguments.

    This avoids scattering ``n_calls``, ``voxel_size``, etc. across the
    pipeline builder and keeps each stitcher's configuration self-contained.

    Parameters
    ----------
    method : callable
        Unbound ``SurfaceStitcher`` method (e.g. ``SurfaceStitcher.stitchRMSE``).
    name : str
        Short identifier used in node names (e.g. ``"rmse"``).
    kwargs : dict
        Fixed keyword arguments forwarded to ``PipelineStep`` — everything
        *except* ``isolator``, which is injected at build time.
    """

    def __init__(self, method: Callable, name: str, kwargs: dict):
        self.method = method
        self.name   = name
        self.kwargs = kwargs

    def build_step(self, isolator: sstitcher.Isolator, node_label: str) -> PipelineStep:
        """
        Instantiate a ``PipelineStep`` for this stitcher with the given isolator.

        Parameters
        ----------
        isolator : sst.Isolator
            Isolator instance to inject.
        node_label : str
            Unique name for the pipeline node (used for logging and result keys).

        Returns
        -------
        PipelineStep
        """
        return PipelineStep(
            self.method,
            name=node_label,
            isolator=isolator,
            **self.kwargs,
        )


def build_default_stitcher_specs(
    build_methods: list[str] = ['rmse', 'icp', 'fgr'],
    rmse_n_calls: int       = 50,   
    icp_threshold_type: str = 'KDTree',
    fgr_voxel_size: float   = 0.1,    
    fgr_threshold_type: str = 'KDTree',
    bplt: bool              = False,
) -> dict[str, StitcherSpec]:
    """
    Build the default registry of stitcher specifications.

    Returns a dict keyed by the short stitcher name.  Pass this dict to
    ``MultilevelPipelineBuilder`` to customise which stitchers participate.

    Parameters
    ----------
    build_methods : list[str]
        Defines the types of stitching methods to build ['rmse', 'icp', 'fgr']
    rmse_n_calls : int
        Number of Bayesian optimisation calls for ``stitchRMSE``.
    icp_threshold_type : str
        Thresholder type for ``stitchICP``.
    fgr_voxel_size : float
        Voxel size for FPFH feature computation in ``stitchFGR``.
    fgr_threshold_type : str
        Thresholder type for ``stitchFGR``.
    bplt : bool
        Whether each step shows visualisations during execution.

    Returns
    -------
    dict[str, StitcherSpec]
    """
    sspecs = {}

    if 'rmse' in build_methods:
        sspecs.update({
            'rmse': StitcherSpec(
            method=sstitcher.SurfaceStitcher.stitchRMSE,
            name='rmse',
            kwargs=dict(n_calls=rmse_n_calls, bplt=bplt))
        })
    if 'icp' in build_methods:
        sspecs.update({
            'icp': StitcherSpec(
                method=sstitcher.SurfaceStitcher.stitchICP,
                name='icp',
                kwargs=dict(thresholder=sstitcher.Thresholder(type=icp_threshold_type), bplt=bplt))
        })
    if 'fgr' in build_methods:
        sspecs.update({
            'fgr': StitcherSpec(
                method=sstitcher.SurfaceStitcher.stitchFGR,
                name='fgr',
                kwargs=dict(
                    voxel_size=fgr_voxel_size,
                    thresholder=sstitcher.Thresholder(type=fgr_threshold_type),
                    bplt=bplt))
        })

    return sspecs


def _node_label(stitcher_name: str, isolator_name: str, depth: int, parent_label: str = "") -> str:
    """
    Produce a unique, readable node label that accumulates chain history.

    Example: ``"d1_rmse_maxmin"``, ``"d1_rmse_maxmin--d2_icp_maxmin"``.
    """
    current_step = f"d{depth}_{stitcher_name}_{isolator_name}"
    
    if parent_label:
        return f"{parent_label}--{current_step}"
    
    return current_step


def _build_chain_progressive_methods(
    stitcher_sequence: list[str],
    isolator_name: str,
    isolator_specs: dict[str, IsolatorSpec],
    stitcher_specs: dict[str, StitcherSpec],
) -> list[PipelineStep]:
    """
    Build a linear chain of ``PipelineStep`` objects for Mode 1.

    Each step in the chain applies the next stitcher in ``stitcher_sequence``
    using the same ``isolator_name`` throughout.  The steps are returned as a
    flat list; the caller is responsible for linking them (parent → child).

    Parameters
    ----------
    stitcher_sequence : list[str]
        Ordered stitcher names, e.g. ``['rmse', 'icp']``.
    isolator_name : str
        Fixed isolator for every step in this chain.
    stitcher_specs : dict[str, StitcherSpec]
        Registry of available stitchers.

    Returns
    -------
    list[PipelineStep]
        Steps in chain order; index 0 is the first child of the root.
    """
    chain = []

    # Track the pipeline path so folders retain their history
    cumulative_label = ""

    for depth, stitcher_name in enumerate(stitcher_sequence, start=1):
        cumulative_label = _node_label(stitcher_name, isolator_name, depth, cumulative_label)   

        step = stitcher_specs[stitcher_name].build_step(
            isolator=isolator_specs[isolator_name].build_isol(),
            node_label=cumulative_label,
        )
        chain.append(step)

    return chain


def _build_chain_progressive_isolators(
    isolator_sequence: list[str],
    stitcher_name: str,
    stitcher_specs: dict[str, StitcherSpec],
    isolator_specs: dict[str, IsolatorSpec],
) -> list[PipelineStep]:
    """
    Build a linear chain of ``PipelineStep`` objects for Mode 2.

    Each step applies the same stitcher with the next isolator in
    ``isolator_sequence``.

    Parameters
    ----------
    isolator_sequence : list[str]
        Ordered isolator names, e.g. ``['maxmin', 'KDTree']``.
    stitcher_name : str
        Fixed stitcher for every step in this chain.
    stitcher_specs : dict[str, StitcherSpec]
        Registry of available stitchers.

    Returns
    -------
    list[PipelineStep]
        Steps in chain order; index 0 is the first child of the root.
    """
    chain = []

    # Track the pipeline path so folders retain their history
    cumulative_label = ""

    for depth, isolator_name in enumerate(isolator_sequence, start=1):
        cumulative_label = _node_label(stitcher_name, isolator_name, depth, cumulative_label)

        step  = stitcher_specs[stitcher_name].build_step(
            isolator=isolator_specs[isolator_name].build_isol(),
            node_label=cumulative_label,
        )
        chain.append(step)

    return chain


def _link_chain_to_parent(
    parent: PipelineStep,
    chain: list[PipelineStep],
) -> None:
    """
    Attach a linear chain of steps as a single path descending from ``parent``.

    Each element of ``chain`` becomes the sole child of the previous one::

        parent → chain[0] → chain[1] → ... → chain[-1]

    Parameters
    ----------
    parent : PipelineStep
        The step that will receive ``chain[0]`` as its child.
    chain : list[PipelineStep]
        Ordered sequence of steps to link.
    """
    current = parent
    for step in chain:
        current.add_child(step)
        current = step


class MultilevelPipelineBuilder:
    """
    Constructs a multi-level ``TreePipeline`` in one of two modes.

    Both modes share the same entry point: ``build()``.  The returned pipeline
    is ready to call ``.run(pcs, folder_path)``.

    Parameters
    ----------
    mode : str
        One of ``PipelineMode.PROGRESSIVE_METHODS`` or
        ``PipelineMode.PROGRESSIVE_ISOLATORS``.
    max_depth : int
        Length of each refinement chain (number of steps after the manual root).
        Must be >= 1.  With ``n`` items in the pool (stitchers in Mode 1,
        isolators in Mode 2), all ordered sequences of length ``max_depth``
        drawn *without replacement* are enumerated — i.e.
        ``P(n, max_depth) = n! / (n - max_depth)!`` chains per fixed element.
    stitcher_specs : dict[str, StitcherSpec]
        Registry of stitchers that may appear in chains.  Produced by
        ``build_default_stitcher_specs()`` or assembled manually.
    isolator_names : list[str] or None
        Subset of ``ISOLATOR_REGISTRY`` keys to include.  Defaults to all
        three if not provided.
    manual_step_name : str
        Pass-through name used to identify the manual alignment result.
    pipeline_name : str
        Human-readable name embedded in the ``TreePipeline``.

    Raises
    ------
    ValueError
        If ``max_depth`` exceeds the size of the permutation pool.
    """

    def __init__(
        self,
        mode: str,
        max_depth: int,
        stitcher_specs: dict[str, StitcherSpec],
        isolator_specs: dict[str, IsolatorSpec],
        save_intermediate = False,
        manual_step_name: str = "pt_manual",
        pipeline_name: str    = "multilevel_evaluation_pipeline",
    ):
        self.mode             = mode
        self.max_depth        = max_depth
        self.stitcher_specs   = stitcher_specs
        self.isolator_specs   = isolator_specs
        self.save_intermediate = save_intermediate
        self.manual_step_name = manual_step_name
        self.pipeline_name    = pipeline_name

        self._validate()

    def build(self) -> TreePipeline:
        """
        Assemble and return the configured ``TreePipeline``.

        Returns
        -------
        TreePipeline
        """
        manual_root = self._build_manual_root()

        if self.mode == PipelineMode.PROGRESSIVE_METHODS:
            self._attach_progressive_methods_branches(manual_root)
        else:
            self._attach_progressive_isolators_branches(manual_root)

        if self.save_intermediate: manual_root.attach_pt()

        return TreePipeline(
            root_steps=[manual_root],
            name=self.pipeline_name,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        valid_modes = {PipelineMode.PROGRESSIVE_METHODS, PipelineMode.PROGRESSIVE_ISOLATORS}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")

        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")

        if self.mode == PipelineMode.PROGRESSIVE_METHODS:
            pool_size = len(self.stitcher_specs)
            pool_name = "stitcher_specs"
        else:
            pool_size = len(self.isolator_specs)
            pool_name = "isolator_names"

        if self.max_depth > pool_size:
            raise ValueError(
                f"max_depth={self.max_depth} exceeds pool size "
                f"{pool_size} ({pool_name}); cannot draw without replacement."
            )

    # ------------------------------------------------------------------
    # Root step
    # ------------------------------------------------------------------

    def _build_manual_root(self) -> PipelineStep:
        """Construct the mandatory manual-alignment root step."""
        root = PipelineStep(
            sstitcher.SurfaceStitcher.stitchManual,
            bplt=False,
            name='manual',
            recall_from_passtrough=self.manual_step_name,
        )
        root.add_child(PipelineStep.pass_through(name=self.manual_step_name))
        return root

    # ------------------------------------------------------------------
    # Mode 1 — progressive methods
    # ------------------------------------------------------------------

    def _attach_progressive_methods_branches(
        self, manual_root: PipelineStep
    ) -> None:
        """
        Enumerate all (isolator × ordered stitcher permutation) combinations
        and attach each as a chain descending from ``manual_root``.

        For a given isolator, all ordered tuples of length ``max_depth`` from
        the pool of stitcher names are produced via
        ``itertools.permutations(stitcher_names, max_depth)``.
        """
        stitcher_names = list(self.stitcher_specs.keys())

        for isolator_name in self.isolator_specs:
            for stitcher_sequence in itertools.permutations(stitcher_names, self.max_depth):
                chain = _build_chain_progressive_methods(
                    stitcher_sequence=list(stitcher_sequence),
                    isolator_name=isolator_name,
                    isolator_specs=self.isolator_specs,
                    stitcher_specs=self.stitcher_specs,
                )
                _link_chain_to_parent(manual_root, chain)

    # ------------------------------------------------------------------
    # Mode 2 — progressive isolators
    # ------------------------------------------------------------------

    def _attach_progressive_isolators_branches(
        self, manual_root: PipelineStep
    ) -> None:
        """
        Enumerate all (stitcher × ordered isolator permutation) combinations
        and attach each as a chain descending from ``manual_root``.

        For a given stitcher, all ordered tuples of length ``max_depth`` from
        the pool of isolator names are produced via
        ``itertools.permutations(isolator_names, max_depth)``.
        """
        isol_names = list(self.isolator_specs.keys())

        for stitcher_name in self.stitcher_specs.keys():
            for isolator_sequence in itertools.permutations(isol_names, self.max_depth):
                chain = _build_chain_progressive_isolators(
                    isolator_sequence=list(isolator_sequence),
                    stitcher_name=stitcher_name,
                    stitcher_specs=self.stitcher_specs,
                    isolator_specs=self.isolator_specs
                )
                _link_chain_to_parent(manual_root, chain)


def _find_subfolders_with_prefix(root: str, prefix: str) -> list[str]:
    """
    Return all immediate subdirectories of ``root`` whose name starts with
    ``prefix``, sorted in reverse lexicographic order (most recent timestamp
    first when names follow the ``<prefix>_<timestamp>`` convention).

    Parameters
    ----------
    root : str
        Directory to search in.
    prefix : str
        Required name prefix.

    Returns
    -------
    list[str]
        Absolute paths of matching subdirectories; empty list if none found or
        ``root`` does not exist.
    """
    root_path = pathlib.Path(root)
    if not root_path.exists():
        return []

    matches = [
        str(item)
        for item in root_path.iterdir()
        if item.is_dir() and item.name.startswith(prefix)
    ]
    return sorted(matches, reverse=True)


def _find_pipeline_folders(pipelines_root: str, pipeline_name: str) -> list[str]:
    """
    Return all pipeline run folders for a given pipeline name, newest first.

    Pipeline folders follow the naming convention
    ``<pipeline_name>_<timestamp>`` and live under ``pipelines_root``.

    Parameters
    ----------
    pipelines_root : str
        The ``pipelines/`` directory under the output root.
    pipeline_name : str
        Name of the pipeline (used as the folder-name prefix).

    Returns
    -------
    list[str]
        Absolute paths of matching folders, sorted newest-first.
    """
    return _find_subfolders_with_prefix(pipelines_root, pipeline_name)


def _find_comparator_result_folders(output_root: str) -> list[str]:
    """
    Return all ``comparator_results*`` folders under ``output_root``,
    newest first.

    Parameters
    ----------
    output_root : str
        Root folder to search in.

    Returns
    -------
    list[str]
        Absolute paths of matching folders.
    """
    return _find_subfolders_with_prefix(output_root, 'comparator_results')


def _prompt_select_from_list(
    items: list[str],
    prompt_header: str,
    cancel_label: str = "Cancel / run new",
) -> str | None:
    """
    Present a numbered list to the user and return the selected item path.

    Parameters
    ----------
    items : list[str]
        Absolute paths to present; displayed by basename only.
    prompt_header : str
        Introductory line printed above the list.
    cancel_label : str
        Description shown next to the cancel option.

    Returns
    -------
    str or None
        The selected path, or ``None`` if the user chooses to cancel.
    """
    print(f"\n[INFO PIPELINE] {prompt_header}")
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {pathlib.Path(item).name}")

    cancel_index = len(items) + 1
    print(f"  {cancel_index}. {cancel_label}")

    while True:
        raw = input(f"Enter choice (1–{cancel_index}): ").strip()
        if not raw.isdigit():
            print("  Please enter a number.")
            continue
        choice = int(raw)
        if choice == cancel_index:
            return None
        if 1 <= choice <= len(items):
            return items[choice - 1]
        print(f"  Please enter a number between 1 and {cancel_index}.")


def _prompt_reuse_level(
    has_pipeline_folders: bool,
    has_comparator_folders: bool,
) -> str:
    """
    Ask the user which level of reuse they want and return the corresponding
    ``ReuseLevel`` constant.

    Only options that are actually available (i.e. the relevant folders exist
    on disk) are offered.

    Parameters
    ----------
    has_pipeline_folders : bool
        Whether past pipeline run folders were found on disk.
    has_comparator_folders : bool
        Whether past comparator result folders were found on disk.

    Returns
    -------
    str
        One of the ``ReuseLevel`` constants.
    """
    options: list[tuple[str, str]] = [
        (ReuseLevel.FULL_RERUN, "Full rerun — re-stitch everything and rerun comparators"),
    ]

    if has_pipeline_folders:
        options.append((
            ReuseLevel.REUSE_STITCHING,
            "Reuse stitching results — skip re-stitching, rerun comparators only",
        ))

    if has_comparator_folders:
        options.append((
            ReuseLevel.REUSE_EVERYTHING,
            "Reuse everything — load existing comparator results and show plots only",
        ))

    print("\n[INFO PIPELINE] What would you like to do?")
    for idx, (_, label) in enumerate(options, start=1):
        print(f"  {idx}. {label}")

    while True:
        raw = input(f"Enter choice (1–{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print(f"  Please enter a number between 1 and {len(options)}.")


def _collect_leaf_results(
    stitching_results: dict,
) -> dict:
    """
    Filter ``stitching_results`` to keep only leaf nodes.

    Intermediate nodes are identified by the ``"_intermediate"`` suffix that
    ``TreePipeline`` appends to their result keys.

    Parameters
    ----------
    stitching_results : dict
        Full result dict from ``TreePipeline.run()``.

    Returns
    -------
    dict
        Subset containing only leaf-node results.
    """
    return {
        key: value
        for key, value in stitching_results.items()
        if not "intermediate" in key
    }


def _run_ball_query_evaluation(
    stitching_results: dict,
    noise_threshold: float,
    save_path: str | None,
    bplt: bool,
    comparator_folder: str | None = None,
) -> tuple:
    """
    Evaluate results using the ``BallQuery`` comparator.

    If ``stitching_results`` is empty and ``comparator_folder`` is given, the
    most recent BallQuery pickle in that folder is loaded instead.

    Parameters
    ----------
    stitching_results : dict
        Result dict from ``TreePipeline.run()``.
    noise_threshold : float
        RMSE threshold for outlier annotation in plots.
    save_path : str or None
        Where to persist the BallQuery pickle; ``None`` when loading from disk.
    bplt : bool
        Whether to show plots.
    comparator_folder : str or None
        Folder to search for existing pickles when results are empty.

    Returns
    -------
    tuple[dict | None, Figure | None, Figure | None, Figure | None]
        (summary, fig_deltas, fig_histograms, fig_colormap)
    """
    summary, fig_deltas, fig_histograms, fig_colormap = None, None, None, None

    if not stitching_results and comparator_folder:
        bq_files = sorted(pathlib.Path(comparator_folder).glob("ball_query_*.pkl"))
        if bq_files:
            try:
                _, summary, fig_deltas, fig_histograms, fig_colormap = (
                    scomp.Comparator.load_ball_query(
                        comparator_folder, bq_files[-1].name, bplt=bplt
                    )
                )
                print(f"[INFO EVAL] Loaded BallQuery from: {bq_files[-1].name}")
                return summary, fig_deltas, fig_histograms, fig_colormap
            except Exception as exc:
                print(f"[WARN EVAL] Could not load BallQuery: {exc}")

    if not stitching_results:
        print("[INFO EVAL] No stitching results and no existing BallQuery pickle found.")
        return summary, fig_deltas, fig_histograms, fig_colormap

    print("\n[INFO EVAL] Running BallQuery comparator...")
    bq      = scomp.BallQuery(save_path, stitching_results, bplt=False)
    summary = bq.print_summary()

    if bplt:
        fig_deltas     = bq.plot_deltas(noise_threshold=noise_threshold)
        fig_histograms = bq.plot_histograms(noise_threshold=noise_threshold)
        fig_colormap   = bq.colormap_deltas()
        plt.show()

    return summary, fig_deltas, fig_histograms, fig_colormap


def _run_density_map_evaluation(
    stitching_results: dict,
    save_path: str | None,
    bplt: bool,
    comparator_folder: str | None = None,
) -> tuple:
    """
    Evaluate results using the ``DensityMapPosterior`` comparator.

    If ``stitching_results`` is empty and ``comparator_folder`` is given, the
    most recent DMP pickle in that folder is loaded instead.

    Parameters
    ----------
    stitching_results : dict
        Result dict from ``TreePipeline.run()``.
    save_path : str or None
        Where to persist the DMP pickle; ``None`` when loading from disk.
    bplt : bool
        Whether to show plots.
    comparator_folder : str or None
        Folder to search for existing pickles when results are empty.

    Returns
    -------
    tuple[dict | None, Figure | None, dict | None, None]
        (summary, fig_deltas, fig_histograms, None)
        The trailing ``None`` aligns the signature with
        ``_run_ball_query_evaluation``.
    """
    summary, fig_deltas, fig_histograms = None, None, None

    if not stitching_results and comparator_folder:
        dmp_files = sorted(pathlib.Path(comparator_folder).glob("density_map_*.pkl"))
        if dmp_files:
            try:
                _, summary, fig_deltas, fig_histograms = (
                    scomp.Comparator.load_density_map_posterior(
                        comparator_folder, dmp_files[-1].name, bplt=bplt
                    )
                )
                print(f"[INFO EVAL] Loaded DensityMapPosterior from: {dmp_files[-1].name}")
                return summary, fig_deltas, fig_histograms, None
            except Exception as exc:
                print(f"[WARN EVAL] Could not load DensityMapPosterior: {exc}")

    if not stitching_results:
        print("[INFO EVAL] No stitching results and no existing DMP pickle found.")
        return summary, fig_deltas, fig_histograms, None

    print("\n[INFO EVAL] Running DensityMapPosterior comparator...")
    dmp     = scomp.DensityMapPosterior(save_path, stitching_results, bplt=False)
    summary = dmp.print_summary()

    if bplt:
        fig_deltas     = dmp.plot_deltas()
        fig_histograms = dmp.plot_histograms()
        plt.show()

    return summary, fig_deltas, fig_histograms, None


def _run_CAD_comparator(
    stitching_results: dict,
    noise_threshold: float,
    save_path: str | None,
    bplt: bool,
    CAD_comparator_folder: str | None = None,
) -> tuple:
    
    print("\n[INFO EVAL] Running CAD BallQuery comparator...")
    CAD_bq  = scomp.CAD(save_path, stitching_results, bplt=False)
    summary = CAD_bq.print_summary()

    if bplt:
        fig_deltas     = CAD_bq.plot_deltas(noise_threshold=noise_threshold)
        fig_histograms = CAD_bq.plot_histograms(noise_threshold=noise_threshold)
        fig_colormap   = CAD_bq.colormap_deltas()
        plt.show()

    return summary, fig_deltas, fig_histograms, fig_colormap


class OutputPaths:
    """
    Holds every folder and file path involved in a single evaluation run.

    Centralising path construction here means ``MultilevelEvaluationRunner``
    never needs to concatenate strings itself.

    Parameters
    ----------
    output_root : str
        Top-level directory for all pipeline and comparator output.
    pipeline_name : str
        Name of the pipeline; used as the subfolder prefix under ``pipelines/``.
    timestamp : str
        Timestamp string obtained from the completed pipeline run.
    """

    def __init__(self, output_root: str, pipeline_name: str, timestamp: str):
        self.pipeline_folder = os.path.join(
            output_root, 'pipelines', f"{pipeline_name}_{timestamp}"
        )
        self.comparator_folder = os.path.join(output_root, 'comparator_results')

        self.ball_query_pickle = os.path.join(
            self.comparator_folder, f"ball_query_{timestamp}.pkl"
        )
        self.density_map_pickle = os.path.join(
            self.comparator_folder, f"density_map_{timestamp}.pkl"
        )

    def create_directories(self) -> None:
        """Create all output directories that do not yet exist."""
        os.makedirs(self.pipeline_folder,  exist_ok=True)
        os.makedirs(self.comparator_folder, exist_ok=True)


class MultilevelEvaluationRunner:
    """
    Orchestrates building the pipeline, running it, and evaluating results,
    with interactive support for three levels of reuse:

    * **Full rerun** — re-stitch and rerun comparators.
    * **Reuse stitching** — load a past pipeline run, rerun comparators only.
    * **Reuse everything** — load existing comparator pickles and show plots.

    Parameters
    ----------
    mode : str
        ``PipelineMode.PROGRESSIVE_METHODS`` or
        ``PipelineMode.PROGRESSIVE_ISOLATORS``.
    max_depth : int
        Chain length after the manual root (see ``MultilevelPipelineBuilder``).
    stitcher_specs : dict[str, StitcherSpec]
        Stitcher registry.  Build with ``build_default_stitcher_specs()``.
    isolator_names : list[str] or None
        Subset of isolator keys to include. 'maxmin' 'KDTree' 'convex_hull'
    evaluate_intermediate : bool
        If ``True``, pass all pipeline results (including intermediate nodes)
        to the comparators.  If ``False`` (default), only leaf results are
        evaluated.
    noise_threshold : float
        Passed to ``BallQuery.plot_deltas`` for outlier annotation.
    manual_step_name : str
        Identifies the manual alignment pass-through node.
    pipeline_name : str
        Embedded in the ``TreePipeline`` and used in output folder names.
    """

    def __init__(
        self,
        builder: MultilevelPipelineBuilder,
        evaluators: list[str] = ['ballquery', 'dmp'],
        evaluate_intermediate: bool      = False,
        noise_threshold: float           = 0.05,
    ):
        self.builder = builder
        self.pipeline_name         = self.builder.pipeline_name
        self.evaluate_intermediate = evaluate_intermediate
        self.noise_threshold       = noise_threshold

        self.evaluators = evaluators

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        pcs: list[np.ndarray],
        folder_path: str,
        bplt_pipeline: bool    = False,
        bplt_comparators: bool = False,
    ) -> dict:
        """
        Interactively select the reuse level, then execute accordingly.

        Parameters
        ----------
        pcs : list[np.ndarray]
            Raw point clouds to stitch.
        folder_path : str
            Root folder for saving pipeline artefacts.
        bplt_pipeline : bool
            Whether each stitching step shows visualisations.
        bplt_comparators : bool
            Whether comparator plots are displayed.

        Returns
        -------
        dict
            Nested dict with keys ``"ball_query"`` and
            ``"density_map_posterior"``, each containing ``"summary"`` and
            figure references.
        """
        output_root = _resolve_output_root(folder_path)

        pipeline_folders     = _find_pipeline_folders(
            os.path.join(output_root, 'pipelines'), self.pipeline_name
        )
        comparator_folders   = _find_comparator_result_folders(output_root)

        reuse_level = _prompt_reuse_level(
            has_pipeline_folders=bool(pipeline_folders),
            has_comparator_folders=bool(comparator_folders),
        )

        if reuse_level == ReuseLevel.FULL_RERUN:
            return self._run_full(pcs, output_root, bplt_pipeline, bplt_comparators)

        if reuse_level == ReuseLevel.REUSE_STITCHING:
            selected_pipeline_folder = _prompt_select_from_list(
                items=pipeline_folders,
                prompt_header="Select a past pipeline run to load:",
                cancel_label="Cancel (fall back to full rerun)",
            )
            if selected_pipeline_folder is None:
                return self._run_full(pcs, output_root, bplt_pipeline, bplt_comparators)
            return self._run_comparators_only(
                selected_pipeline_folder, output_root, pcs, bplt_pipeline, bplt_comparators
            )

        # ReuseLevel.REUSE_EVERYTHING
        selected_comparator_folder = _prompt_select_from_list(
            items=comparator_folders,
            prompt_header="Select a past comparator results folder:",
            cancel_label="Cancel (fall back to full rerun)",
        )
        if selected_comparator_folder is None:
            return self._run_full(pcs, output_root, bplt_pipeline, bplt_comparators)
        return self._load_and_show_comparators(selected_comparator_folder, bplt_comparators)

    # ------------------------------------------------------------------
    # Execution branches
    # ------------------------------------------------------------------

    def _run_full(
        self,
        pcs: list[np.ndarray],
        output_root: str,
        bplt_pipeline: bool,
        bplt_comparators: bool,
    ) -> dict:
        """
        Run the full pipeline from scratch and then both comparators.

        ``TreePipeline.run()`` is intentionally bypassed here because it
        contains its own interactive past-run prompt, which would conflict
        with the reuse-level choice already made by the caller.  Instead,
        only the *new-run* branch of ``TreePipeline.run()`` is replicated:
        initial identity transforms are built and each root step is executed
        directly via ``PipelineStep.run()``.  All public attributes of the
        built pipeline (``root_steps``, ``timestamp``, ``name``) are used.

        Parameters
        ----------
        pcs : list[np.ndarray]
            Raw point clouds.
        output_root : str
            Top-level output directory.
        bplt_pipeline : bool
            Show stitching visualisations.
        bplt_comparators : bool
            Show comparator plots.

        Returns
        -------
        dict
            Evaluation result dict.
        """
        pipeline   = self.builder.build()
        paths      = OutputPaths(output_root, self.pipeline_name, pipeline.timestamp)
        paths.create_directories()

        base_save_path      = paths.pipeline_folder
        initial_transforms  = [np.eye(4) for _ in pcs]
        stitching_results   = {}

        print(f'[INFO PIPELINE] Starting new Tree Pipeline: {pipeline.name}')
        for root_step in pipeline.root_steps:
            root_results = root_step.run(
                pcs, initial_transforms, base_save_path, bplt_override=bplt_pipeline
            )
            stitching_results.update(root_results)

        results_to_evaluate = self._select_results_scope(stitching_results)
        return self._evaluate(results_to_evaluate, paths, bplt_comparators)


    def _run_comparators_only(
        self,
        pipeline_folder: str,
        output_root: str,
        pcs: list[np.ndarray],
        bplt_pipeline: bool,
        bplt_comparators: bool,
    ) -> dict:
        """
        Load a past pipeline run from disk and run both comparators on it.

        Delegate point-cloud reconstruction entirely to
        ``TreePipeline._run_past_pipeline()``, which already knows how to
        walk the saved ``.pkl`` transform files and re-apply them to the raw
        point clouds via ``stitchSavedTransforms``.  No duplication of that
        logic is needed here.

        Parameters
        ----------
        pipeline_folder : str
            Absolute path to the chosen past pipeline run folder (the
            timestamped subfolder, e.g.
            ``…/pipelines/multilevel_evaluation_pipeline_2026-06-18_10-14-17``).
        output_root : str
            Top-level output directory (for writing new comparator pickles).
        pcs : list[np.ndarray]
            The original, untransformed point clouds in acquisition order,
            forwarded to ``_run_past_pipeline`` for transform application.
        bplt_comparators : bool
            Show comparator plots.

        Returns
        -------
        dict
            Evaluation result dict.
        """
        print(f"\n[INFO PIPELINE] Loading stitching results from: {pathlib.Path(pipeline_folder).name}")

        # Build a temporary TreePipeline shell — only name matters here;
        # root_steps are never executed in this branch.
        shell_pipeline     = TreePipeline(root_steps=[], name=self.pipeline_name)
        stitching_results  = shell_pipeline._run_past_pipeline(pcs, pipeline_folder, bplt=bplt_pipeline)

        # Derive a pseudo-timestamp from the folder name so new comparator
        # pickles land in a clearly labelled file and don't overwrite existing ones.
        folder_suffix = pathlib.Path(pipeline_folder).name.replace(self.pipeline_name + "_", "")
        paths = OutputPaths(output_root, self.pipeline_name, folder_suffix)
        paths.create_directories()

        results_to_evaluate = self._select_results_scope(stitching_results)
        return self._evaluate(results_to_evaluate, paths, bplt_comparators)

    def _load_and_show_comparators(
        self,
        comparator_folder: str,
        bplt_comparators: bool,
    ) -> dict:
        """
        Load existing comparator pickles from disk and display their plots.

        No stitching or comparator recomputation is performed.

        Parameters
        ----------
        comparator_folder : str
            Absolute path to the chosen comparator results folder.
        bplt_comparators : bool
            Show loaded plots.

        Returns
        -------
        dict
            Evaluation result dict (populated from loaded pickles).
        """
        print(f"\n[INFO PIPELINE] Loading comparator results from: {pathlib.Path(comparator_folder).name}")

        summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq = _run_ball_query_evaluation(
            stitching_results={},
            noise_threshold=self.noise_threshold,
            save_path=None,
            bplt=bplt_comparators,
            comparator_folder=comparator_folder,
        )
        summary_dmp, fig_deltas_dmp, fig_hist_dmp, _ = _run_density_map_evaluation(
            stitching_results={},
            save_path=None,
            bplt=bplt_comparators,
            comparator_folder=comparator_folder,
        )
        return _package_evaluation_output(
            summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq,
            summary_dmp, fig_deltas_dmp, fig_hist_dmp,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _select_results_scope(self, stitching_results: dict) -> dict:
        """
        Return either the full result dict or only leaf results, based on
        ``self.evaluate_intermediate``.
        """
        if self.evaluate_intermediate:
            return stitching_results
        return _collect_leaf_results(stitching_results)

    def _evaluate(
        self,
        results_to_evaluate: dict,
        paths: OutputPaths,
        bplt_comparators: bool,
    ) -> dict:
        """
        Run both comparators against ``results_to_evaluate`` and package output.

        Parameters
        ----------
        results_to_evaluate : dict
            Stitching results to pass to comparators.
        paths : OutputPaths
            Pre-built path bundle for this run.
        bplt_comparators : bool
            Show comparator plots.

        Returns
        -------
        dict
            Evaluation result dict.
        """
        summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq = None, None, None, None
        summary_dmp, fig_deltas_dmp, fig_hist_dmp = None, None, None

        if 'ballquery' in self.evaluators: 
            summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq = _run_ball_query_evaluation(
                stitching_results=results_to_evaluate,
                noise_threshold=self.noise_threshold,
                save_path=paths.ball_query_pickle,
                bplt=bplt_comparators,
            )
        if 'dmp' in self.evaluators:
            summary_dmp, fig_deltas_dmp, fig_hist_dmp, _ = _run_density_map_evaluation(
                stitching_results=results_to_evaluate,
                save_path=paths.density_map_pickle,
                bplt=bplt_comparators,
            )
        return _package_evaluation_output(
            summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq,
            summary_dmp, fig_deltas_dmp, fig_hist_dmp,
        )
    

    def _evaluate_with_CAD(
        self,
        results_to_evaluate: dict,
        paths: OutputPaths,
        bplt_CAD: bool,
    ) -> dict:
        """
        Run comparator BallQuery against ``results_to_evaluate`` and package output.

        Parameters
        ----------
        results_to_evaluate : dict
            Stitching results to pass to comparator CAD.
        paths : OutputPaths
            Pre-built path bundle for this run.
        bplt_CAD : bool
            Show comparator plots.

        Returns
        -------
        dict
            Evaluation result dict.
        """
        summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq = _run_CAD_comparator(
            stitching_results=results_to_evaluate,
            noise_threshold=self.noise_threshold,
            save_path=paths.ball_query_pickle,
            bplt=bplt_CAD,
        )

        return _package_evaluation_output(
            summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq,
        )


def _resolve_output_root(folder_path: str) -> str:
    """Return the directory component of ``folder_path``."""
    return (
        os.path.dirname(folder_path)
        if os.path.isfile(folder_path)
        else folder_path
    )

def _package_evaluation_output(
    summary_bq, fig_deltas_bq, fig_hist_bq, fig_cmap_bq,
    summary_dmp, fig_deltas_dmp, fig_hist_dmp,
) -> dict:
    """
    Assemble the standard evaluation output dict from comparator results.

    Returns
    -------
    dict
        ``{"ball_query": {...}, "density_map_posterior": {...}}``
    """
    return {
        "ball_query": {
            "summary":        summary_bq,
            "fig_deltas":     fig_deltas_bq,
            "fig_histograms": fig_hist_bq,
            "fig_colormap":   fig_cmap_bq,
        },
        "density_map_posterior": {
            "summary":        summary_dmp,
            "fig_deltas":     fig_deltas_dmp,
            "fig_histograms": fig_hist_dmp,
        },
    }
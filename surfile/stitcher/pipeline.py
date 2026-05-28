from surfile.stitcher import stitcher as sst

import inspect
import os
import pickle
import numpy as np
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
    
    def __init__(self, f, name=None, recall_from_passtrough=None, **kwargs):
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
            return {self.name: (fixed, next_pcs)}

        leaf_results = {}
        for child in self.children:
            child_results = child.run(next_pcs, new_global_transforms, base_save_path)
            leaf_results.update(child_results)
        return leaf_results

    def _save(self, transforms, base_path, leaf_path_name):
        save_folder = os.path.join(base_path, leaf_path_name)
            
        os.makedirs(save_folder, exist_ok=False)
        print(f"[INFO PIPELINE] Saving leaf results to: {leaf_path_name}")
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
                return sst.SurfaceStitcher.stitchSavedTransforms(pcs, subdir, bplt=bplt)
        
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

        for subdir in os.listdir(past_run_folder):
            fixed, next_pcs, _ = sst.SurfaceStitcher.stitchSavedTransforms(pcs, os.path.join(past_run_folder, subdir), bplt=bplt)
            stitching_results[subdir] = (fixed, next_pcs)
        return stitching_results
    
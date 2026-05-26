from surfile.stitcher import stitcher as sst

import inspect
import os
import pickle
import numpy as np
from datetime import datetime

class PipelineStep:
    f: callable
    t: list
    
    def __init__(self, f, name=None, **kwargs):
        self.f = f
        self.name = name
        self.kwargs = kwargs
        self.t = []
        self.children = []
        print(f'[INFO PIPELINE] Created step with function {f.__name__}')
        
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
    
    def run(self, pcs, current_transforms, base_save_path=None, bplt_override=False) -> None:
        self._check_arguments()
        self._check_all_leaves_have_name()
        if bplt_override: self.kwargs['bplt'] = bplt_override
        fixed, next_pcs, local_transforms = self.f(pcs, **self.kwargs)
        
        new_global_transforms = [T_local @ T for T, T_local in zip(current_transforms, local_transforms)]

        if not self.children and base_save_path is not None:
            self._save(new_global_transforms, base_save_path, self.name)
        else:
            for child in self.children:
                child.run(next_pcs, new_global_transforms, base_save_path)

    def _save(self, transforms, base_path, leaf_path_name):
        save_folder = os.path.join(base_path, leaf_path_name)
            
        os.makedirs(save_folder, exist_ok=False)
        print(f"[INFO PIPELINE] Saving leaf results to: {leaf_path_name}")
        for i, T in enumerate(transforms):
            with open(os.path.join(save_folder, f"{i}.pkl"), "wb") as f:
                pickle.dump(T, f)

    @classmethod
    def pass_through(cls, name): # it must return the same shape as the stitcher functions, so that it can be used as a PipelineStep function: fixed, point_clouds_T, transforms_matrices
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
            choice = self._prompt_user_for_run_choice(past_runs)
            if choice is None: # make a new run
                pass

            else: # apply existing
                print(f'[INFO PIPELINE] Applying past pipeline results from: {choice}')
                stitching_results = self._run_past_pipeline(pcs, os.path.join(save_transforms_root, 'pipelines', choice), bplt=bplt)
                return stitching_results

        initial_transforms = [np.eye(4) for _ in pcs]
        print(f'[INFO PIPELINE] Starting new Tree Pipeline: {self.name}')
        for root in self.root_steps:
            root.run(pcs, initial_transforms, base_save_path, bplt_override=bplt)
            return stitching_results
            
    def _check_for_past_runs(self, save_transforms_root):        
        past_runs = []
        if not os.path.exists(os.path.join(save_transforms_root, 'pipelines')): 
            return past_runs
        for subdir in os.listdir(os.path.join(save_transforms_root, 'pipelines')):
            if subdir.startswith(self.name):
                past_runs.append(subdir)
                
        return past_runs
    
    def _prompt_user_for_run_choice(self, past_runs):
        print(f"Found {len(past_runs)} past runs for pipeline '{self.name}':")
        for i, run in enumerate(past_runs):
            print(f"{i}: {run}")
        print(f"{len(past_runs)}: Run new pipeline")
        
        while True:
            choice = input(f"Select a past run to apply or run new pipeline (0-{len(past_runs)}): ")
            if choice.isdigit():
                choice_idx = int(choice)
                if 0 <= choice_idx <= len(past_runs):
                    return past_runs[choice_idx] if choice_idx < len(past_runs) else None
            print("Invalid input. Please enter a number corresponding to the options above.")
            
    def _run_past_pipeline(self, pcs, past_run_folder, bplt=False):
        stitching_results = {}

        for subdir in os.listdir(past_run_folder):
            fixed, next_pcs = sst.SurfaceStitcher.stitchSavedTransforms(pcs, os.path.join(past_run_folder, subdir), bplt=bplt)
            stitching_results[subdir] = (fixed, next_pcs)
        return stitching_results
    
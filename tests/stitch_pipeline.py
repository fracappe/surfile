import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from surfile.stitcher import stitcher as sst
from surfile.stitcher import utils as sutils
from surfile import measfile_io as fio

import numpy as np
import pickle

import tkinter as tk
from tkinter import filedialog

def open_files(folder=None, downsample=1) -> list[np.ndarray]:
    if folder is None:
            root = tk.Tk()
            root.withdraw()  # Hide the main Tkinter window

            folder_path = filedialog.askdirectory(
                title="Select a Folder with Point Cloud Files", 
                initialdir="G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures"
            )
            if not folder_path:
                print("No folder selected. Exiting.")
                exit()
    else:
        folder_path = folder

    # --- 2. Load Point Clouds ---
    print(f"📂 Loading point clouds from: {folder_path}")
    try:
        # Using a downsample factor to speed up tests, adjust if needed.
        point_clouds = fio.open_pc_from_dir(folder_path, downsample=downsample)
        if not point_clouds or len(point_clouds) < 2:
            raise ValueError(
                "Could not load at least two point clouds for stitching."
            )
        print(f"✅ Successfully loaded {len(point_clouds)} point clouds.")
    except Exception as e:
        print(f"❌ Failed to load point clouds: {e}")
        exit()
            
    return point_clouds, folder_path

pipe = sutils.Pipeline(
    steps=[
        sutils.PipelineStep(
            f=sst.SurfaceStitcher.stitchManual,
            bplt=True
        ),
        
        sutils.PipelineStep(
            f=sst.SurfaceStitcher.stitchICP,
            isolator=sst.Isolator(type='convex_hull'),
            thresholder=sst.Thresholder(type='KDTree'),
            bplt=True
        ),
    ]
)

if __name__ == "__main__":
    
    pcs, folder_path = open_files(folder='G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth', downsample=2)

    pipelines_dir = os.path.join(folder_path, 'pipelines')
    if os.path.exists(pipelines_dir):
        subdirs = [d for d in os.listdir(pipelines_dir) if os.path.isdir(os.path.join(pipelines_dir, d))]
        if subdirs:
            print("Existing pipeline results found:")
            for i, name in enumerate(subdirs):
                print(f"{i+1}. {name}")
            print(f"{len(subdirs)+1}. Run new pipeline and save result")
            
            choice = input("Choose an option (number): ")
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(subdirs):
                    # Apply existing
                    result_dir = os.path.join(pipelines_dir, subdirs[choice_idx])
                    sst.SurfaceStitcher.stitchSavedTransforms(pcs, result_dir, bplt=True)
                elif choice_idx == len(subdirs):
                    pipe.run(pcs, save_transforms=folder_path)
                else:
                    print("Invalid choice. Running new pipeline.")
                    pipe.run(pcs, save_transforms=folder_path)
            except ValueError:
                print("Invalid input. Running new pipeline.")
                pipe.run(pcs, save_transforms=folder_path)
        else:
            pipe.run(pcs, save_transforms=folder_path)
    else:
        pipe.run(pcs, save_transforms=folder_path)
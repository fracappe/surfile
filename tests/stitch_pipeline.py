import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from surfile.stitcher import stitcher as sst
from surfile.stitcher import utils as sutils
from surfile import measfile_io as fio

import numpy as np
import pickle

import tkinter as tk
from tkinter import filedialog

def open_files(folder=None, downsample=1, userscales=[1, 1, 1]) -> list[np.ndarray]:
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
        point_clouds = fio.open_pc_from_dir(folder_path, downsample=downsample, userscales=userscales)
        if not point_clouds or len(point_clouds) < 2:
            raise ValueError(
                "Could not load at least two point clouds for stitching."
            )
        print(f"✅ Successfully loaded {len(point_clouds)} point clouds.")
    except Exception as e:
        print(f"❌ Failed to load point clouds: {e}")
        exit()
            
    return point_clouds, folder_path

step_man = sutils.PipelineStep(sst.SurfaceStitcher.stitchManual, bplt=True)
step_icp_ch = sutils.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_ch", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='convex_hull'), bplt=True)
step_icp_mm = sutils.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_mm", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='maxmin', axes='xy'), bplt=True)

step_man.add_child(step_icp_ch)
step_man.add_child(step_icp_mm)

pipe = sutils.TreePipeline(root_steps=[step_man], name="man_icp_pipe")

if __name__ == "__main__":
    
    pcs, folder_path = open_files(folder='G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth', downsample=2, userscales=[1, 1, 1])

    pipe.run(pcs, folder_path)
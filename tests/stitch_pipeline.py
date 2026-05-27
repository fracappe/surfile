import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from surfile.stitcher import stitcher as sst
from surfile.stitcher import utils as sutils
from surfile.stitcher import pipeline as spipe
from surfile.stitcher import comparator
from surfile import measfile_io as fio

import numpy as np
import matplotlib.pyplot as plt
import pickle

import tkinter as tk
from tkinter import filedialog

def open_file(path=None, downsample=1, userscales=[1, 1, 1]) -> np.ndarray:
    if path is None:
            root = tk.Tk()
            root.withdraw()  # Hide the main Tkinter window

            file_path = filedialog.askopenfilename(
                title="Select a Point Cloud File", 
                initialdir="G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures"
            )
            if not file_path:
                print("No file selected. Exiting.")
                exit()
    else:
        file_path = path

    print(f"📂 Loading point cloud from: {file_path}")
    try:
        # Using a downsample factor to speed up tests, adjust if needed.
        point_cloud = fio.open_pc_from_file(file_path, downsample=downsample, userscales=userscales)
        print(f"✅ Successfully loaded point cloud with shape: {point_cloud.shape}")
    except Exception as e:
        print(f"❌ Failed to load point cloud: {e}")
        exit()
            
    return point_cloud, file_path

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

step_man = spipe.PipelineStep(sst.SurfaceStitcher.stitchManual, recall_from_passtrough='pt_manual', bplt=True)
step_icp_ch = spipe.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_ch", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='convex_hull'), bplt=True)
step_icp_mm = spipe.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_mm", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='maxmin', axes='xy'), bplt=True)

step_man.add_child(spipe.PipelineStep.pass_through(name="pt_manual"))
step_man.add_child(step_icp_ch)
step_man.add_child(step_icp_mm)

pipe = spipe.TreePipeline(root_steps=[step_man], name="manpt_icp_pipe_t2")

step_man_CAD = spipe.PipelineStep(sst.SurfaceStitcher.stitchManual, recall_from_passtrough='pt_manual_cad', bplt=True)
step_icp_CAD = spipe.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_cad", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='KDTree'), bplt=True)

step_man_CAD.add_child(spipe.PipelineStep.pass_through(name="pt_manual_cad"))
step_man_CAD.add_child(step_icp_CAD)

pipe_CAD = spipe.TreePipeline(root_steps=[step_man_CAD], name="manual_top_cad_pipe")

if __name__ == "__main__":
    folder = 'G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth_test_Andrea'
    pcs, folder_path = open_files(folder=folder, downsample=3, userscales=[1, 1, 1])
    stitching_results = pipe.run(pcs, folder_path, bplt=False)

    # top_pc, top_path = open_file(path='G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth\\Aocclusale_coordinate.txt_resaved.npy', downsample=1, userscales=[1, 1, 1])
    cad_pc, cad_path = open_file(path='G:\\Drive condivisi\\TIROCINI\\2025 - Francesca Capellino\\Tesi\\Corone\\corona_cad.stl', downsample=1, userscales=[1000, 1000, 1000])

    bq = comparator.CAD(stitching_results, cad_pc, pipeline=pipe_CAD, pipeline_path=folder, names=None)
    bq.print_summary()

    plt.show()
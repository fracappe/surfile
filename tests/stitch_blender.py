"""
Tests blender functionality from the utils submodule
"""
import tkinter as tk
from tkinter import filedialog

import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from surfile import measfile_io as fio
from surfile.stitcher.utils import blender_edit_point_cloud
import surfile.stitcher.plotter as splt


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


if __name__ == "__main__":
    # open first tooth cloud
    file = 'G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth_test_Andrea\\Aocclusale_coordinate.txt_resaved.npy'
    pc, _ = open_file(file, userscales=[1, 1, 1])
    
    edited_pc = blender_edit_point_cloud(pc)
    
    # splt.compare_point_clouds([pc, edited_pc], colors='plasma')
    splt.show_point_clouds([pc, edited_pc], colors='uniform')
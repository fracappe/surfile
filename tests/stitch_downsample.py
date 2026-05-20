"""
stitch_downsample.py
- An interactive test script for the surfile.stitcher.stitcher module, focused on testing the downsampling functionality.

This script will:
1. Ask the user to select a folder containing point cloud files.
2. Load all supported point clouds from that folder.
3. Downsample the point clouds using a specified downsampling factor and resave in .npy format.
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import surfile.measfile_io as fio

def apply_downsampling(downsample_factor=5, folder=None):
    if folder is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        folder = filedialog.askdirectory(
            title="Select a Folder with Point Cloud Files (.txt, .npy, .stl)"
        )

    if not folder:
        print("No folder selected. Exiting.")
        return

    print(f"Loading point clouds from: {folder}")
    try:
        point_clouds_np = fio.open_pc_from_dir(folder, downsample=downsample_factor, resave={'resave': True, 'resample': downsample_factor})
        if not point_clouds_np:
            raise ValueError("No point clouds were loaded.")
        print(f"Successfully loaded and downsampled {len(point_clouds_np)} point clouds.")
    except Exception as e:
        print(f"❌ Failed to load and downsample point clouds: {e}")
        return


if __name__ == "__main__":
    apply_downsampling(downsample_factor=1)

from surfile import measfile_io as fio

import numpy as np

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
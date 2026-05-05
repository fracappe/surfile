"""
'test_stitcher.py'
- An interactive test script for the surfile.stitcher.stitcher module.
This script will:
1. Ask the user to select a folder containing point cloud files.
2. Load all supported point clouds from that folder.
3. Ask the user which stitching methods they want to test.
4. Execute the selected methods with pre-configured parameters and plots.
"""

import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import surfile.measfile_io as fio
from surfile.stitcher.stitcher import SurfaceStitcher, Isolator, Thresholder


def get_user_method_choice(methods):
    """
    Prompts the user to select which stitching methods to run.

    Parameters
    ----------
    methods : dict
        A dictionary of available test methods.

    Returns
    -------
    list[str]
        A list of keys for the selected methods.
    """
    print("\nAvailable stitching methods to test:")
    for i, name in enumerate(methods.keys()):
        print(f"  [{i+1}] {name}")

    while True:
        try:
            choice_str = input(
                "\nEnter the numbers of the methods to run in sequence (e.g., '5 1' for Manual then ICP), or 'all': "
            )
            if choice_str.lower() == 'all':
                return list(methods.keys())

            choices = [int(c) - 1 for c in choice_str.split()]
            selected_keys = [list(methods.keys())[i] for i in choices]

            if not all(0 <= i < len(methods) for i in choices):
                raise IndexError

            return selected_keys
        except (ValueError, IndexError):
            print("❌ Invalid input. Please enter valid numbers from the list.")
            
def apply_stitch_sequence(bplt=True):
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    folder_path = filedialog.askdirectory(
        title="Select a Folder with Point Cloud Files", 
        initialdir="G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures"
    )
    if not folder_path:
        print("No folder selected. Exiting.")
        exit()

    # --- 2. Load Point Clouds ---
    print(f"📂 Loading point clouds from: {folder_path}")
    try:
        # Using a downsample factor to speed up tests, adjust if needed.
        point_clouds = fio.open_pc_from_dir(folder_path, downsample=5)
        if not point_clouds or len(point_clouds) < 2:
            raise ValueError(
                "Could not load at least two point clouds for stitching."
            )
        print(f"✅ Successfully loaded {len(point_clouds)} point clouds.")
    except Exception as e:
        print(f"❌ Failed to load point clouds: {e}")
        exit()

    # --- 3. Define Stitching Methods and Parameters ---
    # Define some default parameters for the different algorithms.
    # These can be tuned as needed for your specific data.
    save_folder = os.path.join(folder_path, "stitch_transforms")

    # A simple isolator based on bounding box intersection.
    isolator_maxmin = Isolator(type='maxmin', axes='xy')
    # A thresholder that estimates distance from the data.
    thresholder_kdtree = Thresholder(type='KDTree')
    
    # Map method names to their corresponding save subfolders.
    method_to_subfolder = {
        "ICP (Iterative Closest Point)": "icp",
        "FGR (Fast Global Registration)": "fgr",
        "RMSE (Bayesian Optimization)": "rmse",
        "2D Correlation": "cc",
        "Manual Point Selection": "man",
        "Robot-based Transforms": "rob",
    }
    
    # Dictionary of available tests
    # Each function now accepts a list of point clouds (`pcs`) as input.
    test_methods = {
        "ICP (Iterative Closest Point)": lambda pcs, save_transform: SurfaceStitcher.stitchICP(
            pcs, thresholder_kdtree, isolator_maxmin, save_transform=save_transform, bplt=bplt
        ),
        "FGR (Fast Global Registration)": lambda pcs, save_transform: SurfaceStitcher.stitchFGR(
            pcs, voxel_size=0.1, thresholder=thresholder_kdtree, isolator=isolator_maxmin, save_transform=save_transform, bplt=bplt
        ),
        "RMSE (Bayesian Optimization)": lambda pcs, save_transform: SurfaceStitcher.stitchRMSE(
            pcs, n_calls=50, isolator=isolator_maxmin, save_transform=save_transform, bplt=bplt
        ),
        "2D Correlation": lambda pcs, save_transform: SurfaceStitcher.stitchCorrelation(
            pcs, dx=0.1, dy=0.1, isolator=isolator_maxmin, save_transform=save_transform, bplt=bplt
        ),
        "Manual Point Selection": lambda pcs, save_transform: SurfaceStitcher.stitchManual(
            pcs, save_transform=save_transform, bplt=bplt
        ),
        "Robot-based Transforms": lambda pcs, save_transform, robot_file_path: SurfaceStitcher.stitchRobot(
            pcs,
            robotTfile=robot_file_path, # robot_file_path will be determined dynamically
            save_transform=save_transform,
            bplt=bplt
        ),
    }

    # --- 4. Get User Selection and Run Chained Tests ---
    selected_methods = get_user_method_choice(test_methods)

    # Initialize the working set of point clouds with the original data
    current_point_clouds = point_clouds

    for method_name in selected_methods:
        print("\n" + "="*60)
        print(f"🚀 Running test for: {method_name}")
        print("="*60)

        stitching_function = test_methods[method_name]

        method_subfolder_name = method_to_subfolder.get(method_name)
        # The full path to the specific method's transform folder
        method_transform_folder = os.path.join(save_folder, method_subfolder_name) if method_subfolder_name else None

        use_saved_transforms = False
        if method_transform_folder and os.path.exists(method_transform_folder):
            # Check if there are actual .pkl files in the folder and if the count matches
            pkl_files = [f for f in os.listdir(method_transform_folder) if f.endswith(".pkl")]
            if len(pkl_files) == len(current_point_clouds) - 1:
                choice = input(
                    f"Transforms for '{method_name}' already exist in '{method_transform_folder}'.\n"
                    "Do you want to [R]un the method or [L]oad saved transforms? (r/L): "
                ).lower()
                if choice in ['r', 'l']:
                    use_saved_transforms = (choice == 'l')
                else:
                    use_saved_transforms = True
                        
            elif len(pkl_files) > 0:
                print(f"⚠️ Warning: Found {len(pkl_files)} transform files for '{method_name}' in '{method_transform_folder}', but expected {len(current_point_clouds) - 1}. Running the method instead of loading.")
                use_saved_transforms = False # Force running the method if file count mismatch

        try:
            if use_saved_transforms:
                print(f"🔄 Loading saved transforms for '{method_name}' from '{method_transform_folder}'...")
                stitched_cloud, transformed_clouds = SurfaceStitcher.stitchSavedTransforms(
                    current_point_clouds, transforms_folder=method_transform_folder, bplt=bplt
                )
            else:
                if method_name == "Robot-based Transforms":
                    robot_file = filedialog.askopenfilename(title="Select Robot Transform File")
                    if not robot_file:
                        print(f"❌ Robot transform file not selected for '{method_name}'. Skipping.")
                        continue # Skip this method if file not selected
                    # Pass the base save_folder; stitchRobot will create the 'rob' subfolder.
                    stitched_cloud, transformed_clouds = stitching_function(
                        current_point_clouds, save_folder, robot_file
                    )
                else:
                    # Pass the base save_folder; the specific stitch method will create its own subfolder.
                    stitched_cloud, transformed_clouds = stitching_function(current_point_clouds, save_folder)
                    
            # The output of this step becomes the input for the next iteration
            current_point_clouds = transformed_clouds
            print(f"✅ Stitching with '{method_name}' complete.")
            print(f"   - Final merged cloud has {len(stitched_cloud)} points.")
        except Exception as e:
            print(f"❌ An error occurred during '{method_name}': {e}")

    print("\n🎉 All selected stitching methods completed!")
    return current_point_clouds
    
if __name__ == "__main__":
    # --- 1. Setup Tkinter and get folder path ---
    apply_stitch_sequence()
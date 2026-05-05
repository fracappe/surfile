"""
test_comparator.py
==================
An interactive test script for comparing stitching methods using the
comparator module from surfile.stitcher.

This script will:
1. Ask the user to select a folder containing point cloud files.
2. Load all supported point clouds from that folder.
3. Execute all available stitching methods.
4. Use the comparator module to compare and visualize the results.
"""

import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..\\")
import surfile.measfile_io as fio
from surfile.stitcher import stitcher as sstitcher
from surfile.stitcher import comparator as scomparator


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


def run_stitching_comparison():
    """
    Main function to run the stitching comparison test.
    """
    # --- 1. Setup Tkinter and get folder path ---
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
        point_clouds = fio.open_pc_from_file(folder_path, downsample=5)
        if not point_clouds or len(point_clouds) < 2:
            raise ValueError(
                "Could not load at least two point clouds for stitching."
            )
        print(f"✅ Successfully loaded {len(point_clouds)} point clouds.")
    except Exception as e:
        print(f"❌ Failed to load point clouds: {e}")
        exit()

    # --- 3. Define Stitching Methods and Parameters ---
    save_folder = os.path.join(folder_path, "stitch_transforms")

    # A simple isolator based on bounding box intersection.
    isolator_maxmin = sstitcher.Isolator(type='maxmin', axes='xy')
    # A thresholder that estimates distance from the data.
    thresholder_kdtree = sstitcher.Thresholder(type='KDTree')
    
    # Map method names to their corresponding save subfolders.
    method_to_subfolder = {
        "ICP (Iterative Closest Point)": "icp",
        "FGR (Fast Global Registration)": "fgr",
        "RMSE (Bayesian Optimization)": "rmse",
        "2D Correlation": "cc",
        "Manual Point Selection": "man",
        "Robot-based Transforms": "rob",
        "Saved Transforms": "saved",
    }
    
    # Dictionary of available stitching methods
    # Each function accepts a list of point clouds (`pcs`) and save_transform path
    test_methods = {
        "ICP (Iterative Closest Point)": lambda pcs, save_transform: sstitcher.SurfaceStitcher.stitchICP(
            pcs, thresholder_kdtree, isolator_maxmin, save_transform=save_transform, bplt=False
        ),
        "FGR (Fast Global Registration)": lambda pcs, save_transform: sstitcher.SurfaceStitcher.stitchFGR(
            pcs, voxel_size=0.1, thresholder=thresholder_kdtree, isolator=isolator_maxmin, save_transform=save_transform, bplt=False
        ),
        "RMSE (Bayesian Optimization)": lambda pcs, save_transform: sstitcher.SurfaceStitcher.stitchRMSE(
            pcs, n_calls=50, isolator=isolator_maxmin, save_transform=save_transform, bplt=False
        ),
        "2D Correlation": lambda pcs, save_transform: sstitcher.SurfaceStitcher.stitchCorrelation(
            pcs, dx=0.1, dy=0.1, isolator=isolator_maxmin, save_transform=save_transform, bplt=False
        ),
        "Manual Point Selection": lambda pcs, save_transform: sstitcher.SurfaceStitcher.stitchManual(
            pcs, save_transform=save_transform, bplt=False
        ),
        "Robot-based Transforms": lambda pcs, save_transform, robot_file_path: sstitcher.SurfaceStitcher.stitchRobot(
            pcs,
            robotTfile=robot_file_path,
            save_transform=save_transform,
            bplt=False
        ),
        "Saved Transforms": lambda pcs, save_transform, saved_folder: sstitcher.SurfaceStitcher.stitchSavedTransforms(
            pcs,
            transforms_folder=saved_folder,
            bplt=False
        ),
    }

    # --- 4. Get User Selection ---
    selected_methods = get_user_method_choice(test_methods)
    print(f"\n🔬 Selected methods: {selected_methods}")

    # Initialize the working set of point clouds with the original data
    current_point_clouds = point_clouds

    # Dictionary to store results from each method
    stitching_results = {}

    for method_name in selected_methods:
        print("\n" + "="*60)
        print(f"🚀 Running stitching method: {method_name}")
        print("="*60)

        stitching_function = test_methods[method_name]

        method_subfolder_name = method_to_subfolder.get(method_name)
        method_transform_folder = os.path.join(save_folder, method_subfolder_name) if method_subfolder_name else None

        use_saved_transforms = False
        if method_transform_folder and os.path.exists(method_transform_folder):
            # Check if there are actual .pkl files in the folder
            pkl_files = [f for f in os.listdir(method_transform_folder) if f.endswith(".pkl")]
            if len(pkl_files) == len(current_point_clouds) - 1:
                while True:
                    choice = input(
                        f"Transforms for '{method_name}' already exist in '{method_transform_folder}'.\n"
                        "Do you want to [R]un the method or [L]oad saved transforms? (R/L): "
                    ).lower()
                    if choice in ['r', 'l']:
                        use_saved_transforms = (choice == 'l')
                        break
                    else:
                        print("Invalid choice. Please enter 'R' or 'L'.")
            elif len(pkl_files) > 0:
                print(f"⚠️ Warning: Found {len(pkl_files)} transform files for '{method_name}' in '{method_transform_folder}', but expected {len(current_point_clouds) - 1}. Running the method instead of loading.")
                use_saved_transforms = False

        try:
            if use_saved_transforms:
                print(f"🔄 Loading saved transforms for '{method_name}' from '{method_transform_folder}'...")
                stitched_cloud, transformed_clouds = sstitcher.SurfaceStitcher.stitchSavedTransforms(
                    current_point_clouds, transforms_folder=method_transform_folder, bplt=False
                )
            else:
                if method_name == "Robot-based Transforms":
                    robot_file = filedialog.askopenfilename(title="Select Robot Transform File")
                    if not robot_file:
                        print(f"❌ Robot transform file not selected for '{method_name}'. Skipping.")
                        continue
                    stitched_cloud, transformed_clouds = stitching_function(
                        current_point_clouds, save_folder, robot_file
                    )
                elif method_name == "Saved Transforms":
                    # For saved transforms, ask user to select the folder
                    saved_path = filedialog.askdirectory(
                        title="Select Saved Transforms Folder",
                        initialdir=save_folder
                    )
                    if not saved_path:
                        print(f"❌ Saved transforms folder not selected for '{method_name}'. Skipping.")
                        continue
                    stitched_cloud, transformed_clouds = stitching_function(
                        current_point_clouds, save_folder, saved_path
                    )
                else:
                    stitched_cloud, transformed_clouds = stitching_function(current_point_clouds, save_folder)
            
            # Store the result
            stitching_results[method_name] = stitched_cloud
            
            # The output of this step becomes the input for the next iteration
            current_point_clouds = transformed_clouds
            print(f"✅ Stitching with '{method_name}' complete.")
            print(f"   - Final merged cloud has {len(stitched_cloud)} points.")
        except Exception as e:
            print(f"❌ An error occurred during '{method_name}': {e}")
            import traceback
            traceback.print_exc()

    # --- 5. Compare Results using Comparator ---
    if len(stitching_results) < 2:
        print("\n⚠️ Skipping comparison.")
        return

    print("\n" + "="*60)
    print("📊 Comparing stitching results using comparator...")
    print("="*60)

    # Prepare data for comparison
    stitched_list = list(stitching_results.values())
    labels = list(stitching_results.keys())

    # Define a radius function for compute_deltas
    def compute_R(point):
        return 0.3

    # Set noise threshold
    noise_threshold = 0.1

    # Generate comparison plot
    print("📈 Generating comparison plots...")
    fig = scomparator.plot_stitching_comparison(
        stitched_list,
        compute_R,
        noise_threshold=noise_threshold,
        labels=labels,
        figsize_scale=5.0
    )

    # Save the comparison figure
    comparison_save_path = os.path.join(folder_path, "stitch_comparison.png")
    fig.savefig(comparison_save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Comparison figure saved to: {comparison_save_path}")

    plt.show()

    print("\n🎉 Stitching comparison complete!")
    print(f"📁 Results saved to: {comparison_save_path}")


if __name__ == "__main__":
    run_stitching_comparison()
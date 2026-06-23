"""
'test_plotter.py'
- A test script for the surfile.stitcher.plotter module.

This script will:
1. Ask the user to select a folder containing point cloud files.
2. Load the point clouds using `measfile_io`.
3. Run a series of tests on the visualization functions in `plotter.py`.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import surfile.stitcher.plotter as splotter
import time

from tests.file_helper import *


def run_test(test_name, func, *args, **kwargs):
    """Helper function to run and time a test."""
    print("\n" + "="*50)
    print(f"🚀 Running test: {test_name}")
    print("="*50)
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    print(f"✅ Test '{test_name}' finished in {end_time - start_time:.2f} seconds.")
    print("➡️ Please close the visualization window to continue to the next test.")


if __name__ == "__main__":
    point_clouds_np = open_files()

    run_test("show_point_clouds with 'uniform' colors",
             splotter.show_point_clouds, point_clouds_np, colors="uniform")

    run_test("show_point_clouds with 'normal' colors",
             splotter.show_point_clouds, point_clouds_np, colors="normal")

    # run_test("show_point_clouds with 'betternormal' colors",
    #          splotter.show_point_clouds, point_clouds_np, colors="betternormal")

    if point_clouds_np:
        run_test("show_point_clouds with 'viridis' colormap",
                 splotter.show_point_clouds, [point_clouds_np[0]], colors="viridis")

    if len(point_clouds_np) >= 2:
        pc_lists_to_compare = [
            [point_clouds_np[0]],  # First window shows the first PC
            [point_clouds_np[1]]   # Second window shows the second PC
        ]
        run_test("compare_point_clouds in separate windows",
                 splotter.compare_point_clouds, pc_lists_to_compare, colors=["afmhot", "coolwarm"])

    print("\n🎉 All plotter tests completed!")
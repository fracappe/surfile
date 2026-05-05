"""
'test_isolator.py'
- An interactive test script for the surfile.stitcher.Isolator class.

This script will:
1. Ask the user to select two point cloud files.
2. Load the selected point clouds.
3. Ask the user which isolator methods they want to test.
4. Execute the selected methods and plot the isolated areas.
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import surfile.measfile_io as fio
import surfile.stitcher.stitcher as sstitcher
import surfile.stitcher.plotter as splotter


def get_user_isolator_choice(isolator_options):
    """
    Prompts the user to select which isolator methods to run.

    Parameters
    ----------
    isolator_options : dict
        A dictionary of available isolator options.

    Returns
    -------
    list[str]
        A list of keys for the selected isolator types.
    """
    print("\nAvailable isolator methods to test:")
    for i, name in enumerate(isolator_options.keys()):
        print(f"  [{i+1}] {name}")

    while True:
        try:
            choice_str = input(
                "\nEnter the numbers of the methods to run (e.g., '1 3'), or 'all': "
            )
            if choice_str.lower() == 'all':
                return list(isolator_options.keys())

            choices = [int(c) - 1 for c in choice_str.split()]
            selected_keys = [list(isolator_options.keys())[i] for i in choices]

            if not all(0 <= i < len(isolator_options) for i in choices):
                raise IndexError

            return selected_keys
        except (ValueError, IndexError):
            print("❌ Invalid input. Please enter valid numbers from the list.")


if __name__ == "__main__":
    from stitch_stitcher import apply_stitch_sequence
    # --- 1. Setup Tkinter and get folder path ---
    pre_registered_pcs = apply_stitch_sequence(bplt=False)  # Get pre-registered point clouds without plotting
    pc1 = pre_registered_pcs[0]
    pc2 = pre_registered_pcs[1]
    
    # --- 3. Define available Isolator types ---
    isolator_options = {
        "Geometrical Isolator": sstitcher.Isolator(type='geometrical', stitchprc=80),
        "Max-Min Bounding Box Isolator": sstitcher.Isolator(type='maxmin', axes='xyz'),
        "KDTree Distance Isolator": sstitcher.Isolator(type='KDTree', max_distance=None), # max_distance will be estimated
        "Convex Hull Isolator": sstitcher.Isolator(type='convex_hull'),
        "Manual Selection Isolator": sstitcher.Isolator(type='manual') # This will open interactive windows
    }

    # --- 4. Get user selection and run tests ---
    selected_isolators = get_user_isolator_choice(isolator_options)

    for isolator_name in selected_isolators:
        print("\n" + "="*60)
        print(f"🧪 Testing Isolator: {isolator_name}")
        print("="*60)
        isolator_instance = isolator_options[isolator_name]
        fixed_subset, moving_subset = isolator_instance.apply_isolator(pc1, pc2, bplt=True)
        print(f"✅ Isolation with '{isolator_name}' complete. Fixed subset: {len(fixed_subset)} points, Moving subset: {len(moving_subset)} points.")

    print("\n🎉 All selected isolator tests completed!")
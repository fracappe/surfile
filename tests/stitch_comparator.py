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
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import surfile.measfile_io as fio
from surfile.stitcher import stitcher as sstitcher
from surfile.stitcher import comparator as scomparator
from surfile.stitcher.plotter import show_point_clouds


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


def run_stitching_comparison(stitching_results):
    """
    Main function to run the stitching comparison test.
    """
    # --- 5. Compare Results using Comparator ---
    if len(stitching_results) < 2:
        print("\n⚠️ Skipping comparison.")
        return

    print("\n" + "="*60)
    print("📊 Comparing stitching results using comparator...")
    print("="*60)

    comparator = scomparator.Comparator(stitching_results)
    comparator.compute_all_deltas()
    # fig = comparator.plot_deltas(noise_threshold=0.01, labels=list(stitching_results.keys()), figsize_scale=4.0)
    comparator.colormap_deltas(mode='xyz')

    # Save the comparison figure
    # comparison_save_path = os.path.join(folder_path, "stitch_comparison.png")
    # fig.savefig(comparison_save_path, dpi=150, bbox_inches='tight')
    # print(f"💾 Comparison figure saved to: {comparison_save_path}")

    plt.show()

    print("\n🎉 Stitching comparison complete!")


if __name__ == "__main__":
    from stitch_stitcher import apply_stitch_sequence
    stitching_results = apply_stitch_sequence(folder='G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth', downsample=3, bplt=False)

    run_stitching_comparison(stitching_results)
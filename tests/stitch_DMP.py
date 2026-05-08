"""
stitch_DMP.py
=============
An interactive test script for evaluating stitching methods using the
Dense Map Posterior (DMP) method from surfile.stitcher.DMP.

This script will:
1. Load stitching results from the stitch_stitcher test.
2. Execute all available stitching methods.
3. Use the DMP module to evaluate and rank the results.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import surfile.measfile_io as fio
from surfile.stitcher import stitcher as sstitcher
from surfile.stitcher import DMP as sDMP


def run_dmp_evaluation(stitching_results):
    """
    Main function to run the DMP evaluation on stitching results.
    
    Parameters
    ----------
    stitching_results : dict
        Dictionary mapping method names to tuples of (stitched_surface, point_clouds_T).
    """
    # --- Evaluate Results using DMP ---
    if len(stitching_results) < 2:
        print("\n⚠️ At least 2 stitching methods required for DMP evaluation.")
        return

    print("\n" + "="*70)
    print("🔍 Evaluating stitching results using Dense Map Posterior (DMP)...")
    print("="*70)

    dmp = sDMP.DensityMapPosterior(stitching_results)
    dmp.compute_all_dmp_errors()
    
    # Get and display the best method
    best_method, best_error = dmp.get_best_method()
    print(f"\n✅ Best method: {best_method}")
    print(f"   Total DMP error: {best_error:.6f}")
    
    # Display all errors
    all_errors = dmp.get_dmp_errors()
    print("\n" + "="*70)
    print("📋 Detailed DMP Error Analysis")
    print("="*70)
    for method_name, errors in all_errors.items():
        print(f"\n{method_name}:")
        print(f"  Total error:        {errors['total_error']:.6f}")
        print(f"  Mean error/cloud:   {errors['mean_error_per_cloud']:.6f}")
        print(f"  Per-cloud errors:   {[f'{e:.6f}' for e in errors['per_cloud_errors']]}")

    print("\n🎉 DMP evaluation complete!")
    
    return dmp


if __name__ == "__main__":
    from stitch_stitcher import apply_stitch_sequence
    
    # Run stitching on test data
    print("🔄 Running stitching methods...")
    stitching_results = apply_stitch_sequence(
        folder='G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth', 
        downsample=3, 
        bplt=False
    )
    
    # Run DMP evaluation
    dmp_evaluator = run_dmp_evaluation(stitching_results)

from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import surfile.stitcher.pipeline as spipe
import surfile.stitcher.stitcher as sstitch

if __name__ == "__main__":
    from tests.file_helper import *

    # folder = 'G:\\\Drive condivisi\\TIROCINI\\2025 - Francesca Capellino\\Tesi\\misure\\stitching_globale\\down_5'
    folder = 'G:\\Drive condivisi\\TIROCINI\\2025 - Francesca Capellino\\Tesi\\misure\\prova_2\\nuvole_di_punti_txt'
    pcs, folder_path = open_files(folder=folder, downsample=50, userscales=[1, 1, 1])

    stitcher_specs = spipe.build_default_stitcher_specs(
        build_methods=['icp','rmse'],
        rmse_n_calls=12,
        icp_threshold_type='KDTree',
        fgr_voxel_size=0.05,
        fgr_threshold_type='KDTree',
        bplt=False,
    )

    isolator_specs = spipe.build_default_isolator_specs(
        build_methods=['KDTree'],
        max_min_axes='xy'
    )

    builder = spipe.MultilevelPipelineBuilder(
            mode=spipe.PipelineMode.PROGRESSIVE_METHODS,
            max_depth=2,
            stitcher_specs=stitcher_specs,
            isolator_specs=isolator_specs,
            save_intermediate=True,
            manual_step_name='pt_manual',
            pipeline_name='multilevel_evaluation_pipeline_test',
    )

   # --- Example: Mode 1, depth 1 ---
    runner = spipe.MultilevelEvaluationRunner(
        builder,
        evaluators=['ballquery'],
        evaluate_intermediate=True,
    )

    sstitch.SurfaceStitcher.cloud_combiner_max_points = 50000
    sstitch.SurfaceStitcher.plotter_colors = ['normal', 'uniform'] # try unicorn
    results = runner.run(
        pcs=pcs,
        folder_path=folder_path,
        bplt_pipeline=False,
        bplt_comparators=True,
    )

    # # --- Example: Mode 1, depth 2 ---
    # runner = spipe.MultilevelEvaluationRunner(
    #     mode=spipe.PipelineMode.PROGRESSIVE_METHODS,
    #     max_depth=2,
    #     isolator_specs=isolator_specs,
    #     stitcher_specs=stitcher_specs,
    #     evaluators=['ballquery', 'dmp'],  # ['dmp'],  ['']
    #     evaluate_intermediate=True,
    # )

    # sstitch.SurfaceStitcher.cloud_combiner_max_points = 200000
    # sstitch.SurfaceStitcher.plotter_colors = ['normal', 'uniform'] # try unicorn
    # results = runner.run(
    #     pcs=pcs,
    #     folder_path=folder_path,
    #     bplt_pipeline=False,
    #     bplt_comparators=True,
    # )

    # # --- Example: Mode 2, depth 2 ---
    # runner_mode2 = MultilevelEvaluationRunner(
    #     mode=PipelineMode.PROGRESSIVE_ISOLATORS,
    #     max_depth=2,
    #     isolator_specs=isolator_specs,
    #     stitcher_specs=stitcher_specs,
    #     evaluators=['ballquery', 'dmp'],  # ['dmp'],  ['']
    #     evaluate_intermediate=False,
    # )
    # results_mode2 = runner_mode2.run(
    #     pcs=pcs,
    #     folder_path=folder_path,
    #     bplt_pipeline=True,
    #     bplt_comparators=True,
    # )
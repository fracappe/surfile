from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import surfile.stitcher.pipeline as spipe

if __name__ == "__main__":
    from tests.file_helper import *

    folder = 'G:\\Drive condivisi\\TIROCINI\\2025 - Francesca Capellino\\Tesi\\misure\\stitching_parziale_alto'
    pcs, folder_path = open_files(folder=folder, downsample=1, userscales=[1, 1, 1])

    stitcher_specs = spipe.build_default_stitcher_specs(
        build_methods=['icp'],
        rmse_n_calls=30,
        icp_threshold_type='KDTree',
        fgr_voxel_size=0.05,
        fgr_threshold_type='KDTree',
        bplt=True,
    )

    runner = spipe.MultilevelEvaluationRunner(
        mode=spipe.PipelineMode.PROGRESSIVE_METHODS,
        max_depth=1,
        isolator_names=['maxmin'],
        stitcher_specs=stitcher_specs,
        evaluate_intermediate=False,
    )

    results = runner.run(
        pcs=pcs,
        folder_path=folder_path,
        bplt_pipeline=True,
        bplt_comparators=True,
    )

    # # --- Example: Mode 1, depth 2 ---
    # runner = MultilevelEvaluationRunner(
    #     mode=PipelineMode.PROGRESSIVE_METHODS,
    #     max_depth=2,
    #     stitcher_specs=stitcher_specs,
    #     evaluate_intermediate=False,
    # )
    # results = runner.run(
    #     pcs=pcs,
    #     folder_path=folder_path,
    #     bplt_pipeline=True,
    #     bplt_comparators=True,
    # )

    # # --- Example: Mode 2, depth 2 ---
    # runner_mode2 = MultilevelEvaluationRunner(
    #     mode=PipelineMode.PROGRESSIVE_ISOLATORS,
    #     max_depth=2,
    #     stitcher_specs=stitcher_specs,
    #     evaluate_intermediate=False,
    # )
    # results_mode2 = runner_mode2.run(
    #     pcs=pcs,
    #     folder_path=folder_path,
    #     bplt_pipeline=True,
    #     bplt_comparators=True,
    # )
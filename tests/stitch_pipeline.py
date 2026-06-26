import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from surfile.stitcher import stitcher as sst
from surfile.stitcher import pipeline as spipe
from surfile.stitcher import comparator

import matplotlib.pyplot as plt

from tests.file_helper import *

step_man = spipe.PipelineStep(sst.SurfaceStitcher.stitchManual, recall_from_passtrough='pt_manual', bplt=True)
step_icp_ch = spipe.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_ch", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='convex_hull'), bplt=True)
step_icp_mm = spipe.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_mm", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='maxmin', axes='xy'), bplt=True)

step_man.add_child(spipe.PipelineStep.pass_through(name="pt_manual"))
step_man.add_child(step_icp_ch)
step_man.add_child(step_icp_mm)

pipe = spipe.TreePipeline(root_steps=[step_man], name="manpt_icp_pipe_t3")

step_man_CAD = spipe.PipelineStepq(sst.SurfaceStitcher.stitchManual, recall_from_passtrough='pt_manual_cad', bplt=True)
step_icp_CAD = spipe.PipelineStep(sst.SurfaceStitcher.stitchICP, name="icp_cad", thresholder=sst.Thresholder(type='KDTree'), isolator=sst.Isolator(type='KDTree'), bplt=True)

step_man_CAD.add_child(spipe.PipelineStep.pass_through(name="pt_manual_cad"))
step_man_CAD.add_child(step_icp_CAD)

pipe_CAD = spipe.TreePipeline(root_steps=[step_man_CAD], name="manual_top_cad_pipe")

if __name__ == "__main__":
    folder = 'G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth_test_Andrea'
    pcs, folder_path = open_files(folder=folder, downsample=3, userscales=[1, 1, 1])
    stitching_results = pipe.run(pcs, folder_path, bplt=False)

    # top_pc, top_path = open_file(path='G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth\\Aocclusale_coordinate.txt_resaved.npy', downsample=1, userscales=[1, 1, 1])
    cad_pc, cad_path = open_file(path='G:\\Drive condivisi\\TIROCINI\\2025 - Francesca Capellino\\Tesi\\Corone\\corona_cad.stl', downsample=1, userscales=[1000, 1000, 1000])

    bq = comparator.CAD(stitching_results, cad_pc, pipeline=pipe_CAD, pipeline_path=folder, names=None)
    bq.print_summary()

    plt.show()
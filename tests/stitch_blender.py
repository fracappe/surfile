"""
Tests blender functionality from the utils submodule
"""
import tkinter as tk
from tkinter import filedialog

import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from surfile import measfile_io as fio
from surfile.stitcher.utils import blender_edit_point_cloud
import surfile.stitcher.plotter as splt
from file_helper import open_file

if __name__ == "__main__":
    # open first tooth cloud
    file = 'G:\\Drive condivisi\\TIROCINI\\2026 - Aysu Oral\\figures\\tooth_test_Andrea\\Aocclusale_coordinate.txt_resaved.npy'
    pc, _ = open_file(file, userscales=[1, 1, 1])
    
    edited_pc = blender_edit_point_cloud(pc)
    
    # splt.compare_point_clouds([pc, edited_pc], colors='plasma')
    splt.show_point_clouds([pc, edited_pc], colors='uniform')
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_hopped
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_unhopped
from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator
import matplotlib.pyplot as plt
import time
import pickle
import resource
from hd2d_src.HopTrack.utils import *

with open('/home/xiaochu/Public/LUV_in_HD/data/input/0785/a_AVE_normalized.obj', "rb") as f:
# with open('/home/xiaochu/Public/LUV_in_HD/data/input/0805/a_cut_69500.obj', "rb") as f:
    custom_unpickler = CustomUnpickler(f)
    a = custom_unpickler.load()
print(np.shape(a.frames))
print(a.L)
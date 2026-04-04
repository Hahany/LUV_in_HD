#! /bin/python3

import os
import sys
from hd2d_src.HopTrack.utils.path import get_sub_project_root
from matplotlib import pyplot as plt
import numpy as np
from hd2d_src.HopTrack.utils import *
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_hopped

project_root=get_sub_project_root()
obj_a_path = f'{project_root}/data/input/0805/a_AVE_normalized.obj'
current_path = os.path.abspath(__file__)
save_dir =  os.path.dirname(current_path)
Rc = 6.0
rh = 0.8
# dt_list = [10, 50, 100, 200, 500, 1000]
# dt_list = [2000, 5000, 10000, 50000, 100000]
dt_list = [25000]
with open(obj_a_path, "rb") as f:
    custom_unpickler = CustomUnpickler(f)
    a = custom_unpickler.load()
for dt in dt_list:
    ld_hp = ld_hopped(a, dt, Rc, rh)
    ld_hp = np.array(ld_hp)
    np.savetxt(f"{save_dir}/ld_hopped_dt_{dt}.txt",
               ld_hp.T, delimiter='\t', fmt='%.6f')
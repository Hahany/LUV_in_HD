#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator
from hd2d_src.HopTrack.utils import *
import matplotlib.pyplot as plt



if __name__ == '__main__':
    obj_a_path = '/home/xiaochu/Public/LUV_in_HD/data/input/0805/a_cut_69500.obj'
    with open(obj_a_path, "rb") as f:
        custom_unpickler = CustomUnpickler(f)
        a = custom_unpickler.load()
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ## check tail head and middle particle id
    # a.unwrap_vec()
    # a.coarsening(20)
    # a.chbox(23, 47, 55, 75)
    # a.hops(0.5)
    # a.showdisp(fig, ax, t_start=0, t_end=20, lw=2, ms=1.5, nodot=True, showvoid=False, 
    #            showpid=True, showforcepid=0, showforcedid=0,
    #            colorbar=False, showradii=False, overlap=3, crange=[0, 19])
    tail_id = 7753
    tail_id = a.frames[0, :, 0] == tail_id

    head_id = 3516
    head_id = a.frames[0, :, 0] == head_id

    mid_id = 13477
    mid_id = a.frames[0, :, 0] == mid_id


    # ax.plot(a.frames[0, tail_id, 2], a.frames[0, tail_id, 3], 's', color='r', ms=10)
    # ax.plot(a.frames[0, head_id, 2], a.frames[0, head_id, 3], 'd', color='g', ms=10)
    # ax.plot(a.frames[0, mid_id, 2], a.frames[0, mid_id, 3], 'd', color='g', ms=10)

    disp_tail = a.frames[:, tail_id, 2:4] - a.frames[0, tail_id, 2:4]
    ds_tail = np.linalg.norm(disp_tail, axis=-1).squeeze()
    ax.plot(np.arange(len(ds_tail))*41, ds_tail, 'rs-')

    disp_mid = a.frames[:, mid_id, 2:4] - a.frames[0, mid_id, 2:4]
    ds_mid = np.linalg.norm(disp_mid, axis=-1).squeeze()
    ax.plot(np.arange(len(ds_mid))*41, ds_mid, 'gv-')

    disp_head = a.frames[:, head_id, 2:4] - a.frames[0, head_id, 2:4]
    ds_head = np.linalg.norm(disp_head, axis=-1).squeeze()
    ax.plot(np.arange(len(ds_head))*41, ds_head, 'bd-')



    ax.set_xlabel(r'$t$', fontsize=24)
    ax.set_ylabel(r'$|r_i^C(t) - r_i^C(0)|$', fontsize=24)
    ax.grid()
    ax.legend([r'$Tail$', r'$Middle$', r'$Head$'], fontsize=20)
    ax.tick_params(axis='both', labelsize=24)
    plt.tight_layout()


    np.savetxt('/home/xiaochu/Public/LUV_in_HD/apps/LUV_vs_time/r_vs_t.txt', np.array([np.arange(len(ds_tail))*41, ds_tail, ds_mid, ds_head]).T)
    fig.savefig('/home/xiaochu/Public/LUV_in_HD/apps/LUV_vs_time/r_vs_t.png', dpi = 600)
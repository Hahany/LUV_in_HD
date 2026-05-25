from hd2d_src.HopTrack.core.share.distPBC2D import distPBC2D
import numpy as np

def dr_overlap(glass, time, ht_pid, pick_list):
    if pick_list:
        dr = glass.frames[time, ht_pid, 2:4] - glass.frames[time, pick_list, 2:4]
        for i_dr in range(len(dr)):
            dr[i_dr] = distPBC2D(dr[i_dr], glass.L[0], glass.L[1])
        dr_norm = np.linalg.norm(dr, axis=1)
        dr_min = np.min(dr_norm)
        return dr_min
    else:
        return 0
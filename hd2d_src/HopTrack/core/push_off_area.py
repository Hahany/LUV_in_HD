import numpy as np
from hd2d_src.HopTrack.core.share.particle_unwrap import particle_unwrap
from hd2d_src.HopTrack.core.share.circle_intersection_area import circle_intersection_area

def push_off_area(glass, WL=False):
    V_push_t = []
    V_push_h = []
    ts, N, _ = np.shape(glass.frames)
    new_startend = glass.starend_of_string
    for i, stringparticles in enumerate(glass.connected_components):
        starttime = new_startend[i][0]
        endtime = new_startend[i][1]
        if WL:  # plot quasivoid location at the first and final frame of the duration
            t_si = 0
            t_sf = -1
        else:
            t_si = starttime
            t_sf = endtime

        x_Qti = glass.frames[t_si, stringparticles[0], 2]
        y_Qti = glass.frames[t_si, stringparticles[0], 3]
        x_Qtf = glass.frames[t_sf, stringparticles[0], 2]
        y_Qtf = glass.frames[t_sf, stringparticles[0], 3]
        x_Qhi = glass.frames[t_si, stringparticles[-1], 2]
        y_Qhi = glass.frames[t_si, stringparticles[-1], 3]
        x_Qhf = glass.frames[t_sf, stringparticles[-1], 2]
        y_Qhf = glass.frames[t_sf, stringparticles[-1], 3]

        r_f = glass.radii[stringparticles[0]]
        x_Qtf = particle_unwrap(x_Qti, x_Qtf, glass.L[0])
        y_Qtf = particle_unwrap(y_Qti, y_Qtf, glass.L[1])
        S_overlap = circle_intersection_area(x_Qtf, y_Qtf, r_f, x_Qti, y_Qti, r_f)
        v_q = np.pi * r_f ** 2 - S_overlap
        V_push_t.append(v_q)

        r_f = glass.radii[stringparticles[-1]]
        x_Qhf = particle_unwrap(x_Qhi, x_Qhf, glass.L[0])
        y_Qhf = particle_unwrap(y_Qhi, y_Qhf, glass.L[1])
        S_overlap = circle_intersection_area(x_Qhf, y_Qhf, r_f, x_Qhi, y_Qhi, r_f)
        v_q = np.pi * r_f ** 2 - S_overlap
        V_push_h.append(v_q)
    return V_push_t, V_push_h


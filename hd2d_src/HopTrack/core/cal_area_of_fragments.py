import numpy as np
from hd2d_src.HopTrack.core.share.select_region_pbc import select_region_pbc
from shapely.geometry import Point

def cal_area_of_fragments(glass, Rc, WL=False, entire_sys=False):
    V_frag_ti = []
    V_frag_hi = []
    V_frag_tf = []
    V_frag_hf = []
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


        def get_region_mask(CalRegion_pbc_frames, circle_center, Rc, t_s):
            x_Q, y_Q = circle_center
            skin = max(glass.radii)
            xlow = np.mod(x_Q - Rc - skin, glass.L[0])
            xup = np.mod(x_Q + Rc + skin, glass.L[0])
            ylow = np.mod(y_Q - Rc - skin, glass.L[1])
            yup = np.mod(y_Q + Rc + skin, glass.L[1])
            if xlow > xup:
                xregion = (CalRegion_pbc_frames[t_s, :, 2] > xlow) | (CalRegion_pbc_frames[t_s, :, 2] < xup)
            else:
                xregion = (CalRegion_pbc_frames[t_s, :, 2] > xlow) & (CalRegion_pbc_frames[t_s, :, 2] < xup)
            if ylow > yup:
                yregion = (CalRegion_pbc_frames[t_s, :, 3] > ylow) | (CalRegion_pbc_frames[t_s, :, 3] < yup)
            else:
                yregion = (CalRegion_pbc_frames[t_s, :, 3] > ylow) & (CalRegion_pbc_frames[t_s, :, 3] < yup)
            region_mask = xregion & yregion
            return region_mask

        def calc_fragments(circle_center, Rc, t_if):
            x_Q, y_Q = circle_center
            t_i, t_f = t_if
            CalRegion_pbc_frames, _, _, _, _ = select_region_pbc(glass, x_Q, y_Q, Rc)
            mask_i = get_region_mask(CalRegion_pbc_frames, circle_center, Rc, t_i)
            mask_f = get_region_mask(CalRegion_pbc_frames, circle_center, Rc, t_f)
            mask = mask_i | mask_f
            particle_in_region_i = CalRegion_pbc_frames[t_i, mask, :]
            particle_index = CalRegion_pbc_frames[t_i, mask, 0] - 1
            r_i = [glass.radii[int(i)] for i in particle_index]
            particle_in_region_f = CalRegion_pbc_frames[t_f, mask, :]
            particle_index = CalRegion_pbc_frames[t_f, mask, 0] - 1
            r_f = [glass.radii[int(i)] for i in particle_index]

            def create_circle(center, radius):
                return Point(center).buffer(radius)

            circle1 = create_circle((x_Q, y_Q), Rc)
            S_fragments = 0
            for p_index, p_i in enumerate(particle_in_region_i):
                particel_index_i = p_i[0]
                x_i = p_i[2]
                y_i = p_i[3]
                r_pi = r_i[p_index]
                circle2 = create_circle((x_i, y_i), r_pi)
                for p_findex, p_f in enumerate(particle_in_region_f):
                    particel_index_f = p_f[0]
                    x_f = p_f[2]
                    y_f = p_f[3]
                    r_pf = r_f[p_findex]
                    circle3 = create_circle((x_f, y_f), r_pf)
                    S_in3 = circle1.intersection(circle2).intersection(circle3)
                    if particel_index_i == particel_index_f:
                        S_in2 = circle1.intersection(circle3)
                        S_fragments = S_fragments + (S_in2.area - S_in3.area)
                        # print(f'S_in2-S_in3 is {S_in2.area-S_in3.area}')
                    else:
                        # print(f'S_in3 is {S_in3.area}')
                        S_fragments = S_fragments - S_in3.area
            return S_fragments

        s_frag_ti = calc_fragments((x_Qti, y_Qti), Rc, (t_si, t_sf))
        V_frag_ti.append(s_frag_ti)
        s_frag_tf = calc_fragments((x_Qtf, y_Qtf), Rc, (t_sf, t_si))
        V_frag_tf.append(s_frag_tf)
        s_frag_hi = calc_fragments((x_Qhi, y_Qhi), Rc, (t_si, t_sf))
        V_frag_hi.append(s_frag_hi)
        s_frag_hf = calc_fragments((x_Qhf, y_Qhf), Rc, (t_sf, t_si))
        V_frag_hf.append(s_frag_hf)
    if entire_sys:
        v_fragi_entire = []
        v_fragf_entire = []
        for t in range(5):
            for i in range(N):
                v_fragi = calc_fragments((glass.frames[t, i, 2], glass.frames[t, i, 3]), Rc, (t, t+1))
                v_fragf = calc_fragments((glass.frames[t+1, i, 2], glass.frames[t+1, i, 3]), Rc, (t+1, t))
                v_fragi_entire.append(v_fragi)
                v_fragf_entire.append(v_fragf)
        return V_frag_ti, V_frag_tf, V_frag_hi, V_frag_hf, v_fragi_entire, v_fragf_entire
    else:
        return V_frag_ti, V_frag_tf, V_frag_hi, V_frag_hf


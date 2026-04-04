import matplotlib.pyplot as plt
import numpy as np
from hd2d_src.HopTrack.core.share import *
import matplotlib.colors as mcolors
from tqdm import tqdm
from hd2d_src.HopTrack.viz import *
from hd2d_src.HopTrack.core import *
from hd2d_src.HopTrack.core.Local_Density_Calculator import *

def luv_fix_cr(glass, regionradius, rho, savefile=None, figshow=False, dr_ht=5):
    # fix the postion of measurement region at string tail and head at pre-hop and post-hop position. For getting the V_T(pre-hop) - V_T(post-hop) > 0 
    # crt_in = False  whether set Rc as criteria for the internal particles in the string


    # get particle id in string
    density_tail_ini = []
    density_head_ini = []
    density_tail_final = []
    density_head_final = []
    ld_in_string = []
    time_select = []
    ld_list = []
    ts, _, _ = np.shape(glass.frames)
    Deduplicate(glass)
    deepDeduplicate(glass)
    for ti, stringparticles in enumerate(tqdm(glass.connected_components, desc="Processing Components")):
        if figshow:
            fig0, ax0 = plt.subplots(2, 2, figsize=(8, 8))

        sp = [stringparticles[0], stringparticles[-1]]
        time_ini = glass.starend_of_string[ti][0]
        time_final = glass.starend_of_string[ti][1]
        # print(f'time initial = {time_ini}, time final = {time_final}')
        # get partile and quasivoid position at the tail
        id_tail = int(sp[0])
        id_head = int(sp[1])
        x_particle_tail = glass.frames[time_ini, id_tail, 2]
        y_particle_tail = glass.frames[time_ini, id_tail, 3]
        x_quasivoid_tail = glass.frames[time_final, id_tail, 2]
        y_quasivoid_tail = glass.frames[time_final, id_tail, 3]
        x_quasivoid_tail = particle_unwrap(x_particle_tail, x_quasivoid_tail, glass.L[0]) # unwrap when particle hops to the otherside
        y_quasivoid_tail = particle_unwrap(y_particle_tail, y_quasivoid_tail, glass.L[1])

        # get partile and quasivoid position at the head
        x_particle_head = glass.frames[time_final, id_head, 2]
        y_particle_head = glass.frames[time_final, id_head, 3]
        x_quasivoid_head = glass.frames[time_ini, id_head, 2]
        y_quasivoid_head = glass.frames[time_ini, id_head, 3]
        x_quasivoid_head = particle_unwrap(x_particle_head, x_quasivoid_head, glass.L[0])
        y_quasivoid_head = particle_unwrap(y_particle_head, y_quasivoid_head, glass.L[1])

        dist_headtail_vec = distPBC2D([x_particle_head-x_particle_tail, y_particle_head-y_particle_tail],
                                        glass.L[0], glass.L[1])
        dist_headtail = np.linalg.norm(dist_headtail_vec)
        if dist_headtail > dr_ht:
            CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti, y_up_ti = select_region_pbc(glass, x_particle_tail, y_particle_tail, regionradius)
            CalRegion_tail_final, x_low_tf, x_up_tf, y_low_tf, y_up_tf = CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti, y_up_ti
            CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf, y_up_hf = select_region_pbc(glass, x_particle_head, y_particle_head, regionradius)
            CalRegion_head_ini, x_low_hi, x_up_hi, y_low_hi, y_up_hi = CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf, y_up_hf




            # plot local density vs time
            ld_ti = cal_density_on_time(glass, CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti, y_up_ti,
                                        x_particle_tail, y_particle_tail, regionradius, time_ini)
            ld_tf = cal_density_on_time(glass, CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti,
                                        y_up_ti, x_particle_tail, y_particle_tail, regionradius, time_final)
            ld_hf = cal_density_on_time(glass, CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf, y_up_hf,
                                        x_particle_head, y_particle_head, regionradius, time_final)
            ld_hi = cal_density_on_time(glass, CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf, y_up_hf,
                                        x_particle_head, y_particle_head, regionradius, time_ini)
            
            t = time_ini
            mask_ti = ((CalRegion_tail_ini[t, :, 2] > x_low_ti) & (CalRegion_tail_ini[t, :, 2] < x_up_ti) &
                        (CalRegion_tail_ini[t, :, 3] > y_low_ti) & (CalRegion_tail_ini[t, :,
                                                                    3] < y_up_ti))  # select particles in the square region to speed up
            mask_tf = mask_ti

            t = time_final
            mask_hf = ((CalRegion_head_final[t, :, 2] > x_low_hf) & (
                        CalRegion_head_final[t, :, 2] < x_up_hf) &
                        (CalRegion_head_final[t, :, 3] > y_low_hf) & (
                                    CalRegion_head_final[t, :, 3] < y_up_hf))
            mask_hi = mask_hf

            density_tail_ini.append(ld_ti)
            density_tail_final.append(ld_tf)
            density_head_final.append(ld_hf)
            density_head_ini.append(ld_hi)
            time_select = [time_ini, time_final]
            
            if figshow and ( ld_ti > ld_tf ):

                entire_density_initial = Local_Density_Calculator(glass, regionradius, [time_ini])
                entire_density_final = Local_Density_Calculator(glass, regionradius, [time_final])
                norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                show_heterogeneity(ax0[0, 0], glass, entire_density_initial, norm, time_ini,
                                        [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                show_heterogeneity(ax0[1, 0], glass, entire_density_initial, norm, time_ini,
                                        [x_low_hf, x_up_hf, y_low_hf, y_up_hf])
                show_heterogeneity(ax0[0, 1], glass, entire_density_final, norm, time_final,
                                        [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                show_heterogeneity(ax0[1, 1], glass, entire_density_final, norm, time_final,
                                        [x_low_hf, x_up_hf, y_low_hf, y_up_hf])

                # ax, x1, y1, regionradius, x_low, x_up, y_low, y_up, x_qt, y_qt, x_qh, y_qh
                set_ax(glass, ax0[0, 0], x_particle_tail, y_particle_tail, regionradius, x_low_ti, x_up_ti,
                                y_low_ti,
                                y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                y_quasivoid_head)
                set_ax(glass, ax0[0, 1], x_particle_tail, y_particle_tail, regionradius, x_low_ti, x_up_ti,
                                y_low_ti,
                                y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                y_quasivoid_head)
                set_ax(glass, ax0[1, 0], x_particle_head, y_particle_head, regionradius, x_low_hf, x_up_hf,
                                y_low_hf,
                                y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                y_quasivoid_head)
                set_ax(glass, ax0[1, 1], x_particle_head, y_particle_head, regionradius, x_low_hf, x_up_hf,
                                y_low_hf,
                                y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                y_quasivoid_head)

                # plot string
                plot_collections(ax0[0, 0], mask_ti, CalRegion_tail_ini, time_ini, time_final, ts, ti,
                                        'initial tail')
                plot_collections(ax0[0, 1], mask_tf, CalRegion_tail_final, time_ini, time_final, ts, ti,
                                        'final tail')
                plot_collections(ax0[1, 0], mask_hi, CalRegion_head_ini, time_ini, time_final, ts, ti,
                                        'initial head')
                plot_collections(ax0[1, 1], mask_hf, CalRegion_head_final, time_ini, time_final, ts, ti,
                                        'final head')


                fig0.savefig(f'{savefile}/string{ti:03d}_config.png', dpi=600)
                plt.close('all')


    return density_tail_ini, density_tail_final, density_head_ini, density_head_final, time_select
        
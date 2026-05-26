import matplotlib.pyplot as plt
import numpy as np
from hd2d_src.HopTrack.core.share import *
import matplotlib.colors as mcolors
from tqdm import tqdm
from hd2d_src.HopTrack.viz import *
from hd2d_src.HopTrack.core import *
from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator



# mode = 0  calculate the local density at string ends
# mode = 1 calculate the local density of the middle/other (without ends) particles in the string
# mode = 2 calculate minial local density of the loop 
# crt_in = False  whether set Rc as criteria for the internal particles in the string

def showdensity(glass, regionradius, r_insert, rho, savefile=None, figshow=False,
                pid=None, select=False, mode=0, Length_of_string=3, crt_in=False, choose_middle=False, dr_ht=5):

    density_tail_ini = []
    density_head_ini = []
    density_tail_final = []
    density_head_final = []
    ld_in_string = []
    time_select = []
    ld_list = []
    ts, _, _ = np.shape(glass.frames)
    # Deduplicate 和 deepDeduplicate 已在 prepare_string_obj 里执行，此处删除

    for ti, stringparticles in enumerate(tqdm(glass.connected_components, desc="Processing Components")):
        if select:
            if pid != stringparticles[0]:
                continue
        if mode == 0:
            if figshow:
                fig4, ax4 = plt.subplots()
                fig0, ax0 = plt.subplots(2, 2, figsize=(8, 8))

            sp = [stringparticles[0], stringparticles[-1]]
            time_ini = glass.starend_of_string[ti][0]
            time_final = glass.starend_of_string[ti][1]

            id_tail = int(sp[0])
            id_head = int(sp[1])
            x_partical_tail = glass.frames[time_ini, id_tail, 2]
            y_particle_tail = glass.frames[time_ini, id_tail, 3]
            x_quasivoid_tail = glass.frames[time_final, id_tail, 2]
            y_quasivoid_tail = glass.frames[time_final, id_tail, 3]
            x_quasivoid_tail = particle_unwrap(x_partical_tail, x_quasivoid_tail, glass.L[0])
            y_quasivoid_tail = particle_unwrap(y_particle_tail, y_quasivoid_tail, glass.L[1])

            x_particle_head = glass.frames[time_final, id_head, 2]
            y_particle_head = glass.frames[time_final, id_head, 3]
            x_quasivoid_head = glass.frames[time_ini, id_head, 2]
            y_quasivoid_head = glass.frames[time_ini, id_head, 3]
            x_quasivoid_head = particle_unwrap(x_particle_head, x_quasivoid_head, glass.L[0])
            y_quasivoid_head = particle_unwrap(y_particle_head, y_quasivoid_head, glass.L[1])

            dist_headtail_vec = distPBC2D([x_particle_head-x_partical_tail, y_particle_head-y_particle_tail],
                                          glass.L[0], glass.L[1])
            dist_headtail = np.linalg.norm(dist_headtail_vec)

            if dist_headtail > dr_ht:
                frame_ti, x_low_ti, x_up_ti, y_low_ti, y_up_ti = select_region_pbc(
                    glass, x_partical_tail, y_particle_tail, regionradius, time_ini)
                frame_tf, x_low_tf, x_up_tf, y_low_tf, y_up_tf = select_region_pbc(
                    glass, x_quasivoid_tail, y_quasivoid_tail, regionradius, time_final)
                frame_hf, x_low_hf, x_up_hf, y_low_hf, y_up_hf = select_region_pbc(
                    glass, x_particle_head, y_particle_head, regionradius, time_final)
                frame_hi, x_low_hi, x_up_hi, y_low_hi, y_up_hi = select_region_pbc(
                    glass, x_quasivoid_head, y_quasivoid_head, regionradius, time_ini)

                if r_insert == 0:
                    ld_ti = cal_density_on_time(glass, frame_ti, x_low_ti, x_up_ti, y_low_ti, y_up_ti,
                                                x_partical_tail, y_particle_tail, regionradius)
                    ld_tf = cal_density_on_time(glass, frame_tf, x_low_tf, x_up_tf, y_low_tf, y_up_tf,
                                                x_quasivoid_tail, y_quasivoid_tail, regionradius)
                    ld_hf = cal_density_on_time(glass, frame_hf, x_low_hf, x_up_hf, y_low_hf, y_up_hf,
                                                x_particle_head, y_particle_head, regionradius)
                    ld_hi = cal_density_on_time(glass, frame_hi, x_low_hi, x_up_hi, y_low_hi, y_up_hi,
                                                x_quasivoid_head, y_quasivoid_head, regionradius)

                    density_tail_ini.append(ld_ti)
                    density_tail_final.append(ld_tf)
                    density_head_final.append(ld_hf)
                    density_head_ini.append(ld_hi)
                    time_select = [time_ini, time_final]

                    if figshow:
                        CalRegion_ti = select_region_pbc_full(glass, x_partical_tail, y_particle_tail, regionradius)
                        CalRegion_tf = select_region_pbc_full(glass, x_quasivoid_tail, y_quasivoid_tail, regionradius)
                        CalRegion_hf = select_region_pbc_full(glass, x_particle_head, y_particle_head, regionradius)
                        CalRegion_hi = select_region_pbc_full(glass, x_quasivoid_head, y_quasivoid_head, regionradius)

                        mask_ti = ((frame_ti[:, 2] > x_low_ti) & (frame_ti[:, 2] < x_up_ti) &
                                   (frame_ti[:, 3] > y_low_ti) & (frame_ti[:, 3] < y_up_ti))
                        mask_tf = ((frame_tf[:, 2] > x_low_tf) & (frame_tf[:, 2] < x_up_tf) &
                                   (frame_tf[:, 3] > y_low_tf) & (frame_tf[:, 3] < y_up_tf))
                        mask_hf = ((frame_hf[:, 2] > x_low_hf) & (frame_hf[:, 2] < x_up_hf) &
                                   (frame_hf[:, 3] > y_low_hf) & (frame_hf[:, 3] < y_up_hf))
                        mask_hi = ((frame_hi[:, 2] > x_low_hi) & (frame_hi[:, 2] < x_up_hi) &
                                   (frame_hi[:, 3] > y_low_hi) & (frame_hi[:, 3] < y_up_hi))

                        entire_density_initial = Local_Density_Calculator(glass, regionradius, [time_ini])
                        entire_density_final = Local_Density_Calculator(glass, regionradius, [time_final])
                        norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                        show_heterogeneity(ax0[0, 0], glass, entire_density_initial, norm, time_ini,
                                           [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                        show_heterogeneity(ax0[1, 0], glass, entire_density_initial, norm, time_ini,
                                           [x_low_hi, x_up_hi, y_low_hi, y_up_hi])
                        show_heterogeneity(ax0[0, 1], glass, entire_density_final, norm, time_final,
                                           [x_low_tf, x_up_tf, y_low_tf, y_up_tf])
                        show_heterogeneity(ax0[1, 1], glass, entire_density_final, norm, time_final,
                                           [x_low_hf, x_up_hf, y_low_hf, y_up_hf])
                        set_ax(glass, ax0[0, 0], x_partical_tail, y_particle_tail, regionradius, x_low_ti, x_up_ti,
                               y_low_ti, y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                        set_ax(glass, ax0[0, 1], x_quasivoid_tail, y_quasivoid_tail, regionradius, x_low_tf, x_up_tf,
                               y_low_tf, y_up_tf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                        set_ax(glass, ax0[1, 0], x_quasivoid_head, y_quasivoid_head, regionradius, x_low_hi, x_up_hi,
                               y_low_hi, y_up_hi, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                        set_ax(glass, ax0[1, 1], x_particle_head, y_particle_head, regionradius, x_low_hf, x_up_hf,
                               y_low_hf, y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                        plot_collections(glass, ax0[0, 0], mask_ti, CalRegion_ti, time_ini, time_final, ts, ti, 'initial tail')
                        plot_collections(glass, ax0[0, 1], mask_tf, CalRegion_tf, time_ini, time_final, ts, ti, 'final tail')
                        plot_collections(glass, ax0[1, 0], mask_hi, CalRegion_hi, time_ini, time_final, ts, ti, 'initial head')
                        plot_collections(glass, ax0[1, 1], mask_hf, CalRegion_hf, time_ini, time_final, ts, ti, 'final head')
                        ax4.vlines([time_ini, time_final], 0.75, 0.85, color=['r', 'b'], linestyles='--')
                        ax4.set_xlabel('Time')
                        ax4.set_ylabel('Area density in black circle region')
                        ax4.set_title(f'locale density of string #{ti}')
                        ax4.legend()
                        ax4.grid()
                        if savefile:
                            fig0.savefig(str(savefile) + f'_string{ti:03d}_config.png', dpi=600)
                            plt.close(fig0)
                            fig4.savefig(str(savefile) + f'_string{ti:03d}_tail_and_head_density.png', dpi=600)
                            plt.close(fig4)
                        else:
                            plt.show()

                else:
                    ld_ti, mindr_ti = cal_density_on_time_with_ri(glass, frame_ti, x_low_ti, x_up_ti,
                                                                   y_low_ti, y_up_ti, x_partical_tail,
                                                                   y_particle_tail, regionradius)
                    ld_tf, mindr_tf = cal_density_on_time_with_ri(glass, frame_tf, x_low_tf, x_up_tf,
                                                                   y_low_tf, y_up_tf, x_quasivoid_tail,
                                                                   y_quasivoid_tail, regionradius)
                    ld_hf, mindr_hf = cal_density_on_time_with_ri(glass, frame_hf, x_low_hf, x_up_hf,
                                                                   y_low_hf, y_up_hf, x_particle_head,
                                                                   y_particle_head, regionradius)
                    ld_hi, mindr_hi = cal_density_on_time_with_ri(glass, frame_hi, x_low_hi, x_up_hi,
                                                                   y_low_hi, y_up_hi, x_quasivoid_head,
                                                                   y_quasivoid_head, regionradius)

                    if mindr_tf > r_insert and mindr_hi > r_insert:
                        density_tail_ini.append(ld_ti)
                        density_tail_final.append(ld_tf)
                        density_head_final.append(ld_hf)
                        density_head_ini.append(ld_hi)
                        time_select = [time_ini, time_final]
                        if figshow:
                            CalRegion_ti = select_region_pbc_full(glass, x_partical_tail, y_particle_tail, regionradius)
                            CalRegion_tf = select_region_pbc_full(glass, x_quasivoid_tail, y_quasivoid_tail, regionradius)
                            CalRegion_hf = select_region_pbc_full(glass, x_particle_head, y_particle_head, regionradius)
                            CalRegion_hi = select_region_pbc_full(glass, x_quasivoid_head, y_quasivoid_head, regionradius)

                            mask_ti = ((frame_ti[:, 2] > x_low_ti) & (frame_ti[:, 2] < x_up_ti) &
                                       (frame_ti[:, 3] > y_low_ti) & (frame_ti[:, 3] < y_up_ti))
                            mask_tf = ((frame_tf[:, 2] > x_low_tf) & (frame_tf[:, 2] < x_up_tf) &
                                       (frame_tf[:, 3] > y_low_tf) & (frame_tf[:, 3] < y_up_tf))
                            mask_hf = ((frame_hf[:, 2] > x_low_hf) & (frame_hf[:, 2] < x_up_hf) &
                                       (frame_hf[:, 3] > y_low_hf) & (frame_hf[:, 3] < y_up_hf))
                            mask_hi = ((frame_hi[:, 2] > x_low_hi) & (frame_hi[:, 2] < x_up_hi) &
                                       (frame_hi[:, 3] > y_low_hi) & (frame_hi[:, 3] < y_up_hi))
                            entire_density_initial = Local_Density_Calculator(glass, regionradius, [time_ini])
                            entire_density_final = Local_Density_Calculator(glass, regionradius, [time_final])
                            norm = mcolors.Normalize(vmin=rho-0.1, vmax=rho+0.1)
                            show_heterogeneity(ax0[0, 0], glass, entire_density_initial, norm, time_ini,
                                               [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                            show_heterogeneity(ax0[1, 0], glass, entire_density_initial, norm, time_ini,
                                               [x_low_hi, x_up_hi, y_low_hi, y_up_hi])
                            show_heterogeneity(ax0[0, 1], glass, entire_density_final, norm, time_final,
                                               [x_low_tf, x_up_tf, y_low_tf, y_up_tf])
                            show_heterogeneity(ax0[1, 1], glass, entire_density_final, norm, time_final,
                                               [x_low_hf, x_up_hf, y_low_hf, y_up_hf])
                            set_ax(glass, ax0[0, 0], x_partical_tail, y_particle_tail, regionradius, x_low_ti, x_up_ti,
                                   y_low_ti, y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            set_ax(glass, ax0[0, 1], x_quasivoid_tail, y_quasivoid_tail, regionradius, x_low_tf, x_up_tf,
                                   y_low_tf, y_up_tf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            set_ax(glass, ax0[1, 0], x_quasivoid_head, y_quasivoid_head, regionradius, x_low_hi, x_up_hi,
                                   y_low_hi, y_up_hi, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            set_ax(glass, ax0[1, 1], x_particle_head, y_particle_head, regionradius, x_low_hf, x_up_hf,
                                   y_low_hf, y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            plot_collections(glass, ax0[0, 0], mask_ti, CalRegion_ti, time_ini, time_final, ts, ti, 'initial tail')
                            plot_collections(glass, ax0[0, 1], mask_tf, CalRegion_tf, time_ini, time_final, ts, ti, 'final tail')
                            plot_collections(glass, ax0[1, 0], mask_hi, CalRegion_hi, time_ini, time_final, ts, ti, 'initial head')
                            plot_collections(glass, ax0[1, 1], mask_hf, CalRegion_hf, time_ini, time_final, ts, ti, 'final head')
                            ax4.vlines([time_ini, time_final], 0.75, 0.85, color=['r', 'b'], linestyles='--')
                            ax4.set_xlabel('Time')
                            ax4.set_ylabel('Area density in black circle region')
                            ax4.set_title(f'locale density of string #{ti}')
                            ax4.legend()
                            ax4.grid()
                            if savefile:
                                fig0.savefig(str(savefile) + f'_string{ti:03d}_config.png', dpi=600)
                                plt.close(fig0)
                                fig4.savefig(str(savefile) + f'_string{ti:03d}_tail_and_head_density.png', dpi=600)
                                plt.close(fig4)
                            else:
                                plt.show()

        elif mode == 1:
            sp = [stringparticles[0], stringparticles[-1]]
            time_ini = glass.starend_of_string[ti][0]
            time_final = glass.starend_of_string[ti][1]
            id_tail = int(sp[0])
            id_head = int(sp[1])
            x_partical_tail = glass.frames[time_ini, id_tail, 2]
            y_particle_tail = glass.frames[time_ini, id_tail, 3]
            x_particle_head = glass.frames[time_final, id_head, 2]
            y_particle_head = glass.frames[time_final, id_head, 3]
            dist_headtail_vec = distPBC2D([x_particle_head-x_partical_tail, y_particle_head-y_particle_tail],
                                          glass.L[0], glass.L[1])
            dist_headtail = np.linalg.norm(dist_headtail_vec)
            if dist_headtail > dr_ht:
                if glass.stringlength[ti] >= Length_of_string:
                    sp1 = stringparticles[1:-1]
                    for index_sti, sti in enumerate(glass.starend_of_string[ti]):
                        if crt_in:
                            new_sp = []
                            for particle_in_string in sp1:
                                dregionradius = glass.frames[sti, particle_in_string, 2:4] - glass.frames[sti, stringparticles[0], 2:4]
                                dr2 = glass.frames[sti, particle_in_string, 2:4] - glass.frames[sti, stringparticles[-1], 2:4]
                                dregionradius_pbc = distPBC2D(dregionradius, glass.L[0], glass.L[1])
                                dr2_pbc = distPBC2D(dr2, glass.L[0], glass.L[1])
                                if (np.linalg.norm(dregionradius_pbc) >= regionradius) & (np.linalg.norm(dr2_pbc) >= regionradius):
                                    new_sp.append(particle_in_string)
                            sp = new_sp
                        if choose_middle:
                            n_in_string = len(stringparticles)
                            if n_in_string % 2 == 0 and n_in_string >= 3:
                                sp = [stringparticles[n_in_string // 2], stringparticles[n_in_string // 2+1]]
                            elif n_in_string % 2 == 1 and n_in_string >= 3:
                                sp = [stringparticles[n_in_string // 2 + 1]]
                        for particle_in_string in sp:
                            x_particle_in_string = glass.frames[sti, particle_in_string, 2]
                            y_particle_in_string = glass.frames[sti, particle_in_string, 3]
                            frame_in, x_low, x_up, y_low, y_up = select_region_pbc(
                                glass, x_particle_in_string, y_particle_in_string, regionradius, sti)
                            ld_particle_in_string = cal_density_on_time(glass, frame_in, x_low, x_up, y_low, y_up,
                                                                        x_particle_in_string, y_particle_in_string, regionradius)
                            ld_in_string.append(ld_particle_in_string)
                            if figshow:
                                mask = ((frame_in[:, 2] > x_low) & (frame_in[:, 2] < x_up) &
                                        (frame_in[:, 3] > y_low) & (frame_in[:, 3] < y_up))
                                mask2 = np.isin(frame_in[:, 0] - 1, stringparticles)
                                mask = mask & mask2
                                CalRegion_in = select_region_pbc_full(glass, x_particle_in_string, y_particle_in_string, regionradius)
                                ld_particle_in_string = cal_density_vs_time(glass, CalRegion_in, x_low, x_up, y_low, y_up,
                                                                            x_particle_in_string, y_particle_in_string, regionradius)
                                fig4, ax4 = plt.subplots()
                                fig0, ax0 = plt.subplots()
                                ax4.plot(ld_particle_in_string, c='k', label=f'Local density of particle {particle_in_string} in string at time {sti}', marker='^')
                                entire_density = Local_Density_Calculator(glass, regionradius, [sti])
                                norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                                show_heterogeneity(ax0, glass, entire_density, norm, sti, [x_low, x_up, y_low, y_up])
                                set_ax(glass, ax0, x_particle_in_string, y_particle_in_string, regionradius, x_low, x_up, y_low, y_up)
                                plot_collections(glass, ax0, mask, CalRegion_in, time_ini, time_final, ts, ti, 'In string')
                                ax4.vlines([time_ini, time_final], 0.75, 0.85, color=['r', 'b'], linestyles='--')
                                ax4.set_xlabel('Time')
                                ax4.set_ylabel('Area density in black circle region')
                                ax4.set_title(f'locale density of string #{ti}')
                                ax4.legend()
                                ax4.grid()
                                if savefile:
                                    fig0.savefig(str(savefile) + f'_string{ti:03d}_config_instring.png', dpi=600)
                                    plt.close(fig0)
                                    fig4.savefig(str(savefile) + f'_string{ti:03d}_in_string_density.png', dpi=600)
                                    plt.close(fig4)
                                else:
                                    plt.show()

        elif mode == 2:
            ld_in_string = []
            for sti_index, sti in enumerate(glass.starend_of_string[ti]):
                for particle_in_string in stringparticles:
                    x_particle_in_string = glass.frames[sti, particle_in_string, 2]
                    y_particle_in_string = glass.frames[sti, particle_in_string, 3]
                    frame_in, x_low, x_up, y_low, y_up = select_region_pbc(
                        glass, x_particle_in_string, y_particle_in_string, regionradius, sti)
                    ld_particle_in_string = cal_density_on_time(glass, frame_in, x_low, x_up, y_low, y_up,
                                                                x_particle_in_string, y_particle_in_string, regionradius)
                    ld_in_string.append(ld_particle_in_string)
            min_index = np.argmin(ld_in_string)
            plist = stringparticles * 2
            pid = plist[min_index]
            print('min_index:', np.mod(min_index, len(stringparticles)))
            print(np.shape(ld_in_string))
            ld_list.append(np.min(ld_in_string))

    if mode == 0:
        return density_tail_ini, density_tail_final, density_head_ini, density_head_final, time_select
    elif mode == 1:
        return ld_in_string
    elif mode == 2:
        return ld_list

def showdensity_no_overlap(glass, regionradius, r_insert, rho, savefile=None, figshow=False,
                pid=None, select=False, mode=0, Length_of_string=3, crt_in=False, choose_middle=False, dr_ht=5):

    density_tail_ini = []
    density_head_ini = []
    density_tail_final = []
    density_head_final = []
    ld_in_string = []
    time_select = []
    ld_list = []
    ts, _, _ = np.shape(glass.frames)
    pick_list_tail = [[] for _ in range(ts)]
    pick_list_head = [[] for _ in range(ts)]

    for ti, stringparticles in enumerate(tqdm(glass.connected_components, desc="Processing Components")):
        if select:
            if pid != stringparticles[0]:
                continue
        if mode == 0:
            if figshow:
                fig4, ax4 = plt.subplots()
                fig0, ax0 = plt.subplots(2, 2, figsize=(8, 8))

            sp = [stringparticles[0], stringparticles[-1]]
            time_ini = glass.starend_of_string[ti][0]
            time_final = glass.starend_of_string[ti][1]

            dr_tail_1 = dr_overlap(glass, time_ini, sp[0], pick_list_tail[time_ini])
            dr_tail_2 = dr_overlap(glass, time_final, sp[1], pick_list_head[time_final])

            if (dr_tail_1 > 2*regionradius and dr_tail_2 > 2*regionradius) or \
                    (not pick_list_tail[time_ini] and not pick_list_head[time_final]):
                id_tail = sp[0]
                id_head = sp[1]
                pick_list_tail[time_ini].append(id_tail)
                pick_list_head[time_final].append(id_head)

                x_partical_tail = glass.frames[time_ini, id_tail, 2]
                y_particle_tail = glass.frames[time_ini, id_tail, 3]
                x_quasivoid_tail = glass.frames[time_final, id_tail, 2]
                y_quasivoid_tail = glass.frames[time_final, id_tail, 3]
                x_quasivoid_tail = particle_unwrap(x_partical_tail, x_quasivoid_tail, glass.L[0])
                y_quasivoid_tail = particle_unwrap(y_particle_tail, y_quasivoid_tail, glass.L[1])

                x_particle_head = glass.frames[time_final, id_head, 2]
                y_particle_head = glass.frames[time_final, id_head, 3]
                x_quasivoid_head = glass.frames[time_ini, id_head, 2]
                y_quasivoid_head = glass.frames[time_ini, id_head, 3]
                x_quasivoid_head = particle_unwrap(x_particle_head, x_quasivoid_head, glass.L[0])
                y_quasivoid_head = particle_unwrap(y_particle_head, y_quasivoid_head, glass.L[1])

                dist_headtail_vec = distPBC2D([x_particle_head-x_partical_tail, y_particle_head-y_particle_tail],
                                              glass.L[0], glass.L[1])
                dist_headtail = np.linalg.norm(dist_headtail_vec)

                if dist_headtail > dr_ht:
                    # 单帧版本用于密度计算
                    frame_ti, x_low_ti, x_up_ti, y_low_ti, y_up_ti = select_region_pbc(
                        glass, x_partical_tail, y_particle_tail, regionradius, time_ini)
                    frame_tf, x_low_tf, x_up_tf, y_low_tf, y_up_tf = select_region_pbc(
                        glass, x_quasivoid_tail, y_quasivoid_tail, regionradius, time_final)
                    frame_hf, x_low_hf, x_up_hf, y_low_hf, y_up_hf = select_region_pbc(
                        glass, x_particle_head, y_particle_head, regionradius, time_final)
                    frame_hi, x_low_hi, x_up_hi, y_low_hi, y_up_hi = select_region_pbc(
                        glass, x_quasivoid_head, y_quasivoid_head, regionradius, time_ini)

                    if r_insert == 0:
                        ld_ti = cal_density_on_time(glass, frame_ti, x_low_ti, x_up_ti, y_low_ti, y_up_ti,
                                                    x_partical_tail, y_particle_tail, regionradius)
                        ld_tf = cal_density_on_time(glass, frame_tf, x_low_tf, x_up_tf, y_low_tf, y_up_tf,
                                                    x_quasivoid_tail, y_quasivoid_tail, regionradius)
                        ld_hf = cal_density_on_time(glass, frame_hf, x_low_hf, x_up_hf, y_low_hf, y_up_hf,
                                                    x_particle_head, y_particle_head, regionradius)
                        ld_hi = cal_density_on_time(glass, frame_hi, x_low_hi, x_up_hi, y_low_hi, y_up_hi,
                                                    x_quasivoid_head, y_quasivoid_head, regionradius)

                        density_tail_ini.append(ld_ti)
                        density_tail_final.append(ld_tf)
                        density_head_final.append(ld_hf)
                        density_head_ini.append(ld_hi)
                        time_select = [time_ini, time_final]

                        if figshow:
                            # 完整版本仅用于 plot_collections
                            CalRegion_ti = select_region_pbc_full(glass, x_partical_tail, y_particle_tail, regionradius)
                            CalRegion_tf = select_region_pbc_full(glass, x_quasivoid_tail, y_quasivoid_tail, regionradius)
                            CalRegion_hf = select_region_pbc_full(glass, x_particle_head, y_particle_head, regionradius)
                            CalRegion_hi = select_region_pbc_full(glass, x_quasivoid_head, y_quasivoid_head, regionradius)

                            mask_ti = ((frame_ti[:, 2] > x_low_ti) & (frame_ti[:, 2] < x_up_ti) &
                                       (frame_ti[:, 3] > y_low_ti) & (frame_ti[:, 3] < y_up_ti))
                            mask_tf = ((frame_tf[:, 2] > x_low_tf) & (frame_tf[:, 2] < x_up_tf) &
                                       (frame_tf[:, 3] > y_low_tf) & (frame_tf[:, 3] < y_up_tf))
                            mask_hf = ((frame_hf[:, 2] > x_low_hf) & (frame_hf[:, 2] < x_up_hf) &
                                       (frame_hf[:, 3] > y_low_hf) & (frame_hf[:, 3] < y_up_hf))
                            mask_hi = ((frame_hi[:, 2] > x_low_hi) & (frame_hi[:, 2] < x_up_hi) &
                                       (frame_hi[:, 3] > y_low_hi) & (frame_hi[:, 3] < y_up_hi))

                            entire_density_initial = Local_Density_Calculator(glass, regionradius, [time_ini])
                            entire_density_final = Local_Density_Calculator(glass, regionradius, [time_final])
                            norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                            show_heterogeneity(ax0[0, 0], glass, entire_density_initial, norm, time_ini,
                                               [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                            show_heterogeneity(ax0[1, 0], glass, entire_density_initial, norm, time_ini,
                                               [x_low_hi, x_up_hi, y_low_hi, y_up_hi])
                            show_heterogeneity(ax0[0, 1], glass, entire_density_final, norm, time_final,
                                               [x_low_tf, x_up_tf, y_low_tf, y_up_tf])
                            show_heterogeneity(ax0[1, 1], glass, entire_density_final, norm, time_final,
                                               [x_low_hf, x_up_hf, y_low_hf, y_up_hf])
                            set_ax(ax0[0, 0], x_partical_tail, y_particle_tail, regionradius, x_low_ti, x_up_ti,
                                         y_low_ti, y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            set_ax(ax0[0, 1], x_quasivoid_tail, y_quasivoid_tail, regionradius, x_low_tf, x_up_tf,
                                         y_low_tf, y_up_tf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            set_ax(ax0[1, 0], x_quasivoid_head, y_quasivoid_head, regionradius, x_low_hi, x_up_hi,
                                         y_low_hi, y_up_hi, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            set_ax(ax0[1, 1], x_particle_head, y_particle_head, regionradius, x_low_hf, x_up_hf,
                                         y_low_hf, y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                            plot_collections(ax0[0, 0], mask_ti, CalRegion_ti, time_ini, time_final, ts, ti, 'initial tail')
                            plot_collections(ax0[0, 1], mask_tf, CalRegion_tf, time_ini, time_final, ts, ti, 'final tail')
                            plot_collections(ax0[1, 0], mask_hi, CalRegion_hi, time_ini, time_final, ts, ti, 'initial head')
                            plot_collections(ax0[1, 1], mask_hf, CalRegion_hf, time_ini, time_final, ts, ti, 'final head')
                            ax4.vlines([time_ini, time_final], 0.75, 0.85, color=['r', 'b'], linestyles='--')
                            ax4.set_xlabel('Time')
                            ax4.set_ylabel('Area density in black circle region')
                            ax4.set_title(f'locale density of string #{ti}')
                            ax4.legend()
                            ax4.grid()
                            if savefile:
                                fig0.savefig(str(savefile) + f'_string{ti:03d}_config.png', dpi=600)
                                plt.close(fig0)
                                fig4.savefig(str(savefile) + f'_string{ti:03d}_tail_and_head_density.png', dpi=600)
                                plt.close(fig4)
                            else:
                                plt.show()

                    else:
                        ld_ti, mindr_ti = cal_density_on_time_with_ri(glass, frame_ti, x_low_ti, x_up_ti,
                                                                       y_low_ti, y_up_ti, x_partical_tail,
                                                                       y_particle_tail, regionradius)
                        ld_tf, mindr_tf = cal_density_on_time_with_ri(glass, frame_tf, x_low_tf, x_up_tf,
                                                                       y_low_tf, y_up_tf, x_quasivoid_tail,
                                                                       y_quasivoid_tail, regionradius)
                        ld_hf, mindr_hf = cal_density_on_time_with_ri(glass, frame_hf, x_low_hf, x_up_hf,
                                                                       y_low_hf, y_up_hf, x_particle_head,
                                                                       y_particle_head, regionradius)
                        ld_hi, mindr_hi = cal_density_on_time_with_ri(glass, frame_hi, x_low_hi, x_up_hi,
                                                                       y_low_hi, y_up_hi, x_quasivoid_head,
                                                                       y_quasivoid_head, regionradius)

                        if mindr_tf > r_insert and mindr_hi > r_insert:
                            density_tail_ini.append(ld_ti)
                            density_tail_final.append(ld_tf)
                            density_head_final.append(ld_hf)
                            density_head_ini.append(ld_hi)
                            time_select = [time_ini, time_final]
                            if figshow:
                                # 完整版本仅用于 plot_collections
                                CalRegion_ti = select_region_pbc_full(glass, x_partical_tail, y_particle_tail, regionradius)
                                CalRegion_tf = select_region_pbc_full(glass, x_quasivoid_tail, y_quasivoid_tail, regionradius)
                                CalRegion_hf = select_region_pbc_full(glass, x_particle_head, y_particle_head, regionradius)
                                CalRegion_hi = select_region_pbc_full(glass, x_quasivoid_head, y_quasivoid_head, regionradius)

                                mask_ti = ((frame_ti[:, 2] > x_low_ti) & (frame_ti[:, 2] < x_up_ti) &
                                           (frame_ti[:, 3] > y_low_ti) & (frame_ti[:, 3] < y_up_ti))
                                mask_tf = ((frame_tf[:, 2] > x_low_tf) & (frame_tf[:, 2] < x_up_tf) &
                                           (frame_tf[:, 3] > y_low_tf) & (frame_tf[:, 3] < y_up_tf))
                                mask_hf = ((frame_hf[:, 2] > x_low_hf) & (frame_hf[:, 2] < x_up_hf) &
                                           (frame_hf[:, 3] > y_low_hf) & (frame_hf[:, 3] < y_up_hf))
                                mask_hi = ((frame_hi[:, 2] > x_low_hi) & (frame_hi[:, 2] < x_up_hi) &
                                           (frame_hi[:, 3] > y_low_hi) & (frame_hi[:, 3] < y_up_hi))
                                entire_density_initial = Local_Density_Calculator(glass, regionradius, [time_ini])
                                entire_density_final = Local_Density_Calculator(glass, regionradius, [time_final])
                                norm = mcolors.Normalize(vmin=rho-0.1, vmax=rho+0.1)
                                show_heterogeneity(ax0[0, 0], glass, entire_density_initial, norm, time_ini,
                                                   [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                                show_heterogeneity(ax0[1, 0], glass, entire_density_initial, norm, time_ini,
                                                   [x_low_hi, x_up_hi, y_low_hi, y_up_hi])
                                show_heterogeneity(ax0[0, 1], glass, entire_density_final, norm, time_final,
                                                   [x_low_tf, x_up_tf, y_low_tf, y_up_tf])
                                show_heterogeneity(ax0[1, 1], glass, entire_density_final, norm, time_final,
                                                   [x_low_hf, x_up_hf, y_low_hf, y_up_hf])
                                set_ax(ax0[0, 0], x_partical_tail, y_particle_tail, regionradius, x_low_ti, x_up_ti,
                                             y_low_ti, y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                                set_ax(ax0[0, 1], x_quasivoid_tail, y_quasivoid_tail, regionradius, x_low_tf, x_up_tf,
                                             y_low_tf, y_up_tf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                                set_ax(ax0[1, 0], x_quasivoid_head, y_quasivoid_head, regionradius, x_low_hi, x_up_hi,
                                             y_low_hi, y_up_hi, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                                set_ax(ax0[1, 1], x_particle_head, y_particle_head, regionradius, x_low_hf, x_up_hf,
                                             y_low_hf, y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head, y_quasivoid_head)
                                plot_collections(ax0[0, 0], mask_ti, CalRegion_ti, time_ini, time_final, ts, ti, 'initial tail')
                                plot_collections(ax0[0, 1], mask_tf, CalRegion_tf, time_ini, time_final, ts, ti, 'final tail')
                                plot_collections(ax0[1, 0], mask_hi, CalRegion_hi, time_ini, time_final, ts, ti, 'initial head')
                                plot_collections(ax0[1, 1], mask_hf, CalRegion_hf, time_ini, time_final, ts, ti, 'final head')
                                ax4.vlines([time_ini, time_final], 0.75, 0.85, color=['r', 'b'], linestyles='--')
                                ax4.set_xlabel('Time')
                                ax4.set_ylabel('Area density in black circle region')
                                ax4.set_title(f'locale density of string #{ti}')
                                ax4.legend()
                                ax4.grid()
                                if savefile:
                                    fig0.savefig(str(savefile) + f'_string{ti:03d}_config.png', dpi=600)
                                    plt.close(fig0)
                                    fig4.savefig(str(savefile) + f'_string{ti:03d}_tail_and_head_density.png', dpi=600)
                                    plt.close(fig4)
                                else:
                                    plt.show()

        elif mode == 1:
            sp = [stringparticles[0], stringparticles[-1]]
            time_ini = glass.starend_of_string[ti][0]
            time_final = glass.starend_of_string[ti][1]
            id_tail = int(sp[0])
            id_head = int(sp[1])
            x_partical_tail = glass.frames[time_ini, id_tail, 2]
            y_particle_tail = glass.frames[time_ini, id_tail, 3]
            x_particle_head = glass.frames[time_final, id_head, 2]
            y_particle_head = glass.frames[time_final, id_head, 3]
            dist_headtail_vec = distPBC2D([x_particle_head-x_partical_tail, y_particle_head-y_particle_tail],
                                          glass.L[0], glass.L[1])
            dist_headtail = np.linalg.norm(dist_headtail_vec)
            if dist_headtail > dr_ht:
                if glass.stringlength[ti] >= Length_of_string:
                    sp1 = stringparticles[1:-1]
                    for index_sti, sti in enumerate(glass.starend_of_string[ti]):
                        if crt_in:
                            new_sp = []
                            for particle_in_string in sp1:
                                dregionradius = glass.frames[sti, particle_in_string, 2:4] - glass.frames[sti, stringparticles[0], 2:4]
                                dr2 = glass.frames[sti, particle_in_string, 2:4] - glass.frames[sti, stringparticles[-1], 2:4]
                                dregionradius_pbc = distPBC2D(dregionradius, glass.L[0], glass.L[1])
                                dr2_pbc = distPBC2D(dr2, glass.L[0], glass.L[1])
                                if (np.linalg.norm(dregionradius_pbc) >= regionradius) & (np.linalg.norm(dr2_pbc) >= regionradius):
                                    new_sp.append(particle_in_string)
                            sp = new_sp
                        if choose_middle:
                            n_in_string = len(stringparticles)
                            if n_in_string % 2 == 0 and n_in_string >= 3:
                                sp = [stringparticles[n_in_string // 2], stringparticles[n_in_string // 2+1]]
                            elif n_in_string % 2 == 1 and n_in_string >= 3:
                                sp = [stringparticles[n_in_string // 2 + 1]]
                        for particle_in_string in sp:
                            x_particle_in_string = glass.frames[sti, particle_in_string, 2]
                            y_particle_in_string = glass.frames[sti, particle_in_string, 3]
                            frame_in, x_low, x_up, y_low, y_up = select_region_pbc(
                                glass, x_particle_in_string, y_particle_in_string, regionradius, sti)
                            ld_particle_in_string = cal_density_on_time(glass, frame_in, x_low, x_up, y_low, y_up,
                                                                        x_particle_in_string, y_particle_in_string, regionradius)
                            ld_in_string.append(ld_particle_in_string)
                            if figshow:
                                mask = ((frame_in[:, 2] > x_low) & (frame_in[:, 2] < x_up) &
                                        (frame_in[:, 3] > y_low) & (frame_in[:, 3] < y_up))
                                mask2 = np.isin(frame_in[:, 0] - 1, stringparticles)
                                mask = mask & mask2
                                # 完整版本仅用于 plot_collections
                                CalRegion_in = select_region_pbc_full(glass, x_particle_in_string, y_particle_in_string, regionradius)
                                fig4, ax4 = plt.subplots()
                                fig0, ax0 = plt.subplots()
                                entire_density = Local_Density_Calculator(glass, regionradius, [sti])
                                norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                                show_heterogeneity(ax0, glass, entire_density, norm, sti, [x_low, x_up, y_low, y_up])
                                set_ax(ax0, x_particle_in_string, y_particle_in_string, regionradius, x_low, x_up, y_low, y_up)
                                plot_collections(ax0, mask, CalRegion_in, time_ini, time_final, ts, ti, 'In string')
                                ax4.vlines([time_ini, time_final], 0.75, 0.85, color=['r', 'b'], linestyles='--')
                                ax4.set_xlabel('Time')
                                ax4.set_ylabel('Area density in black circle region')
                                ax4.set_title(f'locale density of string #{ti}')
                                ax4.legend()
                                ax4.grid()
                                if savefile:
                                    fig0.savefig(str(savefile) + f'_string{ti:03d}_config_instring.png', dpi=600)
                                    plt.close(fig0)
                                    fig4.savefig(str(savefile) + f'_string{ti:03d}_in_string_density.png', dpi=600)
                                    plt.close(fig4)
                                else:
                                    plt.show()

        elif mode == 2:
            ld_in_string = []
            for sti_index, sti in enumerate(glass.starend_of_string[ti]):
                for particle_in_string in stringparticles:
                    x_particle_in_string = glass.frames[sti, particle_in_string, 2]
                    y_particle_in_string = glass.frames[sti, particle_in_string, 3]
                    frame_in, x_low, x_up, y_low, y_up = select_region_pbc(
                        glass, x_particle_in_string, y_particle_in_string, regionradius, sti)
                    ld_particle_in_string = cal_density_on_time(glass, frame_in, x_low, x_up, y_low, y_up,
                                                                x_particle_in_string, y_particle_in_string, regionradius)
                    ld_in_string.append(ld_particle_in_string)
            min_index = np.argmin(ld_in_string)
            plist = stringparticles * 2
            pid = plist[min_index]
            print('min_index:', np.mod(min_index, len(stringparticles)))
            print(np.shape(ld_in_string))
            ld_list.append(np.min(ld_in_string))

    if mode == 0:
        return density_tail_ini, density_tail_final, density_head_ini, density_head_final, time_select
    elif mode == 1:
        return ld_in_string
    elif mode == 2:
        return ld_list



def showdensity_no_overlap_tail_only(glass, regionradius, r_insert, rho, savefile=None, figshow=False,
                pid=None, select=False, dr_ht=5):

    density_tail_ini = []
    density_tail_final = []
    time_select = []
    ts, _, _ = np.shape(glass.frames)
    Deduplicate(glass)
    deepDeduplicate(glass)
    
    pick_list_tail = [[] for _ in range(ts)]  # 只需要 tail 的去重列表

    for ti, stringparticles in enumerate(tqdm(glass.connected_components, desc="Processing Components")):
        if select:
            if pid != stringparticles[0]:
                continue

        sp = [stringparticles[0], stringparticles[-1]]
        time_ini = glass.starend_of_string[ti][0]
        time_final = glass.starend_of_string[ti][1]

        # 只检查 tail 是否与已有 tail 重叠
        dr_tail = dr_overlap(glass, time_ini, sp[0], pick_list_tail[time_ini])
        if not (dr_tail > 2 * regionradius or not pick_list_tail[time_ini]):
            continue

        id_tail = sp[0]
        pick_list_tail[time_ini].append(id_tail)

        x_tail = glass.frames[time_ini, id_tail, 2]
        y_tail = glass.frames[time_ini, id_tail, 3]
        x_quasivoid_tail = glass.frames[time_final, id_tail, 2]
        y_quasivoid_tail = glass.frames[time_final, id_tail, 3]
        x_quasivoid_tail = particle_unwrap(x_tail, x_quasivoid_tail, glass.L[0])
        y_quasivoid_tail = particle_unwrap(y_tail, y_quasivoid_tail, glass.L[1])

        # 只检查 head-tail 距离（保留这个过滤条件的话）
        x_head = glass.frames[time_final, sp[1], 2]
        y_head = glass.frames[time_final, sp[1], 3]
        dist_headtail = np.linalg.norm(
            distPBC2D([x_head - x_tail, y_head - y_tail], glass.L[0], glass.L[1])
        )
        if dist_headtail <= dr_ht:
            continue

        # 只计算 tail 相关区域
        CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti, y_up_ti = \
            select_region_pbc(glass, x_tail, y_tail, regionradius)
        CalRegion_tail_final, x_low_tf, x_up_tf, y_low_tf, y_up_tf = \
            select_region_pbc(glass, x_quasivoid_tail, y_quasivoid_tail, regionradius)

        if r_insert == 0:
            ld_ti = cal_density_on_time(glass, CalRegion_tail_ini, x_low_ti, x_up_ti,
                                        y_low_ti, y_up_ti, x_tail, y_tail, regionradius, time_ini)
            ld_tf = cal_density_on_time(glass, CalRegion_tail_final, x_low_tf, x_up_tf,
                                        y_low_tf, y_up_tf, x_quasivoid_tail, y_quasivoid_tail, regionradius, time_final)
            density_tail_ini.append(ld_ti)
            density_tail_final.append(ld_tf)
            time_select = [time_ini, time_final]
        else:
            ld_ti, mindr_ti = cal_density_vs_time_with_ri(glass, CalRegion_tail_ini, x_low_ti, x_up_ti,
                                                          y_low_ti, y_up_ti, x_tail, y_tail, regionradius)
            ld_tf, mindr_tf = cal_density_vs_time_with_ri(glass, CalRegion_tail_final, x_low_tf, x_up_tf,
                                                          y_low_tf, y_up_tf, x_quasivoid_tail, y_quasivoid_tail, regionradius)
            if mindr_tf[time_ini] > r_insert:  # 只检查 tail 侧的 r_insert 条件
                density_tail_ini.append(ld_ti[time_ini])
                density_tail_final.append(ld_tf[time_final])
                time_select = [time_ini, time_final]

    return density_tail_ini, density_tail_final, time_select

def showdensity_simu_diff(glass, regionradius, dr_ht):
    # mode = 0  calculate the local density at string ends
    # mode = 1 calculate the local density of the middle/other (without ends) particles in the string
    # mode = 2 calculate minial local density of the loop
    # crt_in = False  whether set Rc as criteria for the internal particles in the string
    ts, N, _ = np.shape(glass.frames)
    Deduplicate(glass)
    deepDeduplicate(glass)

    # initialize the parameters
    ld_str = [[] for _ in range(ts)]
    ld_all = [[] for _ in range(ts)]
    tail_all = [[] for _ in range(ts)]

    t_all = [s[0] for s in glass.connected_components]
    t_t = [t[0] for t in glass.starend_of_string]
    h_all = [s[-1] for s in glass.connected_components]
    h_t = [t[1] for t in glass.starend_of_string]
    t_all.extend(h_all)  # all tail particle id
    t_t.extend(h_t)      # all tail location time
    #reconstruct with time
    for i, t in enumerate(t_t):
        tail_all[t].append(t_all[i])

    new_t = set(t_t)

    pick_list_tail = [[] for _ in range(ts)]
    pick_list_head = [[] for _ in range(ts)]
    for ti, stringparticles in enumerate(tqdm(glass.connected_components, desc="Processing Components")):

        sp = [stringparticles[0], stringparticles[-1]]
        time_ini = glass.starend_of_string[ti][0]
        time_final = glass.starend_of_string[ti][1]

        ##exclude overlap region
        dr_tail_1 = dr_overlap(glass, time_ini, sp[0], pick_list_tail[time_ini])
        dr_tail_2 = dr_overlap(glass, time_final, sp[1], pick_list_head[time_final])
        # get partile and quasivoid position at the tail
        if (dr_tail_1 > 2*regionradius and dr_tail_2 > 2*regionradius) or \
                (not pick_list_tail[time_ini] and not pick_list_head[time_final]):
            id_tail = sp[0]
            id_head = sp[1]
            pick_list_tail[time_ini].append(id_tail)
            pick_list_head[time_final].append(id_head)
            x_partical_tail = glass.frames[time_ini, id_tail, 2]
            y_particle_tail = glass.frames[time_ini, id_tail, 3]


            # get partile and quasivoid position at the head
            x_particle_head = glass.frames[time_final, id_head, 2]
            y_particle_head = glass.frames[time_final, id_head, 3]

            dist_headtail_vec = distPBC2D([x_particle_head-x_partical_tail, y_particle_head-y_particle_tail],
                                          glass.L[0], glass.L[1])
            dist_headtail = np.linalg.norm(dist_headtail_vec)
            if dist_headtail > dr_ht:
                CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti, y_up_ti = select_region_pbc(glass, x_partical_tail, y_particle_tail, regionradius)
                CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf, y_up_hf = select_region_pbc(glass, x_particle_head, y_particle_head, regionradius)

                ld_ti = cal_density_on_time(glass, CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti, y_up_ti,
                                            x_partical_tail, y_particle_tail, regionradius, time_ini)
                ld_hf = cal_density_on_time(glass, CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf, y_up_hf,
                                            x_particle_head, y_particle_head, regionradius, time_final)

                ld_str[time_ini].append(ld_ti)
                ld_str[time_final].append(ld_hf)
            else:
                print(f'{time_ini} and {time_final}' )
    new_ld_str = [np.mean(x) for x in ld_str if x]
        # calculate the simultaneous total system ld
    time_list = []
    for i, ldi in enumerate(ld_str):
        if ldi:
            time_list.append(i)
    for iT in tqdm(time_list, desc="Process all particles in frames:"):
        for iN in range(N):
            dr_min = dr_overlap(glass, iT, iN, tail_all[iT])
            if dr_min > 2*regionradius or dr_min == 0:  # no overlap
                x1 = glass.frames[iT, iN, 2]
                y1 = glass.frames[iT, iN, 3]
                skin = max(glass.radii)
                xlow = np.mod(x1 - regionradius - skin, glass.L[0])
                xup = np.mod(x1 + regionradius + skin, glass.L[0])
                ylow = np.mod(y1 - regionradius - skin, glass.L[1])
                yup = np.mod(y1 + regionradius + skin, glass.L[1])
                if xlow>xup:
                    xregion = (glass.frames[iT, :, 2] > xlow) | (glass.frames[iT, :, 2] < xup)
                else:
                    xregion = (glass.frames[iT, :, 2] > xlow) & (glass.frames[iT, :, 2] < xup)
                if ylow>yup:
                    yregion = (glass.frames[iT, :, 3] > ylow) | (glass.frames[iT, :, 3] < yup)
                else:
                    yregion = (glass.frames[iT, :, 3] > ylow) & (glass.frames[iT, :, 3] < yup)
                mask = xregion & yregion
                slctp = glass.frames[iT, mask, :]
                s = 0
                for sj in slctp:
                    x2 = sj[2]
                    y2 = sj[3]
                    r2 = glass.radii[int(sj[0]-1)]
                    s += Local_Density_Calculator.circle_intersection_area(x1, y1, regionradius, x2, y2, r2, glass.L[0], glass.L[1])
                if s == 0:
                    print('s=0， error: 2(R+skin)>L ')
                ld_all[iT].append(s / (math.pi * regionradius ** 2))
    new_ld_all = [np.mean(x) for x in ld_all if x]

    return np.array(new_ld_str), np.array(new_ld_all)

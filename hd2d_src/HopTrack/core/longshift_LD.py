import matplotlib.pyplot as plt
import numpy as np
from hd2d_src.HopTrack.core.share import *
from hd2d_src.HopTrack.core import Local_Density_Calculator as LDC
import matplotlib.colors as mcolors


def calcdensity(glass, regionradius, r_insert, rho, savefile=None, figshow=False,
                    pid=None, select=False, mode=0, Length_of_string=3, crt_in=False, choose_middle=False, dr_ht=5):
        # mode = 0  calculate the local density at string ends
        # mode = 1 calculate the local density of the particles in the string （without ends）
        # crt_in = False  whether set Rc as criteria for the internal particles in the string

        # get particle id in string
        density_tail_ini = []
        density_head_ini = []
        density_tail_final = []
        density_head_final = []
        ld_in_string = []
        time_select = []
        ts, _, _ = np.shape(glass.frames)
        for ti, stringparticles in enumerate(glass.connected_components):
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
                # get partile and quasivoid position at the tail
                id_tail = int(sp[0])
                id_head = int(sp[1])
                x_partical_tail = glass.frames[time_ini, id_tail, 2]
                y_particle_tail = glass.frames[time_ini, id_tail, 3]
                x_quasivoid_tail = glass.frames[time_final, id_tail, 2]
                y_quasivoid_tail = glass.frames[time_final, id_tail, 3]
                x_quasivoid_tail = particle_unwrap(x_partical_tail, x_quasivoid_tail,
                                                   glass.L[0])  # unwrap when particle hops to the otherside
                y_quasivoid_tail = particle_unwrap(y_particle_tail, y_quasivoid_tail, glass.L[1])

                # get partile and quasivoid position at the head
                x_particle_head = glass.frames[time_final, id_head, 2]
                y_particle_head = glass.frames[time_final, id_head, 3]
                x_quasivoid_head = glass.frames[time_ini, id_head, 2]
                y_quasivoid_head = glass.frames[time_ini, id_head, 3]
                x_quasivoid_head = particle_unwrap(x_particle_head, x_quasivoid_head, glass.L[0])
                y_quasivoid_head = particle_unwrap(y_particle_head, y_quasivoid_head, glass.L[1])

                dist_headtail_vec = distPBC2D([x_particle_head - x_partical_tail, y_particle_head - y_particle_tail],
                                              glass.L[0], glass.L[1])
                dist_headtail = np.linalg.norm(dist_headtail_vec)
                if dist_headtail > dr_ht:
                    CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti, y_up_ti = select_region_pbc(glass, x_partical_tail,
                                                                                                 y_particle_tail,
                                                                                                 regionradius)
                    CalRegion_tail_final, x_low_tf, x_up_tf, y_low_tf, y_up_tf = select_region_pbc(glass,
                                                                                                   x_quasivoid_tail,
                                                                                                   y_quasivoid_tail,
                                                                                                   regionradius)
                    CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf, y_up_hf = select_region_pbc(glass,
                                                                                                   x_particle_head,
                                                                                                   y_particle_head,
                                                                                                   regionradius)
                    CalRegion_head_ini, x_low_hi, x_up_hi, y_low_hi, y_up_hi = select_region_pbc(glass,
                                                                                                 x_quasivoid_head,
                                                                                                 y_quasivoid_head,
                                                                                                 regionradius)

                    ld_ti, mindr_ti = cal_density_vs_time(glass, CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti,
                                                          y_up_ti,
                                                          x_partical_tail, y_particle_tail, regionradius)
                    ld_tf, mindr_tf = cal_density_vs_time(glass, CalRegion_tail_final, x_low_tf, x_up_tf, y_low_tf,
                                                          y_up_tf, x_quasivoid_tail, y_quasivoid_tail, regionradius)
                    ld_hf, mindr_hf = cal_density_vs_time(glass, CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf,
                                                          y_up_hf,
                                                          x_particle_head, y_particle_head, regionradius)
                    ld_hi, mindr_hi = cal_density_vs_time(glass, CalRegion_head_ini, x_low_hi, x_up_hi, y_low_hi,
                                                          y_up_hi,
                                                          x_quasivoid_head, y_quasivoid_head, regionradius)

                    # plot local density vs time
                    if figshow:
                        ax4.plot(ld_ti, c='r', label='Initial tail density', marker='o')
                        ax4.plot(ld_tf, linestyle='--', c='b', label='Final tail density', marker='x')
                        ax4.plot(ld_hi, c='r', label='Initial head density', marker='s')
                        ax4.plot(ld_hf, linestyle='--', c='b', label='Final head density', marker='*')

                    if mindr_tf[time_ini] > r_insert and mindr_hi[time_final] > r_insert:
                        t = time_ini
                        mask_ti = ((CalRegion_tail_ini[t, :, 2] > x_low_ti) & (CalRegion_tail_ini[t, :, 2] < x_up_ti) &
                                   (CalRegion_tail_ini[t, :, 3] > y_low_ti) & (CalRegion_tail_ini[t, :,
                                                                               3] < y_up_ti))  # select particles in the square region to speed up
                        mask_tf = ((CalRegion_tail_final[t, :, 2] > x_low_tf) & (
                                    CalRegion_tail_final[t, :, 2] < x_up_tf)
                                   & (CalRegion_tail_final[t, :, 3] > y_low_tf) & (
                                               CalRegion_tail_final[t, :, 3] < y_up_tf))
                        mask_hf = ((CalRegion_head_final[t, :, 2] > x_low_hf) & (
                                    CalRegion_head_final[t, :, 2] < x_up_hf) &
                                   (CalRegion_head_final[t, :, 3] > y_low_hf) & (
                                               CalRegion_head_final[t, :, 3] < y_up_hf))
                        mask_hi = ((CalRegion_head_ini[t, :, 2] > x_low_hi) & (CalRegion_head_ini[t, :, 2] < x_up_hi) &
                                   (CalRegion_head_ini[t, :, 3] > y_low_hi) & (CalRegion_head_ini[t, :, 3] < y_up_hi))

                        density_tail_ini.append(ld_ti[time_ini])
                        density_tail_final.append(ld_tf[time_final])
                        density_head_final.append(ld_hf[time_final])
                        density_head_ini.append(ld_hi[time_ini])
                        time_select = [time_ini, time_final]
                        if figshow:
                            entire_density_initial = LDC.Local_Density_Calculator(glass, regionradius, [time_ini])
                            entire_density_final = LDC.Local_Density_Calculator(glass, regionradius, [time_final])
                            norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                            LDC.show_heterogeneity(ax0[0, 0], glass, entire_density_initial, norm, time_ini,
                                                   [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                            LDC.show_heterogeneity(ax0[1, 0], glass, entire_density_initial, norm, time_ini,
                                                   [x_low_hi, x_up_hi, y_low_hi, y_up_hi])
                            LDC.show_heterogeneity(ax0[0, 1], glass, entire_density_final, norm, time_final,
                                                   [x_low_tf, x_up_tf, y_low_tf, y_up_tf])
                            LDC.show_heterogeneity(ax0[1, 1], glass, entire_density_final, norm, time_final,
                                                   [x_low_hf, x_up_hf, y_low_hf, y_up_hf])

                            # ax, x1, y1, r1, x_low, x_up, y_low, y_up, x_qt, y_qt, x_qh, y_qh
                            glass.set_ax(ax0[0, 0], x_partical_tail, y_particle_tail, regionradius, x_low_ti, x_up_ti,
                                         y_low_ti,
                                         y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                         y_quasivoid_head)
                            glass.set_ax(ax0[0, 1], x_quasivoid_tail, y_quasivoid_tail, regionradius, x_low_tf, x_up_tf,
                                         y_low_tf,
                                         y_up_tf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                         y_quasivoid_head)
                            glass.set_ax(ax0[1, 0], x_quasivoid_head, y_quasivoid_head, regionradius, x_low_hi, x_up_hi,
                                         y_low_hi,
                                         y_up_hi, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                         y_quasivoid_head)
                            glass.set_ax(ax0[1, 1], x_particle_head, y_particle_head, regionradius, x_low_hf, x_up_hf,
                                         y_low_hf,
                                         y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                         y_quasivoid_head)

                            # plot string
                            glass.plot_collections(ax0[0, 0], mask_ti, CalRegion_tail_ini, time_ini, time_final, ts, ti,
                                                   'initial tail')
                            glass.plot_collections(ax0[0, 1], mask_tf, CalRegion_tail_final, time_ini, time_final, ts,
                                                   ti,
                                                   'final tail')
                            glass.plot_collections(ax0[1, 0], mask_hi, CalRegion_head_ini, time_ini, time_final, ts, ti,
                                                   'initial head')
                            glass.plot_collections(ax0[1, 1], mask_hf, CalRegion_head_final, time_ini, time_final, ts,
                                                   ti,
                                                   'final head')
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
                # print(f'mode={mode}, calculate the local density of partilces in string')
                sp = [stringparticles[0], stringparticles[-1]]
                time_ini = glass.starend_of_string[ti][0]
                time_final = glass.starend_of_string[ti][1]
                # get partile and quasivoid position at the tail
                id_tail = int(sp[0])
                id_head = int(sp[1])
                x_partical_tail = glass.frames[time_ini, id_tail, 2]
                y_particle_tail = glass.frames[time_ini, id_tail, 3]

                # get partile and quasivoid position at the head
                x_particle_head = glass.frames[time_final, id_head, 2]
                y_particle_head = glass.frames[time_final, id_head, 3]
                dist_headtail_vec = distPBC2D([x_particle_head - x_partical_tail, y_particle_head - y_particle_tail],
                                              glass.L[0], glass.L[1])
                dist_headtail = np.linalg.norm(dist_headtail_vec)
                if dist_headtail > dr_ht:
                    if glass.stringlength[ti] >= Length_of_string:
                        sp1 = stringparticles[1:-1]
                        time_ini = glass.starend_of_string[ti][0]
                        time_final = glass.starend_of_string[ti][1]
                        for sti, index_sti in enumerate(glass.starend_of_string[ti]):
                            if crt_in:  # the distance between the internal particle and string tail and head is lager than Rc
                                new_sp = []
                                for particle_in_string in sp1:
                                    dr1 = glass.frames[index_sti, particle_in_string, 2:4] - glass.frames[index_sti,
                                                                                             stringparticles[0], 2:4]
                                    dr2 = glass.frames[index_sti, particle_in_string, 2:4] - glass.frames[index_sti,
                                                                                             stringparticles[-1], 2:4]
                                    dr1_pbc = distPBC2D(dr1, glass.L[0], glass.L[1])
                                    dr2_pbc = distPBC2D(dr2, glass.L[0], glass.L[1])
                                    dr1_norm = np.linalg.norm(dr1_pbc)
                                    dr2_norm = np.linalg.norm(dr2_pbc)
                                    if (dr1_norm >= regionradius) & (dr2_norm >= regionradius):
                                        new_sp.append(particle_in_string)
                                sp = new_sp
                            if choose_middle:
                                n_in_string = len(stringparticles)
                                if n_in_string % 2 == 0 and n_in_string >= 3:
                                    sp = [stringparticles[n_in_string // 2], stringparticles[n_in_string // 2 + 1]]
                                elif n_in_string % 2 == 1 and n_in_string >= 3:
                                    sp = [stringparticles[n_in_string // 2 + 1]]
                            for particle_in_string in sp:
                                x_particle_in_string = glass.frames[sti, particle_in_string, 2]
                                y_particle_in_string = glass.frames[sti, particle_in_string, 3]
                                CalRegion_in_string, x_low, x_up, y_low, y_up = select_region_pbc(glass,
                                                                                                  x_particle_in_string,
                                                                                                  y_particle_in_string,
                                                                                                  regionradius)
                                ld_particle_in_string, _ = cal_density_vs_time(glass, CalRegion_in_string, x_low, x_up,
                                                                               y_low, y_up, x_particle_in_string,
                                                                               y_particle_in_string, regionradius)
                                mask = ((CalRegion_in_string[sti, :, 2] > x_low) & (
                                            CalRegion_in_string[sti, :, 2] < x_up)
                                        & (CalRegion_in_string[sti, :, 3] > y_low) & (
                                                    CalRegion_in_string[sti, :, 3] < y_up))
                                mask2 = np.isin(CalRegion_in_string[0, :, 0] - 1, stringparticles)
                                mask = mask & mask2
                                ld_in_string.append(ld_particle_in_string[sti])
                                if figshow:
                                    fig4, ax4 = plt.subplots()
                                    fig0, ax0 = plt.subplots()
                                    ax4.plot(ld_particle_in_string, c='k',
                                             label=f'Local density of particle {particle_in_string} in string at time {sti}',
                                             marker='^')
                                    entire_density = LDC.Local_Density_Calculator(glass, regionradius, [sti])
                                    norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                                    LDC.show_heterogeneity(ax0, glass, entire_density, norm, sti,
                                                           [x_low, x_up, y_low, y_up])
                                    glass.set_ax(ax0, x_particle_in_string, y_particle_in_string, regionradius, x_low,
                                                 x_up, y_low, y_up)
                                    glass.plot_collections(ax0, mask, CalRegion_in_string, time_ini, time_final, ts, ti,
                                                           'In string')
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
        if mode == 0:
            return density_tail_ini, density_tail_final, density_head_ini, density_head_final, time_select
        elif mode == 1:
            return ld_in_string


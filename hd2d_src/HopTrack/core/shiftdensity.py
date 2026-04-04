import matplotlib.pyplot as plt
import numpy as np
from hd2d_src.HopTrack.core import *
from hd2d_src.HopTrack.core.share import *
from hd2d_src.HopTrack.core import Local_Density_Calculator as LDC
import matplotlib.colors as mcolors


def shiftdensity(glass, shifting_window, regionradius, r_insert, rho, savefile=None, figshow=False,
                    pid=None, select=False, dr_ht=5):
        # mode = 0  calculate the local density at string ends
        # mode = 1 calculate the local density of the particles in the string （without ends）

        # get particle id in string
        density_tail_ini = []
        density_head_ini = []
        density_tail_final = []
        density_head_final = []
        ld_in_string = []
        time_select = []
        ts, _, _ = np.shape(glass.frames)
        time_ini_list = np.array([tif[0] for tif in glass.starend_of_string])
        for ti, stringparticles in enumerate(glass.connected_components):
            if select:
                if pid != stringparticles[0]:
                    continue
            sp = [stringparticles[0], stringparticles[-1]]
            time_ini = glass.starend_of_string[ti][0] # this is the time after nseg choosing
            time_final = glass.starend_of_string[ti][-1]
            shift_ini = time_ini + shifting_window
            shift_next = time_ini_list == shift_ini
            shift_final = time_final + shifting_window
            if shift_next.any():
                print(f'hop time = {time_ini}, shift_time = {shift_final}')
                shift_ms = [component for component, select in zip(glass.connected_components, shift_next) if
                            select]
                for shift_i in shift_ms:
                    if figshow:
                        fig4, ax4 = plt.subplots()
                        fig0, ax0 = plt.subplots(2, 2, figsize=(8, 8))

                    # get particle and quasivoid position at the tail
                    id_tail = int(sp[0])
                    id_shift_head = int(shift_i[-1])
                    x_partical_tail = glass.frames[time_ini, id_tail, 2]
                    y_particle_tail = glass.frames[time_ini, id_tail, 3]
                    x_quasivoid_tail = glass.frames[time_final, id_tail, 2]
                    y_quasivoid_tail = glass.frames[time_final, id_tail, 3]
                    x_quasivoid_tail = particle_unwrap(x_partical_tail, x_quasivoid_tail,
                                                       glass.L[0])  # unwrap when particle hops to the otherside
                    y_quasivoid_tail = particle_unwrap(y_particle_tail, y_quasivoid_tail, glass.L[1])

                    # get partile and quasivoid position at the head
                    x_particle_head = glass.frames[shift_final, id_shift_head, 2]
                    y_particle_head = glass.frames[shift_final, id_shift_head, 3]
                    x_quasivoid_head = glass.frames[shift_ini, id_shift_head, 2]
                    y_quasivoid_head = glass.frames[shift_ini, id_shift_head, 3]
                    x_quasivoid_head = particle_unwrap(x_particle_head, x_quasivoid_head, glass.L[0])
                    y_quasivoid_head = particle_unwrap(y_particle_head, y_quasivoid_head, glass.L[1])

                    dist_headtail_vec = distPBC2D([x_quasivoid_head - x_quasivoid_tail, y_quasivoid_head - y_quasivoid_tail],
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
                        if r_insert != 0:
                            ld_ti, mindr_ti = cal_density_vs_time_with_ri(glass, CalRegion_tail_ini, x_low_ti,
                                                                          x_up_ti, y_low_ti,
                                                                          y_up_ti,
                                                                          x_partical_tail, y_particle_tail,
                                                                          regionradius)
                            ld_tf, mindr_tf = cal_density_vs_time_with_ri(glass, CalRegion_tail_final, x_low_tf,
                                                                          x_up_tf, y_low_tf,
                                                                          y_up_tf, x_quasivoid_tail,
                                                                          y_quasivoid_tail, regionradius)
                            ld_hf, mindr_hf = cal_density_vs_time_with_ri(glass, CalRegion_head_final, x_low_hf,
                                                                          x_up_hf, y_low_hf,
                                                                          y_up_hf,
                                                                          x_particle_head, y_particle_head,
                                                                          regionradius)
                            ld_hi, mindr_hi = cal_density_vs_time_with_ri(glass, CalRegion_head_ini, x_low_hi,
                                                                          x_up_hi, y_low_hi,
                                                                          y_up_hi,
                                                                          x_quasivoid_head, y_quasivoid_head,
                                                                          regionradius)

                            if mindr_tf[time_ini] > r_insert and mindr_hi[time_final] > r_insert:
                                t = time_ini
                                mask_ti = ((CalRegion_tail_ini[t, :, 2] > x_low_ti) & (
                                            CalRegion_tail_ini[t, :, 2] < x_up_ti) &
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
                                mask_hi = ((CalRegion_head_ini[t, :, 2] > x_low_hi) & (
                                            CalRegion_head_ini[t, :, 2] < x_up_hi) &
                                           (CalRegion_head_ini[t, :, 3] > y_low_hi) & (
                                                       CalRegion_head_ini[t, :, 3] < y_up_hi))

                                density_tail_ini.append(ld_ti[time_ini])
                                density_tail_final.append(ld_tf[time_final])
                                density_head_final.append(ld_hf[time_final])
                                density_head_ini.append(ld_hi[time_ini])
                                time_select = [time_ini, time_final]
                                if figshow:
                                    entire_density_initial = LDC.Local_Density_Calculator(glass, regionradius,
                                                                                          [time_ini])
                                    entire_density_final = LDC.Local_Density_Calculator(glass, regionradius,
                                                                                        [time_final])
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
                                    glass.set_ax(ax0[0, 0], x_partical_tail, y_particle_tail, regionradius,
                                                 x_low_ti, x_up_ti, y_low_ti,
                                                 y_up_ti, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                                 y_quasivoid_head)
                                    glass.set_ax(ax0[0, 1], x_quasivoid_tail, y_quasivoid_tail, regionradius,
                                                 x_low_tf, x_up_tf, y_low_tf,
                                                 y_up_tf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                                 y_quasivoid_head)
                                    glass.set_ax(ax0[1, 0], x_quasivoid_head, y_quasivoid_head, regionradius,
                                                 x_low_hi, x_up_hi, y_low_hi,
                                                 y_up_hi, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                                 y_quasivoid_head)
                                    glass.set_ax(ax0[1, 1], x_particle_head, y_particle_head, regionradius,
                                                 x_low_hf, x_up_hf, y_low_hf,
                                                 y_up_hf, x_quasivoid_tail, y_quasivoid_tail, x_quasivoid_head,
                                                 y_quasivoid_head)

                                    # plot string
                                    glass.plot_collections(ax0[0, 0], mask_ti, CalRegion_tail_ini, time_ini,
                                                           time_final, ts, ti, 'initial tail')
                                    glass.plot_collections(ax0[0, 1], mask_tf, CalRegion_tail_final, time_ini,
                                                           time_final, ts, ti,
                                                           'final tail')
                                    glass.plot_collections(ax0[1, 0], mask_hi, CalRegion_head_ini, time_ini,
                                                           time_final, ts, ti,
                                                           'initial head')
                                    glass.plot_collections(ax0[1, 1], mask_hf, CalRegion_head_final, time_ini,
                                                           time_final, ts, ti,
                                                           'final head')
                                    ax4.vlines([time_ini, time_final], 0.75, 0.85, color=['r', 'b'],
                                               linestyles='--')
                                    ax4.set_xlabel('Time')
                                    ax4.set_ylabel('Area density in black circle region')
                                    ax4.set_title(f'locale density of string #{ti}')
                                    ax4.legend()
                                    ax4.grid()
                                    if savefile:
                                        fig0.savefig(str(savefile) + f'_string{ti:03d}_config.png', dpi=600)
                                        plt.close(fig0)
                                        fig4.savefig(str(savefile) + f'_string{ti:03d}_tail_and_head_density.png',
                                                     dpi=600)
                                        plt.close(fig4)
                                    else:
                                        plt.show()

                        else:
                            ld_ti = cal_density_vs_time(glass, CalRegion_tail_ini, x_low_ti, x_up_ti, y_low_ti,
                                                                  y_up_ti,
                                                                  x_partical_tail, y_particle_tail, regionradius)
                            ld_tf = cal_density_vs_time(glass, CalRegion_tail_final, x_low_tf, x_up_tf, y_low_tf,
                                                                  y_up_tf, x_quasivoid_tail, y_quasivoid_tail, regionradius)
                            ld_hf = cal_density_vs_time(glass, CalRegion_head_final, x_low_hf, x_up_hf, y_low_hf,
                                                                  y_up_hf,
                                                                  x_particle_head, y_particle_head, regionradius)
                            ld_hi = cal_density_vs_time(glass, CalRegion_head_ini, x_low_hi, x_up_hi, y_low_hi,
                                                                  y_up_hi,
                                                                  x_quasivoid_head, y_quasivoid_head, regionradius)

                            # plot local density vs time
                            if figshow:
                                ax4.plot(ld_ti, c='r', label='Initial tail density', marker='o')
                                ax4.plot(ld_tf, linestyle='--', c='b', label='Final tail density', marker='x')
                                ax4.plot(ld_hi, c='r', label='Initial head density', marker='s')
                                ax4.plot(ld_hf, linestyle='--', c='b', label='Final head density', marker='*')

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

                            density_head_final.append(ld_hf[shift_final])
                            density_head_ini.append(ld_hi[shift_ini])
                            time_select = [time_ini, shift_final]
                            if figshow:
                                entire_density_initial = LDC.Local_Density_Calculator(glass, regionradius, [time_ini])
                                entire_density_final = LDC.Local_Density_Calculator(glass, regionradius, [time_final])
                                norm = mcolors.Normalize(vmin=rho - 0.1, vmax=rho + 0.1)
                                LDC.show_heterogeneity(ax0[0, 0], glass, entire_density_initial, norm, time_ini,
                                                       [x_low_ti, x_up_ti, y_low_ti, y_up_ti])
                                LDC.show_heterogeneity(ax0[1, 0], glass, entire_density_initial, norm, time_ini,
                                                       [x_low_hi, x_up_hi, y_low_hi, y_up_hi])
                                LDC.show_heterogeneity(ax0[0, 1], glass, entire_density_final, norm, shift_final,
                                                       [x_low_tf, x_up_tf, y_low_tf, y_up_tf])
                                LDC.show_heterogeneity(ax0[1, 1], glass, entire_density_final, norm, shift_final,
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
                density_tail_ini.append(ld_ti[time_ini])
                density_tail_final.append(ld_tf[shift_final])

        return density_tail_ini, density_tail_final, density_head_ini, density_head_final, time_select



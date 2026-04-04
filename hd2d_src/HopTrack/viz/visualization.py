import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from hd2d_src.HopTrack.core import Local_Density_Calculator as LDC
import matplotlib.colors as mcolors
from shapely.geometry import Point
from hd2d_src.HopTrack.core import *
from matplotlib.patches import Polygon as MatplotlibPolygon
from shapely.geometry import Polygon, MultiPolygon
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from matplotlib.colors import Normalize

def showdisp(glass, fig,  ax, t_start, t_end, sid=0, lw=0.5, ms=0.5, nodot=False, showvoid=False,
             showpid=False, showforcepid=0, showforcedid=0, colorbar=False,
             radii=None, showradii=False, overlap=True, blacktraj=False, crange=[0, 1000]):  # get rgba matrix without particle radii
    line_collections = []
    norm = mcolors.Normalize(vmin=crange[0], vmax=crange[1])
    cmap = cm.get_cmap('rainbow')
    # cmap = cm.get_cmap('Greys')
    cmap = cmap.reversed()
    # cmap = cm.colors.ListedColormap(cmap(np.linspace(0, 0.5, 100)))
    t1, n, d = np.shape(glass.frames)
    glass.n = n
    t_length = t_end - t_start + 1
    if blacktraj:
        for i in range(glass.n):
            cline = np.zeros((t_length - 1, 2, 2))
            cline[:, 0, :] = glass.frames[t_start:t_end, i, 2:4]
            cline[:, 1, :] = glass.frames[t_start + 1:t_end + 1, i, 2:4]
            line_collections.append(LineCollection(cline, linewidths=lw, color='k', zorder=4))
    else:
        for i in range(glass.n):
            cline = np.zeros((t_length - 1, 2, 2))
            cline[:, 0, :] = glass.frames[t_start:t_end, i, 2:4]
            cline[:, 1, :] = glass.frames[t_start + 1:t_end + 1, i, 2:4]
            line_collections.append(LineCollection(cline, linewidths=lw, cmap=cmap, norm=norm, zorder=4))
    for i, lc in enumerate(line_collections):
        lc.set_zorder(10)
        time_values = np.arange(t_start, t_end)  # ºÍ cline ¶ÔÓ¦µÄÊ±¼ä¶Î
        lc.set_array(time_values)
        ax.add_collection(lc)
        del lc
    if colorbar:
        cb = fig.colorbar(line_collections[0], ax=ax, cmap=cmap, pad=0.1)
        x = np.linspace(t_start, t_end-1, 5)
        cb.set_ticks(x)
        cblabel = cb.ax.set_ylabel('Time')
        cblabel.set_fontsize(16)
        ax.tick_params(axis='both', labelsize=16)
        cb.ax.tick_params(labelsize=16)
    line_collections.clear()
    if not nodot:
        ax.plot(glass.frames[t_start, 0:glass.n, 2], glass.frames[t_start, 0:glass.n, 3], 'ro', markersize=ms, zorder=0)
    if showvoid:
        if np.ndim(glass.void) == 1:
            ax.plot(glass.void[0], glass.void[1], 'ms', markersize=2 * ms, zorder=1)
        else:
            ax.plot(glass.void[:, 0], glass.void[:, 1], 'ms', markersize=2 * ms, zorder=1)
    if showpid:
        for i in range(glass.n):
            ax.text(glass.frames[t_start, i, 2], glass.frames[t_start, i, 3], str(int(glass.frames[t_start, i, 0])))
    if showforcepid:
        showpaid = np.where(glass.frames[t_start, :, 0] == showforcepid)[0][0]
        print(glass.frames[t_start, showpaid, :])
        print(showforcedid)
        ax.plot(glass.frames[t_start, showpaid, 2], glass.frames[t_start, showpaid, 3], 'go', markersize=2 * ms, alpha=0.9)
        ax.arrow(glass.frames[t_start, showpaid, 2], glass.frames[t_start, showpaid, 3], np.cos(showforcedid * np.pi / 8),
                 np.sin(showforcedid * np.pi / 8), width=0.1, head_width=0.2, head_length=0.2,
                 length_includes_head=True, color='purple', zorder=10)
    if glass.n != glass.n0:
        ax.set(xlim=(glass.x_low, glass.x_up), ylim=(glass.y_low, glass.y_up), aspect="equal")
    else:
        ax.set(xlim=(0, glass.L[0]), ylim=(0, glass.L[1]), aspect="equal")
    if not glass.radii.any():
        if radii is not None:
            with open(radii) as rf:
                radii = np.loadtxt(rf)
            radii = np.array(radii)
            glass.radii = np.zeros((glass.n, 1))
            for i in range(glass.n):
                glass.radii[i] = radii[int(glass.frames[t_start, i, 1] - 1), 1]
    if showradii:
        if hasattr(glass, 'starend_of_string'):
            if len(glass.starend_of_string) >= 2:
                print("more than one string is found!, check the frame number, and provide the choosen microstring")
                t_si = glass.starend_of_string[sid][0]
                t_sf = glass.starend_of_string[sid][1]
            else:
                t_si = glass.starend_of_string[0][0]
                t_sf = glass.starend_of_string[0][1]
        else:
            t_si = 0
            t_sf = -1
        for icir in range(glass.n):
            if overlap == 1:
                x = glass.frames[t_si, icir, 2]
                y = glass.frames[t_si, icir, 3]
                r = glass.radii[int(glass.frames[t_si, icir, 0] - 1)]
                citcle = plt.Circle((x, y), r, color='k', fill=False, zorder=1, linewidth=0.5, alpha=1)
                ax.add_patch(citcle)

                x = glass.frames[-1, icir, 2]
                y = glass.frames[-1, icir, 3]
                r = glass.radii[int(glass.frames[t_sf, icir, 0] - 1)]
                citcle = plt.Circle((x, y), r, color='b', fill=False, zorder=0, linewidth=0.5, alpha=0.5)
                ax.add_patch(citcle)
            elif overlap == -1:
                x = glass.frames[-1, icir, 2]
                y = glass.frames[-1, icir, 3]
                r = glass.radii[int(glass.frames[t_sf, icir, 0] - 1)]
                citcle = plt.Circle((x, y), r, color='k', fill=False, zorder=1, linewidth=0.5, alpha=1)
                ax.add_patch(citcle)

                x = glass.frames[t_si, icir, 2]
                y = glass.frames[t_si, icir, 3]
                r = glass.radii[int(glass.frames[t_si, icir, 0] - 1)]
                citcle = plt.Circle((x, y), r, color='b', fill=False, zorder=0, linewidth=0.5, alpha=0.5)
                ax.add_patch(citcle)

            elif overlap == 0:
                x = glass.frames[-1, icir, 2]
                y = glass.frames[-1, icir, 3]
                r = glass.radii[int(glass.frames[t_sf, icir, 0] - 1)]
                citcle = plt.Circle((x, y), r, facecolor='w', edgecolor='k', fill=True, zorder=5, linewidth=0.5, alpha=1)
                ax.add_patch(citcle)
            else:
                x = glass.frames[t_si, icir, 2]
                y = glass.frames[t_si, icir, 3]
                r = glass.radii[int(glass.frames[t_si, icir, 0] - 1)]
                citcle = plt.Circle((x, y), r, facecolor='w', edgecolor='k', fill=True, zorder=5, linewidth=0.5, alpha=1)
                ax.add_patch(citcle)

        ax.tick_params(axis='x', labelsize=4)
        ax.tick_params(axis='y', labelsize=4)


def showstring(glass, ax, lw=0.8, nodot=False, findquasivoid=False, size=5, showid=False,
               select=False, sid=0, show_length_distribusion=False, showtraj=True,
               stringID=None, SSC=None, WL=False, show_localdensity_region=False, Rc = 4,
               mode=0, reverse=False, colors = ['m', 'y', 'g', 'b'], showvec=False):
    if not hasattr(glass, 'connected_components'):
        print('no string is found!!')
    else:
        Tcolor = len(glass.frames)
        cmap = cm.get_cmap('rainbow')
        norm = mcolors.Normalize(vmin=0, vmax=len(glass.connected_components))
        if SSC: # set the same string color
            dotcolor = [SSC for i in range(0, len(glass.connected_components))]
            colors = [SSC for i in range(0, Tcolor)]
        else:
            colors = cmap(np.linspace(0, 1, Tcolor))
            dotcolor = [cmap(norm(i)) for i in range(0, len(glass.connected_components))]
        if stringID:
            IDlist = stringID
            new_strings = tuple(glass.connected_components[idstr] for idstr in stringID)
            new_startend = tuple(glass.starend_of_string[idstr] for idstr in stringID)
            glass.connected_components = new_strings
        else:
            IDlist = range(0, len(glass.connected_components))
            new_startend = glass.starend_of_string
        for i, stringparticles in enumerate(glass.connected_components):
            if select:
                if sid != i:
                    continue
            # print(stringparticles)
            starttime = new_startend[i][0]
            endtime = new_startend[i][1] #np.shape(glass.frames)[0]-1 #new_startend[i][1]
            line_collections = []
            cmap = cm.get_cmap('rainbow')
            cmap = cmap.reversed()
            if showtraj:
                for iline in stringparticles:
                    if endtime - starttime > 1:
                        # time_range = np.linspace(starttime/Tcolor, 1, endtime - starttime)
                        cline = np.zeros((endtime - starttime, 2, 2))
                        cline[:, 0, :] = glass.frames[starttime:endtime, iline, 2:4]
                        cline[:, 1, :] = glass.frames[starttime + 1:endtime + 1, iline, 2:4]
                        line_collections.append(LineCollection(cline, linewidths=lw, cmap=cmap, zorder=5))
                        # line_collections[-1].set_array(time_range)
                    else:
                        cline = np.zeros((1, 2, 2))
                        cline[:, 0, :] = glass.frames[starttime, iline, 2:4]
                        cline[:, 1, :] = glass.frames[endtime, iline, 2:4]
                        line_collections.append(LineCollection(cline, linewidths=lw, cmap=cmap, zorder=5))
                for icline, lc in enumerate(line_collections):
                    ax.add_collection(lc)
                    lc.set_array(np.arange(len(lc.get_segments())))
                line_collections.clear()
            if showvec:
                for iline in stringparticles:
                    dx = glass.frames[endtime, iline, 2] - glass.frames[starttime, iline, 2]
                    dy = glass.frames[endtime, iline, 3] - glass.frames[starttime, iline, 3]
                    ax.arrow(glass.frames[starttime, iline, 2], glass.frames[starttime, iline, 3], dx-0.4*(dx/(dx**2+dy**2)**0.5), dy-0.4*(dy/(dx**2+dy**2)**0.5), color='r', head_width=0.2, head_length=0.4)

            if not nodot:
                # ax.plot(glass.frames[starttime, stringparticles, 2], glass.frames[starttime, stringparticles, 3],
                #             'o', markersize=size, color=dotcolor[i],
                #             zorder=0)
                for pinsdex, pins in enumerate(stringparticles):
                    itype = int(glass.frames[0, pins, 0] - 1)
                    r = glass.radii[itype]
                    circle = plt.Circle((glass.frames[starttime, pins, 2], glass.frames[starttime, pins, 3]),
                                        r, color='r', fill=False, zorder=8, linewidth=0.5,
                                        alpha=0.5)
                    ax.add_artist(circle)
            else:
                if reverse:
                    for par_i  in range(len(glass.frames[0,:,0])):
                        if not par_i in stringparticles:
                            itype = int(glass.frames[0, par_i, 0] - 1)
                            r = glass.radii[itype]
                            circle = plt.Circle((glass.frames[endtime, par_i, 2], glass.frames[endtime, par_i, 3]),
                                                r, color='lightgray', fill=True, zorder=6, linewidth=0.5,
                                                alpha=1)
                            ax.add_artist(circle)
                        else:
                            itype = int(glass.frames[0, par_i, 0] - 1)
                            r = glass.radii[itype]
                            circle = plt.Circle((glass.frames[endtime, par_i, 2], glass.frames[endtime, par_i, 3]), r,
                                                edgecolor='b', fill=True, facecolor='w', zorder=6, linewidth=1,
                                                alpha=1, )
                            ax.add_artist(circle)
                else:
                    for par_i  in range(len(glass.frames[0,:,0])):
                        if not par_i in stringparticles:
                            itype = int(glass.frames[0, par_i, 0] - 1)
                            r = glass.radii[itype]
                            circle = plt.Circle((glass.frames[starttime, par_i, 2], glass.frames[starttime, par_i, 3]),
                                                r, color='lightgray', fill=True, zorder=6, linewidth=0.5,
                                                alpha=1)
                            ax.add_artist(circle)
                        else:
                            itype = int(glass.frames[0, par_i, 0] - 1)
                            r = glass.radii[itype]
                            circle = plt.Circle((glass.frames[starttime, par_i, 2], glass.frames[starttime, par_i, 3]), r,
                                                edgecolor='r', fill=True, facecolor='w', zorder=6, linewidth=1,
                                                alpha=1)
                            ax.add_artist(circle)

            if showid:
                for iid in range(len(stringparticles)):
                    # print(glass.frames[0, stringparticles[i], 0])
                    # ax.text(glass.frames[0, stringparticles[i], 2], glass.frames[0, stringparticles[i], 3],
                    #         f'{int(glass.frames[0, stringparticles[i], 0])}', ha="center", va="bottom")
                    ax.text(glass.frames[0, stringparticles[iid], 2], glass.frames[0, stringparticles[iid], 3],
                            f'{int(stringparticles[iid])}', ha="center", va="bottom")  # the index in stringparticle is the same with slice number in glass.frames
            if findquasivoid:
                if WL: # plot quasivoid location at the first and final frame of the duration
                    t_si = 0
                    t_sf = -1
                    t_Qt = -1
                    t_Qh = 0
                else:
                    t_si = starttime
                    t_sf = endtime
                    t_Qt = int(starttime + 1)
                    t_Qh = int(endtime - 1)
                x_Qt = glass.frames[t_Qt, stringparticles[0], 2]
                y_Qt = glass.frames[t_Qt, stringparticles[0], 3]
                x_Qh = glass.frames[t_Qh, stringparticles[-1], 2]
                y_Qh = glass.frames[t_Qh, stringparticles[-1], 3]
                if not show_localdensity_region:
                    # ax.plot(x_Qt, y_Qt, 's', markersize = size * 2, color = 'g', zorder = 0)
                    # ax.plot(x_Qh, y_Qh, 'o', markersize = size * 2, color = 'b', zorder = 0)
                    # head
                    itype = glass.frames[t_Qh, stringparticles[-1], 1]
                    r = glass.radii[int(itype-1)]
                    circle = plt.Circle((glass.frames[starttime,  stringparticles[-1], 2], glass.frames[starttime,  stringparticles[-1], 3]),
                                        r, color='r', fill=True, zorder=8, linewidth=0.5,
                                        alpha=0.5)
                    ax.add_artist(circle)
                    #tail
                    itype = glass.frames[t_Qt, stringparticles[0], 1]
                    r = glass.radii[int(itype - 1)]
                    circle = plt.Circle((glass.frames[endtime,  stringparticles[0], 2], glass.frames[endtime,  stringparticles[0], 3]),
                                        r, color='b', fill=False, zorder=8, linewidth=1, linestyle='--',
                                        alpha=1)
                    ax.add_artist(circle)
                if show_localdensity_region:
                    if mode == 1 or mode == 5 or mode == 50 or mode==6: # background is the final postion of particles
                        x_Qt = glass.frames[t_si, stringparticles[0], 2]
                        y_Qt = glass.frames[t_si, stringparticles[0], 3]
                    elif mode == 2:# background is the initial postion of particles
                        x_Qh = glass.frames[t_sf, stringparticles[-1], 2]
                        y_Qh = glass.frames[t_sf, stringparticles[-1], 3]
                    elif mode == 3 or mode == 4:
                        x_Qt = glass.frames[t_si, stringparticles[0], 2]
                        y_Qt = glass.frames[t_si, stringparticles[0], 3]
                        x_Qh = glass.frames[t_sf, stringparticles[-1], 2]
                        y_Qh = glass.frames[t_sf, stringparticles[-1], 3]
                    x_middle = glass.frames[t_si, stringparticles[int(len(stringparticles)/2-2)], 2]
                    y_middle = glass.frames[t_si, stringparticles[int(len(stringparticles)/2-2)], 3]
                    def plot_Rc(circle_center, Rc, t_s, P_color='w', zorder=0, alpha=1):
                        circle_region = plt.Circle(circle_center, Rc, facecolor='grey', edgecolor='black', zorder=0, alpha=0.1)
                        ax.add_patch(circle_region)
                        x_Q, y_Q = circle_center
                        S = 0
                        for icir in range(glass.n):
                            x = glass.frames[t_s, icir, 2]
                            y = glass.frames[t_s, icir, 3]
                            r = glass.radii[int(glass.frames[0, icir, 0] - 1)]
                            distance = np.linalg.norm([x - x_Q, y - y_Q])
                            S = S + circle_intersection_area(x_Q, y_Q, Rc, x, y, r, glass.L[0], glass.L[1])
                            if distance + r < Rc:
                                C_particle = plt.Circle((x, y), r, color=P_color, fill=True, zorder=zorder, linewidth=0.1,
                                                        alpha=alpha)
                                ax.add_patch(C_particle)
                            elif distance - r < Rc:
                                C_particle = plt.Circle((x, y), r, color=P_color, fill=True, zorder=zorder, linewidth=0.1,
                                                        alpha=alpha)
                                C_particle.set_clip_path(circle_region)
                                ax.add_patch(C_particle)
                        return S

                    def get_region_mask(circle_center, Rc, t_s):
                        x_Q, y_Q = circle_center
                        skin = max(glass.radii)
                        xlow = np.mod(x_Q - Rc - skin, glass.L[0])
                        xup = np.mod(x_Q + Rc + skin, glass.L[0])
                        ylow = np.mod(y_Q - Rc - skin, glass.L[1])
                        yup = np.mod(y_Q + Rc + skin, glass.L[1])
                        if xlow > xup:
                            xregion = (glass.frames[t_s, :, 2] > xlow) | (glass.frames[t_s, :, 2] < xup)
                        else:
                            xregion = (glass.frames[t_s, :, 2] > xlow) & (glass.frames[t_s, :, 2] < xup)
                        if ylow > yup:
                            yregion = (glass.frames[t_s, :, 3] > ylow) | (glass.frames[t_s, :, 3] < yup)
                        else:
                            yregion = (glass.frames[t_s, :, 3] > ylow) & (glass.frames[t_s, :, 3] < yup)
                        region_mask = xregion & yregion
                        return region_mask
                    def calc_fragments(ax1, circle_center, Rc, t_if, color):
                        x_Q, y_Q = circle_center
                        t_i, t_f = t_if
                        mask_i = get_region_mask(circle_center, Rc, t_i)
                        mask_f = get_region_mask(circle_center, Rc, t_f)
                        mask = mask_i | mask_f
                        particle_in_region_i = glass.frames[t_i, mask, :]
                        particle_index = glass.frames[t_i, mask, 0] - 1
                        r_i = [glass.radii[int(i)] for i in particle_index]
                        particle_in_region_f = glass.frames[t_f, mask, :]
                        particle_index = glass.frames[t_f, mask, 0] - 1
                        r_f = [glass.radii[int(i)] for i in particle_index]

                        def create_circle(center, radius):
                            return Point(center).buffer(radius)

                        def plot_intersection(ax, shape, color='r', alpha=1):
                            """Plot the specified intersection area."""
                            # x,y = shape.exterior.xy
                            if isinstance(shape, MultiPolygon):
                                for polygon in shape.geoms:  # Use `geoms` to iterate over individual `Polygon` objects
                                    mp_polygon = MatplotlibPolygon(polygon.exterior.coords, closed=True,
                                                                   fill=True, facecolor=color,
                                                                   edgecolor='black', alpha=alpha, zorder=5)
                                    ax1.add_patch(mp_polygon)
                            elif isinstance(shape, Polygon):
                                mp_polygon = MatplotlibPolygon(shape.exterior.coords, closed=True, fill=True,
                                                               alpha=alpha, edgecolor='black', facecolor=color, zorder=5)
                                ax.add_patch(mp_polygon)
                            # ax.plot(x, y, color=color, alpha=alpha, linewidth=2, solid_capstyle='round', zorder=2)


                        circle1 = create_circle((x_Q, y_Q), Rc)
                        S_fragments = 0
                        S_diff_list = []
                        S_diff_list3 = []
                        # fig1, ax1 = plt.subplots()
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
                                    if S_in2.area > 0:
                                        testing.plot_test(ax1, (x_Q, y_Q), Rc, (x_i, y_i), r_pi, (x_f, y_f), r_pf)
                                        S_diff = S_in2.difference(S_in3)
                                        S_diff_list.append(S_diff)
                                    S_fragments =  S_fragments + (S_in2.area - S_in3.area)
                                    # print(f'S_in2-S_in3 is {S_in2.area-S_in3.area}')
                                else:
                                    # print(f'S_in3 is {S_in3.area}')
                                    if S_in3.area > 0:
                                        testing.plot_test(ax1, (x_Q, y_Q), Rc, (x_i, y_i), r_pi, (x_f, y_f), r_pf)
                                        S_diff_list3.append(S_in3)
                                    S_fragments = S_fragments - S_in3.area
                        for S_polygon_i in S_diff_list:
                            for S_ploygon_j in S_diff_list3:
                                S_polygon_i = S_polygon_i.difference(S_ploygon_j)
                            plot_intersection(ax1, S_polygon_i, color=color, alpha=1)
                        return S_fragments
                    # plot the fragments of quasivoid
                    if mode == 1: # blue background
                        plot_Rc((x_Qt, y_Qt), Rc, t_si, P_color='w', zorder=1, alpha=1)
                        plot_Rc((x_Qt, y_Qt), Rc, t_sf, P_color='b', zorder=0, alpha=0.5)
                        plot_Rc((x_Qh, y_Qh), Rc, t_si, P_color='w', zorder=1, alpha=1)
                        plot_Rc((x_Qh, y_Qh), Rc, t_sf, P_color='b', zorder=0, alpha=0.5)
                    elif mode == 2:
                        plot_Rc((x_Qt, y_Qt), Rc, t_si, P_color='r', zorder=0, alpha=0.5)
                        plot_Rc((x_Qt, y_Qt), Rc, t_sf, P_color='w', zorder=1, alpha=1)
                        plot_Rc((x_Qh, y_Qh), Rc, t_si, P_color='r', zorder=0, alpha=0.5)
                        plot_Rc((x_Qh, y_Qh), Rc, t_sf, P_color='w', zorder=1, alpha=1)
                    elif mode == 3:
                        S_ti = plot_Rc((x_Qt, y_Qt), Rc, t_si, P_color='w', zorder=1, alpha=1)
                        S_tf = plot_Rc((x_Qt, y_Qt), Rc, t_sf, P_color='b', zorder=0, alpha=0.5)
                        S_hi = plot_Rc((x_Qh, y_Qh), Rc, t_si, P_color='r', zorder=0, alpha=0.5)
                        S_hf = plot_Rc((x_Qh, y_Qh), Rc, t_sf, P_color='w', zorder=1, alpha=1)
                        ax.text(x_Qt, y_Qt-Rc, fr'$\Delta S_{{sum area of particle}} =${(S_tf - S_ti)/(np.pi * 0.4993804085195311 **2):.03f}')
                        ax.text(x_Qh, y_Qh-Rc, fr'$\Delta S_{{sum area of particle}} =${-(S_hf - S_hi)/(np.pi * 0.4993804085195311 **2):.03f}')
                    elif mode == 4:
                        S_ti = plot_Rc((x_Qt, y_Qt), Rc, t_si, P_color='w', zorder=1, alpha=1)
                        S_tf = plot_Rc((x_Qt, y_Qt), Rc, t_sf, P_color='b', zorder=0, alpha=0.5)
                        S_hi = plot_Rc((x_Qh, y_Qh), Rc, t_si, P_color='r', zorder=0, alpha=0.5)
                        S_hf = plot_Rc((x_Qh, y_Qh), Rc, t_sf, P_color='w', zorder=1, alpha=1)
                        ax.text(x_Qt, y_Qt - Rc-0.5,
                                fr'$\Delta S_{{sum area of particle}} =${(S_tf - S_ti) / (np.pi * 0.4993804085195311 ** 2):.03f}')
                        ax.text(x_Qh, y_Qh - Rc-0.5,
                                fr'$\Delta S_{{sum area of particle}} =${-(S_hf - S_hi) / (np.pi * 0.4993804085195311 ** 2):.03f}')
                        #calculate the fragments area
                        s_frag1 = calc_fragments(ax, (x_Qt, y_Qt), Rc, (t_si, t_sf), color='b')
                        s_frag2 = calc_fragments(ax, (x_Qh, y_Qh), Rc, (t_sf, t_si), color='r')
                        ax.text(x_Qt, y_Qt-Rc-1, f'S_blue_area = {(s_frag1)/(np.pi * 0.4993804085195311 **2):.03f}', color='r')
                        ax.text(x_Qh, y_Qh-Rc-1, f'S_red_area = {(s_frag2)/(np.pi * 0.4993804085195311 **2):.03f}', color='r')
                    elif mode == 5:  #only show the circle at the string tail and head
                        circle_region1 = plt.Circle((x_Qt, y_Qt), Rc, facecolor='goldenrod', edgecolor='red', zorder=0,
                                                   alpha=1)
                        circle_region2 = plt.Circle((x_Qh, y_Qh), Rc, facecolor='goldenrod', edgecolor='blue', zorder=0,
                                                    alpha=1)
                        circle_region3 = plt.Circle((x_middle, y_middle), Rc, facecolor='goldenrod', edgecolor='green', zorder=0)

                        circle_region4 = plt.Circle((x_Qt, y_Qt), Rc, facecolor='none', edgecolor='red', zorder=10,
                                                    alpha=1, linestyle='--', linewidth=1.5)
                        circle_region5 = plt.Circle((x_Qh, y_Qh), Rc, facecolor='none', edgecolor='blue', zorder=10,
                                                    alpha=1, linestyle='--', linewidth=1.5)
                        circle_region6 = plt.Circle((x_middle, y_middle), Rc, facecolor='none', edgecolor='green',
                                                    zorder=10, linestyle='--', linewidth=1.5)

                        ax.add_patch(circle_region1)
                        ax.add_patch(circle_region2)
                        ax.add_patch(circle_region3)
                        ax.add_patch(circle_region4)
                        ax.add_patch(circle_region5)
                        ax.add_patch(circle_region6)
                    elif mode == 50:
                        x_all = glass.frames[t_si, glass.frames[t_si, :, 0] == 387, 2][0]
                        y_all = glass.frames[t_si, glass.frames[t_si, :, 0] == 387, 3][0]
                        plot_Rc((x_Qt, y_Qt), Rc, t_si, P_color='w', zorder=1, alpha=1)
                        circle_region1 = plt.Circle((x_Qt, y_Qt), Rc, facecolor='goldenrod', edgecolor='red', zorder=0,
                                                   alpha=1, lw=1.5)
                        plot_Rc((x_all, y_all), Rc, t_si, P_color='w', zorder=1, alpha=1)
                        circle_region2 = plt.Circle((x_all, y_all), Rc, facecolor='goldenrod', edgecolor='blue', zorder=0,
                                                    alpha=1, lw=1.5)
                        ax.add_patch(circle_region1)
                        ax.add_patch(circle_region2)
                        ax.annotate(
                            '',
                            xy=(x_all, y_all), xycoords='data',
                            xytext=(x_all+np.cos(np.pi/4)*Rc*1.1, y_all+np.sin(np.pi/4)*Rc*1.1), textcoords='data',
                            arrowprops=dict(arrowstyle='<-', connectionstyle='arc3', lw=1) #
                        )
                        ax.text(x_all+np.cos(np.pi/4)*Rc*0.9, y_all-np.sin(np.pi/4)*Rc*0.1, fr'$R$',
                                ha='center', va='bottom', fontsize=16, fontname='Times New Roman')

                    elif mode==6:
                        plot_Rc((x_Qt, y_Qt), Rc, t_si, P_color='yellow', zorder=1, alpha=1)
                        plot_Rc((x_Qh, y_Qh), Rc, t_si, P_color='yellow', zorder=1, alpha=1)
                        circle1 = plt.Circle((x_Qt, y_Qt), Rc, facecolor='none', edgecolor='red', zorder=5)
                        circle2 = plt.Circle((x_Qh, y_Qh), Rc, facecolor='none', edgecolor='red', zorder=5)
                        circle1.set_linestyle('--')
                        circle2.set_linestyle('--')
                        ax.add_patch(circle1)
                        ax.add_patch(circle2)

            # if showid:
            #     ax.text(glass.frames[starttime + 1, stringparticles[0], 2],
            #             glass.frames[starttime + 1, stringparticles[0], 3],
            #             f'r(0)#{IDlist[i]} ', ha="center", va="bottom")
            #     ax.text(glass.frames[endtime - 1, stringparticles[-1], 2],
            #             glass.frames[endtime - 1, stringparticles[-1], 3],
            #             f'r(t)#{IDlist[i]}', ha="center", va="bottom")
            if glass.n != glass.n0:
                ax.set(xlim=(glass.x_low, glass.x_up), ylim=(glass.y_low, glass.y_up), aspect="equal")
            else:
                ax.set(xlim=(0-5, glass.L[0]+5), ylim=(0-5, glass.L[1]+5), aspect="equal")
        if show_length_distribusion:
            return glass.stringlength
    return 0
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(family='Times New Roman', size=20)
font_path = '/home/xiaochu/.fonts/times.ttf'
fm.fontManager.addfont(font_path)
from hd2d_src.HopTrack.core.share.dr_overlap import dr_overlap
#Calculate local denstity around the given circle region and particle position
def distPBC2D(vec, Lx, Ly):
    hLx = Lx / 2
    hLy = Ly / 2
    if vec[0] > hLx:
        vec[0] = vec[0] - Lx
    elif vec[0] < -hLx:
        vec[0] = vec[0] + Lx

    if vec[1] > hLy:
        vec[1] = vec[1] - Ly
    elif vec[1] < -hLy:
        vec[1] = vec[1] + Ly

    return vec

def circle_intersection_area(x1, y1, r1, x2, y2, r2, lx, ly):
    # distance between circle center
    dr = np.array([x1, y1]) - np.array([x2, y2])
    dr_pbc = distPBC2D(dr, lx, ly)
    d = np.linalg.norm(dr_pbc)

    # To determine the positional relationship between two circles and calculate area
    if d > r1 + r2:
        return 0
    elif d < abs(r1 - r2):
        r = min(r1, r2)
        return math.pi * r ** 2
    else:
        l = (-r1 ** 2 + r2 ** 2 + d ** 2) / (2 * d)
        theta1 = math.acos(l / r2)
        theta2 = math.acos((d - l) / r1)
        s = math.sqrt(r2 ** 2 - l ** 2)
        return theta1 * (r2 ** 2) + theta2 * (r1 ** 2) - d * s



def Local_Density_Calculator(a, r1, stime):
    a.pbc()
    T, N, D = np.shape(a.frames)
    local_density = []
    for iT in stime:
        for iN in range(N):
            x1 = a.frames[iT, iN, 2]
            y1 = a.frames[iT, iN, 3]
            skin = max(a.radii)
            xlow = np.mod(x1 - r1 - skin, a.L[0])
            xup = np.mod(x1 + r1 + skin, a.L[0])
            ylow = np.mod(y1 - r1 - skin, a.L[1])
            yup = np.mod(y1 + r1 + skin, a.L[1])
            if xlow>xup:
                xregion = (a.frames[iT, :, 2] > xlow) | (a.frames[iT, :, 2] < xup)
            else:
                xregion = (a.frames[iT, :, 2] > xlow) & (a.frames[iT, :, 2] < xup)
            if ylow>yup:
                yregion = (a.frames[iT, :, 3] > ylow) | (a.frames[iT, :, 3] < yup)
            else:
                yregion = (a.frames[iT, :, 3] > ylow) & (a.frames[iT, :, 3] < yup)
            mask = xregion & yregion
            slctp = a.frames[iT, mask, :]
            s = 0
            for sj in slctp:
                x2 = sj[2]
                y2 = sj[3]
                r2 = a.radii[int(sj[0]-1)]
                s += circle_intersection_area(x1, y1, r1, x2, y2, r2, a.L[0], a.L[1])
            if s == 0:
                print('s==0')
            local_density.append(s / (math.pi * r1 ** 2))
    return local_density

def Local_Density_Calculator_Nooverlap(a, r1, stime):
    a.pbc()
    T, N, D = np.shape(a.frames)
    local_density = []
    for iT in stime:
        pick_list = [0]
        for iN in range(N):
            dr_min = dr_overlap(a, iT, iN, pick_list)
            if dr_min > 2*r1 or dr_min == 0:  # no overlap
                x1 = a.frames[iT, iN, 2]
                y1 = a.frames[iT, iN, 3]
                skin = max(a.radii)
                xlow = np.mod(x1 - r1 - skin, a.L[0])
                xup = np.mod(x1 + r1 + skin, a.L[0])
                ylow = np.mod(y1 - r1 - skin, a.L[1])
                yup = np.mod(y1 + r1 + skin, a.L[1])
                if xlow>xup:
                    xregion = (a.frames[iT, :, 2] > xlow) | (a.frames[iT, :, 2] < xup)
                else:
                    xregion = (a.frames[iT, :, 2] > xlow) & (a.frames[iT, :, 2] < xup)
                if ylow>yup:
                    yregion = (a.frames[iT, :, 3] > ylow) | (a.frames[iT, :, 3] < yup)
                else:
                    yregion = (a.frames[iT, :, 3] > ylow) & (a.frames[iT, :, 3] < yup)
                mask = xregion & yregion
                slctp = a.frames[iT, mask, :]
                s = 0
                for sj in slctp:
                    x2 = sj[2]
                    y2 = sj[3]
                    r2 = a.radii[int(sj[0]-1)]
                    s += circle_intersection_area(x1, y1, r1, x2, y2, r2, a.L[0], a.L[1])
                if s == 0:
                    print('s==0')
                local_density.append(s / (math.pi * r1 ** 2))
    return local_density


def ld_hopped(a, dt, r1, hop):
    a.pbc()
    T, N, D = np.shape(a.frames)
    local_density = []
    Tdt = T//dt
    for i in range(Tdt):
        iT0 = i*dt
        iT1 = (i+1)*dt-1
        for iN in range(N):
            x1 = a.frames[iT0, iN, 2]
            y1 = a.frames[iT0, iN, 3]
            dx = x1 - a.frames[iT1, iN, 2]
            dy = y1 - a.frames[iT1, iN, 3]
            [dx, dy] = distPBC2D([dx, dy], a.L[0], a.L[1])
            r = np.sqrt(dx**2 + dy**2)
            if r >= hop:
                skin = max(a.radii)
                xlow = np.mod(x1 - r1 - skin, a.L[0])
                xup = np.mod(x1 + r1 + skin, a.L[0])
                ylow = np.mod(y1 - r1 - skin, a.L[1])
                yup = np.mod(y1 + r1 + skin, a.L[1])
                if xlow > xup:
                    xregion = (a.frames[iT0, :, 2] > xlow) | (a.frames[iT0, :, 2] < xup)
                else:
                    xregion = (a.frames[iT0, :, 2] > xlow) & (a.frames[iT0, :, 2] < xup)
                if ylow > yup:
                    yregion = (a.frames[iT0, :, 3] > ylow) | (a.frames[iT0, :, 3] < yup)
                else:
                    yregion = (a.frames[iT0, :, 3] > ylow) & (a.frames[iT0, :, 3] < yup)
                mask = xregion & yregion
                slctp = a.frames[iT0, mask, :]
                s = 0
                for sj in slctp:
                    x2 = sj[2]
                    y2 = sj[3]
                    r2 = a.radii[int(sj[0] - 1)]
                    s += circle_intersection_area(x1, y1, r1, x2, y2, r2, a.L[0], a.L[1])
                if s == 0:
                    print('s==0')
                local_density.append(s / (math.pi * r1 ** 2))
    return local_density

def ld_hopped_no_overlap(a, dt, r1, hop):
    a.pbc()
    T, N, D = np.shape(a.frames)
    local_density = []
    Tdt = T//dt
    for i in range(Tdt):
        iT0 = i*dt
        iT1 = (i+1)*dt-1
        pick_list = []
        for iN in range(N):
            x1 = a.frames[iT0, iN, 2]
            y1 = a.frames[iT0, iN, 3]
            dx = x1 - a.frames[iT1, iN, 2]
            dy = y1 - a.frames[iT1, iN, 3]
            [dx, dy] = distPBC2D([dx, dy], a.L[0], a.L[1])
            r = np.sqrt(dx**2 + dy**2)
            if r >= hop:
                if len(pick_list) > 1:
                    dr_min = dr_overlap(a, iT0, iN, pick_list)
                else:
                    pick_list.append(iN)
                    dr_min = 0
                # no overlap dr > 2* r1
                if dr_min > 2*r1 or dr_min == 0:
                    skin = max(a.radii)
                    xlow = np.mod(x1 - r1 - skin, a.L[0])
                    xup = np.mod(x1 + r1 + skin, a.L[0])
                    ylow = np.mod(y1 - r1 - skin, a.L[1])
                    yup = np.mod(y1 + r1 + skin, a.L[1])
                    if xlow > xup:
                        xregion = (a.frames[iT0, :, 2] > xlow) | (a.frames[iT0, :, 2] < xup)
                    else:
                        xregion = (a.frames[iT0, :, 2] > xlow) & (a.frames[iT0, :, 2] < xup)
                    if ylow > yup:
                        yregion = (a.frames[iT0, :, 3] > ylow) | (a.frames[iT0, :, 3] < yup)
                    else:
                        yregion = (a.frames[iT0, :, 3] > ylow) & (a.frames[iT0, :, 3] < yup)
                    mask = xregion & yregion
                    slctp = a.frames[iT0, mask, :]
                    s = 0
                    for sj in slctp:
                        x2 = sj[2]
                        y2 = sj[3]
                        r2 = a.radii[int(sj[0] - 1)]
                        s += circle_intersection_area(x1, y1, r1, x2, y2, r2, a.L[0], a.L[1])
                    if s == 0:
                        print('s==0')
                    local_density.append(s / (math.pi * r1 ** 2))
    return local_density

def ld_unhopped(a, dt, r1, hop):
    a.pbc()
    T, N, D = np.shape(a.frames)
    local_density = []
    Tdt = T//dt
    for i in range(Tdt):
        iT0 = i*dt
        iT1 = (i+1)*dt-1
        for iN in range(N):
            x1 = a.frames[iT0, iN, 2]
            y1 = a.frames[iT0, iN, 3]
            dx = x1 - a.frames[iT1, iN, 2]
            dy = y1 - a.frames[iT1, iN, 3]
            [dx, dy] = distPBC2D([dx, dy], a.L[0], a.L[1])
            r = np.sqrt(dx**2 + dy**2)
            if r<hop:
                skin = max(a.radii)
                xlow = np.mod(x1 - r1 - skin, a.L[0])
                xup = np.mod(x1 + r1 + skin, a.L[0])
                ylow = np.mod(y1 - r1 - skin, a.L[1])
                yup = np.mod(y1 + r1 + skin, a.L[1])
                if xlow > xup:
                    xregion = (a.frames[iT0, :, 2] > xlow) | (a.frames[iT0, :, 2] < xup)
                else:
                    xregion = (a.frames[iT0, :, 2] > xlow) & (a.frames[iT0, :, 2] < xup)
                if ylow > yup:
                    yregion = (a.frames[iT0, :, 3] > ylow) | (a.frames[iT0, :, 3] < yup)
                else:
                    yregion = (a.frames[iT0, :, 3] > ylow) & (a.frames[iT0, :, 3] < yup)
                mask = xregion & yregion
                slctp = a.frames[iT0, mask, :]
                s = 0
                for sj in slctp:
                    x2 = sj[2]
                    y2 = sj[3]
                    r2 = a.radii[int(sj[0] - 1)]
                    s += circle_intersection_area(x1, y1, r1, x2, y2, r2, a.L[0], a.L[1])
                if s == 0:
                    print('s==0')
                local_density.append(s / (math.pi * r1 ** 2))
    return local_density

def LD_one_particle(a, xycood, Rc, partciles_in_region, Lxy):
    s = 0
    x1, y1 = xycood
    r1 = Rc
    lx, ly = Lxy
    for sj in partciles_in_region:
        x2 = sj[2]
        y2 = sj[3]
        r2 = a.radii[int(sj[0] - 1)]
        s += circle_intersection_area(x1, y1, r1, x2, y2, r2, lx, ly)
    if s == 0:
        print('error: s==0')
    return s/(np.pi * Rc ** 2)

def Local_Density_Calculator_Exclude_String_Ends(a, r1, stime):
    a.pbc()
    T, N, D = np.shape(a.frames)
    local_density = []
    for i, iT in enumerate(stime):
        tail_particle_id = [string[0] for string in a.connected_components]
        head_particle_id = [string[-1] for string in a.connected_components]
        exclude_list = tail_particle_id+head_particle_id
        list_all_string = set(exclude_list)
        for iN in range(N):
            if int(a.frames[iT, iN, 0] - 1) not in list_all_string:  # exclude ends of string
                x1 = a.frames[iT, iN, 2]
                y1 = a.frames[iT, iN, 3]
                skin = max(a.radii)
                xlow = np.mod(x1 - r1 - skin, a.L[0])
                xup = np.mod(x1 + r1 + skin, a.L[0])
                ylow = np.mod(y1 - r1 - skin, a.L[1])
                yup = np.mod(y1 + r1 + skin, a.L[1])
                if xlow>xup:
                    xregion = (a.frames[iT, :, 2] > xlow) | (a.frames[iT, :, 2] < xup)
                else:
                    xregion = (a.frames[iT, :, 2] > xlow) & (a.frames[iT, :, 2] < xup)
                if ylow>yup:
                    yregion = (a.frames[iT, :, 3] > ylow) | (a.frames[iT, :, 3] < yup)
                else:
                    yregion = (a.frames[iT, :, 3] > ylow) & (a.frames[iT, :, 3] < yup)
                mask = xregion & yregion
                slctp = a.frames[iT, mask, :]
                s = 0
                for sj in slctp:
                    x2 = sj[2]
                    y2 = sj[3]
                    r2 = a.radii[int(sj[0]-1)]
                    s += circle_intersection_area(x1, y1, r1, x2, y2, r2, a.L[0], a.L[1])
                if s == 0:
                    print('s==0')
                local_density.append(s / (math.pi * r1 ** 2))
    return local_density

def show_heterogeneity(ax, a, local_density, norm, plt_fr, plt_region, filled=True, colormap='jet', cmap='None',
                       colorthreshold=1, binarycolor=False, Dup=0.01, Ddown=-0.01, label='Local density'):
    _, N, _ = np.shape(a.frames)
    # cmap = plt.get_cmap('jet')
    if cmap != 'None':
        cmap = cmap
    else:
        cmap = plt.get_cmap(colormap)
    # r_all = a.radii
    x_low, x_up, y_low, y_up = plt_region
    for i in range(N):
        x0 = a.frames[plt_fr, i, 2]
        y0 = a.frames[plt_fr, i, 3]
        x=x0
        y=y0

        if x_low < a.x_low:
            if x > a.L[0]/2:
                x = x0- a.L[0]
        if y_low < a.y_low:
            if y > a.L[1]/2:
                y = y0 - a.L[1]
        if x_up > a.x_up:
            if x < a.L[0]/2:
                x = x0 + a.L[0]
        if y_up > a.y_up:
            if y < a.L[1]/2:
                y = y0 + a.L[1]

        if (x > x_low) & (x < x_up) & (y > y_low) & (y < y_up):
            if local_density[i] <= colorthreshold:
                itype = int(a.frames[0, i, 0]-1)
                r = a.radii[itype]
                if binarycolor:
                    if local_density[i] > Dup:
                        color = 'cyan'
                    elif local_density[i] > Ddown:
                        color = 'white'
                    else:
                        color = 'grey'
                    circle = plt.Circle((x, y), r, edgecolor='k', facecolor=color, fill=filled, zorder=0, linewidth=0,
                                        alpha=1)
                    ax.add_artist(circle)
                else:
                    color = cmap(norm(local_density[i]))
                    circle = plt.Circle((x, y), r, color=color, fill=True, zorder=0, linewidth=0,
                                        alpha=1)
                    ax.add_artist(circle)
                # ra, rb = ht.circle(x, y, r)
                # ax.plot(ra, rb, color=color, linewidth=0.5)

    scatter = ax.scatter(a.frames[0, :, 2], a.frames[0, :, 3], c=local_density, norm=norm, cmap=cmap, s=0)
    ax.set_aspect('equal', adjustable='box')
    # ticks = [0.765 + i*0.01 for i in range(9)]
    # ax.figure.colorbar(scatter, ax=ax, norm=norm, label=r'Local Density changes', ticks=ticks)
    # ax.figure.colorbar(scatter, ax=ax, norm=norm, label=r'Local Density changes', fontname='Times New Roman', fontsize=18)
    cbar = ax.figure.colorbar(scatter, ax=ax, norm=norm)
    cbar.set_label(label, fontsize=12, family='Times New Roman')




if __name__ == '__main__':
    with open("../a.obj", "rb") as f:
        a = pickle.load(f)
    t0 = 6
    t1 = 7
    nseg = 2
    r1 = 3.0
    a.setduration(t0, t1)
    a.chooseWithoutCoarsening(nseg)
    local_density = Local_Density_Calculator(a, r1, [i for i in range(nseg)])
    with open('local_density/dt5_Rh0.40_Rs_0.80_Rc_1.90/entire_density.obj', 'wb') as fe:
        pickle.dump(local_density, fe)

    fig1, ax1 = plt.subplots()
    edges = np.linspace(0.6, 1.0, 100)
    ax1.hist(local_density, bins=edges, label='The entire system', density=True, color='k')
    mu, std = norm.fit(local_density)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, mu, std)
    ax1.plot(x, p, 'r', linewidth=2, label=f'mu {mu:.04f}, std {std:.04f}')
    ax1.legend()
    fig1.tight_layout()
    # plt.savefig(f'{obj_path}/delt_rho.png', dpi=300)
    plt.show()

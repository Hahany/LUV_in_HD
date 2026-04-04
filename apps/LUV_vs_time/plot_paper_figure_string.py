import os.path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from hd2d_src.HopTrack.utils.CustomUnpickler import CustomUnpickler
from hd2d_src.HopTrack.core.Local_Density_Calculator import LD_one_particle
import hd2d_src.HopTrack.core.Local_Density_Calculator as LDC
from hd2d_src.HopTrack.core.Local_Density_Calculator import show_heterogeneity
from hd2d_src.HopTrack.utils.path import get_sub_project_root

def plot_tracj(b, start_time, end_time, color, region):
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    b.showdisp(ax, lw=1, ms=1.5, nodot=True, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False)
def plot_string(b, start_time, end_time, color, region, Rc, rh, rs, dr_ht, stlength, savepath, crange):
    fig, ax = plt.subplots()
    mean_radius = np.min(b.radii)
    # b.chbox(20, 40, 50, 60, update_bak=True)
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    # b.short_burst(1, 1, 10, 0)
    b.chooseWithoutCoarsening(2)
    b.findstring(HopThreshold=rh*2*mean_radius, ConnectThreshold=rs*2*mean_radius,
                 ignoreLoop=True, stlength=stlength, dr_ht=dr_ht*2*mean_radius, hop_whole=True)
    b.showstring(ax=ax, showid=False, showtraj=False, stringID=[], SSC=color,
                 size=10, WL=True, show_localdensity_region=True, Rc=Rc, mode=5, findquasivoid=True, nodot=True)
    # plot trajectory
    b.setduration(start_time, end_time)
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time-start_time), lw=2, ms=1.5, nodot=True, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False, overlap=3, crange=crange)
    plt.xticks([])
    plt.yticks([])
    plt.box(on=True)
    fig.savefig(f"{savepath}", dpi=600)





    # ax.set_xlim([b.frames[0, b.connected_components[0][-1], 2] - Rc - 1,
    #              b.frames[0, b.connected_components[0][-1], 2] + Rc + 1])
    # ax.set_ylim([b.frames[0, b.connected_components[0][-1], 3] - Rc - 1,
    #              b.frames[0, b.connected_components[0][-1], 3] + Rc + 1])
    # fig.savefig(f"{savepath}_head.svg", dpi=600)

def plot_string_tail_config(b, Rc, start_time, end_time, color, region):
    fig, ax=plt.subplots()
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    # b.short_burst(1, 1, 10, 0)
    b.chooseWithoutCoarsening(2)
    b.findstring(HopThreshold=0.5, ConnectThreshold=0.8, ignoreLoop=True, stlength=0, dr_ht=0, hop_whole=True)
    b.showstring(ax=ax, nodot=True, findquasivoid=True, showid=False, showtraj=False, stringID=[], SSC=color,
                 size=10, WL=True, show_localdensity_region=True, Rc = Rc, mode=2)
    b.setduration(start_time, end_time)
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time-start_time), lw=2, ms=2.5, nodot=True, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False)
    plt.xticks([])
    plt.yticks([])

def plot_string_test(b, start_time, end_time, color, rs, rh, Lst, disp_head_tail):
    # b.chbox(40, 55, 40, 55, update_bak=True)
    b.setduration(start_time, end_time)
    b.short_burst(1, 5, 100, 0)
    mean_radius = np.min(b.radii)
    rh = mean_radius * rh * 2
    rs = mean_radius * rs * 2
    b.findstring(HopThreshold=rh, ConnectThreshold=rs, ignoreLoop=True, stlength=Lst, dr_ht=disp_head_tail, hop_whole=False)
    b.showstring(ax=ax, findquasivoid=True, showid=False, stringID=[], SSC=color, size=2, WL=False, show_localdensity_region=True, Rc = 1.72, mode=2)
    # b.showdisp(ax, lw=1, ms=1.5, nodot=True, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
    #            colorbar=False, showradii=False)

def plot_regional_local_density(b, Rc, start_time, end_time, color, region, rh, rs, savepath):
    fig, ax = plt.subplots()
    x1, x2, y1, y2 = region
    norm=mcolors.Normalize(vmin=-0.03, vmax=0.03)
    b.setduration(start_time, end_time)
    # b.short_burst(1, 2, 10, 0)
    b.chooseWithoutCoarsening(2)
    # b.findstring(HopThreshold=0.4, ConnectThreshold=0.8, ignoreLoop=True, stlength=0, dr_ht=0, hop_whole=False)
    # b.showdensity(Rc, 0, 0.805, savefile=None, figshow=True,
    #                 pid=None, select=False, mode=0, Length_in_string=3, crt_in=False)
    entire_density0 = Local_Density_Calculator(b, Rc, [0])
    entire_density1 = Local_Density_Calculator(b, Rc, [-1])
    delta_ld = np.array(entire_density1) - np.array(entire_density0)
    show_heterogeneity(ax, b, delta_ld, norm, [0], [0, 64, 0, 64], filled=True, colormap='RdBu',
                           binarycolor=False, label=r'$\Delta V$')
    b.chbox(x1, x2, y1, y2, update_bak=False)
    b.findstring(HopThreshold=rh, ConnectThreshold=rs, ignoreLoop=True, stlength=0, dr_ht=0, hop_whole=False)
    b.showstring(ax=ax, findquasivoid=True, showid=False, stringID=[], SSC=color, size=2, WL=True,showtraj=False,
                 show_localdensity_region=True, Rc=Rc, mode=5, nodot=True)
    b.setduration(start_time, end_time)
    b.chbox(x1, x2, y1, y2, update_bak=False)
    b.showdisp(ax, lw=1, ms=1.5, nodot=False, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    fig.savefig(f"{savepath}", dpi=600)

def plot_regional_volume(b, Rc, start_time, end_time, color, region, rh, rs, savepath):
    fig, ax = plt.subplots()
    x1, x2, y1, y2 = region
    v1 = (Rc * 2) ** 2 * (- 0.03)
    v2 = (Rc * 2) ** 2 * ( 0.03)
    norm=mcolors.Normalize(vmin=v1, vmax=v2)
    b.setduration(start_time, end_time)
    # b.short_burst(1, 2, 10, 0)
    b.chooseWithoutCoarsening(2)
    # b.findstring(HopThreshold=0.4, ConnectThreshold=0.8, ignoreLoop=True, stlength=0, dr_ht=0, hop_whole=False)
    # b.showdensity(Rc, 0, 0.805, savefile=None, figshow=True,
    #                 pid=None, select=False, mode=0, Length_in_string=3, crt_in=False)
    entire_density0 = Local_Density_Calculator(b, Rc, [0])
    entire_density1 = Local_Density_Calculator(b, Rc, [-1])
    delta_ld = np.array(entire_density1) - np.array(entire_density0)
    delta_ld = - delta_ld * ((Rc * 2) ** 2)
    show_heterogeneity(ax, b, delta_ld, norm, [0], [0, b.L[0], 0, b.L[1]], filled=True, colormap='RdBu_r',
                           binarycolor=False, label=r'$\Delta V / v_1$')
    b.findstring(HopThreshold=rh, ConnectThreshold=rs, ignoreLoop=True, stlength=0, dr_ht=2*Rc, hop_whole=False)
    b.setduration(start_time, end_time)
    # b.chbox(x1, x2, y1, y2, update_bak=False)
    b.showstring(ax=ax, findquasivoid=True, showid=False, stringID=[], SSC=color, size=2, WL=True,showtraj=True,
                 show_localdensity_region=True, Rc=Rc, mode=0, nodot=True)
    b.setduration(start_time, end_time)
    b.chbox(x1, x2, y1, y2, update_bak=False)
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time-start_time), lw=1, ms=1.5, nodot=False, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlim([region[0], region[1]])
    ax.set_ylim([region[2], region[3]])
    fig.savefig(f"{savepath}", dpi=300)

def x_ld(b, Rc, save_file_name, fix_pos=False):
    def get_region(xycood):
        x1, y1 = xycood
        r1 = Rc
        skin = max(b.radii)
        xlow = x1 - r1 - skin
        xup = x1 + r1 + skin
        ylow = y1 - r1 - skin
        yup = y1 + r1 + skin
        if xlow > xup:
            xregion = (b.frames[t, :, 2] > xlow) | (b.frames[t, :, 2] < xup)
        else:
            xregion = (b.frames[t, :, 2] > xlow) & (b.frames[t, :, 2] < xup)
        if ylow > yup:
            yregion = (b.frames[t, :, 3] > ylow) | (b.frames[t, :, 3] < yup)
        else:
            yregion = (b.frames[t, :, 3] > ylow) & (b.frames[t, :, 3] < yup)
        mask = xregion & yregion
        particles_in_region = b.frames[t, mask, :]
        return particles_in_region



    T, N, M = b.frames.shape
    fig, ax = plt.subplots()
    time_list = [i for i in range(T)]
    for si, string in enumerate(b.connected_components):
        tail_id = int(string[0])
        head_id = int(string[-1])
        middle_id = int(string[int(len(string)/2-1)])
        ld_tail_list = []
        ld_head_list = []
        ld_middle_list = []
        for t in range(T):
            if fix_pos:
                xycood_tail = b.frames[0, tail_id, 2:4]
                xycood_head = b.frames[0, head_id, 2:4]
                xycood_middle = b.frames[0, middle_id, 2:4]
            else:
                xycood_tail = b.frames[t, tail_id, 2:4]
                xycood_head = b.frames[t, head_id, 2:4]
                xycood_middle = b.frames[t, middle_id, 2:4]
            particles_in_region_tail = get_region(xycood_tail)
            particles_in_region_head = get_region(xycood_head)
            particles_in_region_middle = get_region(xycood_middle)
            ld_tail = LD_one_particle(b, xycood_tail, Rc, particles_in_region_tail, [64, 64])
            ld_head = LD_one_particle(b, xycood_head, Rc, particles_in_region_head, [64, 64])
            ld_middle = LD_one_particle(b, xycood_middle, Rc, particles_in_region_middle, [64, 64])
            ld_tail_list.append(ld_tail)
            ld_head_list.append(ld_head)
            ld_middle_list.append(ld_middle)

        plt.plot(time_list, ld_head_list, 's-', label=f'Head ', color='b')
        plt.plot(time_list, ld_middle_list, 'o-', label=f'Middle ', color='g')
        plt.plot(time_list, ld_tail_list, 'd-', label=f'Tail', color='r')
        plt.legend(loc='center', prop={'family': 'Times New Roman', 'size': 18}, frameon=False, bbox_to_anchor=(0.86, 0.12))
        plt.ylabel(r'$\rho$', fontsize=18, fontname='Times New Roman')
        plt.xlabel(r'Time', fontsize=18, fontname='Times New Roman')
        plt.xticks(fontsize=18, fontname='Times New Roman')
        plt.yticks(fontsize=18, fontname='Times New Roman')
        # plt.ylim(0.755, 0.805)
        plt.grid()
        plt.tight_layout()
    fig.savefig(f"{save_file_name}", dpi=600)
    plt.show()

def x_V(b, Rc, save_file_name, L_data1, L_data2, fix_pos=False):
    def get_region(xycood):
        x1, y1 = xycood
        r1 = Rc
        skin = max(b.radii)
        xlow = x1 - r1 - skin
        xup = x1 + r1 + skin
        ylow = y1 - r1 - skin
        yup = y1 + r1 + skin
        if xlow > xup:
            xregion = (b.frames[t, :, 2] > xlow) | (b.frames[t, :, 2] < xup)
        else:
            xregion = (b.frames[t, :, 2] > xlow) & (b.frames[t, :, 2] < xup)
        if ylow > yup:
            yregion = (b.frames[t, :, 3] > ylow) | (b.frames[t, :, 3] < yup)
        else:
            yregion = (b.frames[t, :, 3] > ylow) & (b.frames[t, :, 3] < yup)
        mask = xregion & yregion
        particles_in_region = b.frames[t, mask, :]
        return particles_in_region



    T, N, M = b.frames.shape
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for si, string in enumerate(b.connected_components):
        tail_id = int(string[0])
        head_id = int(string[-1])
        middle_id = int(string[int(len(string)/2-2)])
        ld_tail_list = []
        ld_head_list = []
        ld_middle_list = []
        for t in range(T):
            if fix_pos:
                xycood_tail = b.frames[0, tail_id, 2:4]
                xycood_head = b.frames[0, head_id, 2:4]
                xycood_middle = b.frames[0, middle_id, 2:4]
            else:
                xycood_tail = b.frames[t, tail_id, 2:4]
                xycood_head = b.frames[t, head_id, 2:4]
                xycood_middle = b.frames[t, middle_id, 2:4]
            particles_in_region_tail = get_region(xycood_tail)
            particles_in_region_head = get_region(xycood_head)
            particles_in_region_middle = get_region(xycood_middle)
            ld_tail = LD_one_particle(b, xycood_tail, Rc, particles_in_region_tail, b.L)
            ld_head = LD_one_particle(b, xycood_head, Rc, particles_in_region_head, b.L)
            ld_middle = LD_one_particle(b, xycood_middle, Rc, particles_in_region_middle, b.L)
            ld_tail_list.append(ld_tail)
            ld_head_list.append(ld_head)
            ld_middle_list.append(ld_middle)
        ld_head_v = (Rc*2)**2*(1-np.array(ld_head_list))[0:-1:2]
        ld_middle_v = (Rc * 2) ** 2 * (1 - np.array(ld_middle_list))[0:-1:2]
        ld_tail_v = (Rc * 2) ** 2 * (1 - np.array(ld_tail_list))[0:-1:2]
        time_list = [i*41*2 for i in range(len(ld_head_v))]
        L_data = len(ld_head_v)
        lh_1 = 25
        lh_2 = 26
        lt_1 = 21
        lt_2 = 22
        lm_1 = 21
        lm_2 = 22
        lm_3 = 25
        lm_4 = 26
        mean_ld_tail_v1 = np.mean(ld_tail_v[0:lt_1])
        mean_ld_head_v1 = np.mean(ld_head_v[0:lh_1])
        mean_ld_middle_v1 = np.mean(ld_middle_v[0:lm_1])
        mean_ld_tail_v2 = np.mean(ld_tail_v[lt_2:-1])
        mean_ld_head_v2 = np.mean(ld_head_v[lh_2:-1])
        mean_ld_middle_v2 = np.mean(ld_middle_v[lm_4:-1])
        mean_ld_middle_v3 = np.mean(ld_middle_v[lm_2:lm_3])
        plt.hlines(mean_ld_tail_v1, 0, lt_1*82, 'r', linestyle='-', linewidth=2)
        plt.hlines(mean_ld_head_v1, 0, lh_1*82, 'b', linestyle='-', linewidth=2)
        plt.hlines(mean_ld_middle_v1, 0, lm_1*82, 'g', linestyle='-', linewidth=2)
        plt.hlines(mean_ld_tail_v2-0.02, lt_2*82, L_data*82, 'r', linestyle='-', linewidth=2)
        plt.hlines(mean_ld_head_v2+0.02, lh_2*82, L_data*82, 'b', linestyle='-', linewidth=2)
        plt.hlines(mean_ld_middle_v2, lm_4*82, L_data*82, 'g', linestyle='-', linewidth=2)
        plt.hlines(mean_ld_middle_v3, lm_2*82, lm_3*82, 'g', linestyle='-', linewidth=2)
        # plt.text(int(L_data1/4), np.max(ld_tail_v[0:L_data1]), fr'$\bar{{V}}/v_1 = {mean_ld_tail_v1:.02f} $', fontsize=14, fontname='Times New Roman')
        # plt.text(int(L_data1/4), np.max(ld_head_v[0:L_data1]), fr'$\bar{{V}}/v_1 = {mean_ld_head_v1:.02f} $', fontsize=14,
        #          fontname='Times New Roman')
        # plt.text(int(L_data1/4), np.min(ld_middle_v[0:L_data1])-0.15, fr'$\bar{{V}}/v_1 = {mean_ld_middle_v1:.02f} $', fontsize=14,
        #          fontname='Times New Roman')
        # plt.text(int(L_data2/3+L_data1-4), np.max(ld_tail_v[L_data2:L_data])-0.1, fr'$\bar{{V}}/v_1 = {mean_ld_tail_v2:.02f} $', fontsize=14, fontname='Times New Roman')
        # plt.text(int(L_data2/3+L_data1-4), np.max(ld_head_v[L_data2:L_data])-0.1, fr'$\bar{{V}}/v_1 = {mean_ld_head_v2:.02f} $', fontsize=14,
        #          fontname='Times New Roman')
        # plt.text(int(L_data2/3+L_data1-4), np.min(ld_middle_v[L_data2:L_data])-0.15, fr'$\bar{{V}}/v_1 = {mean_ld_middle_v2:.02f} $', fontsize=14,
        #          fontname='Times New Roman')
        # plt.text(int(L_data2 / 2), mean_ld_middle_v3,
        #          fr'$\bar{{V}}/v_1 = {mean_ld_middle_v3:.02f} $', fontsize=14,
        #          fontname='Times New Roman')

        plt.plot(time_list, ld_head_v, marker='s', linestyle='--', label=fr'$\mathrm{{Head}}$', color='b', markerfacecolor='none', markeredgecolor='b')
        plt.plot(time_list, ld_middle_v, marker='o', linestyle='--', label=fr'$\mathrm{{Middle}}$', color='g', markerfacecolor='none')
        plt.plot(time_list, ld_tail_v, marker='d', linestyle='--', label=fr'$\mathrm{{Tail}}$', color='r', markerfacecolor='none')
        plt.legend(ncol=3, loc='upper center', prop={'family': 'Times New Roman', 'size': 18}, frameon=False)#, bbox_to_anchor=(0, 1))
        plt.ylabel(r'$V/v$', fontsize=18, fontname='Times New Roman')
        plt.xlabel(r'$t$', fontsize=18, fontname='Times New Roman')
        plt.xticks(fontsize=18, fontname='Times New Roman')
        plt.yticks(fontsize=18, fontname='Times New Roman')
        plt.ylim(np.min(ld_middle_v)-0.1, np.max(ld_tail_v)+0.5)
        # plt.grid()

    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(f"{save_file_name}", dpi=600)
    plt.show()


def ld_map(ax, b, Rc, start_time, end_time, savepath):
    norm = mcolors.Normalize(vmin=-0.03, vmax=+0.03)
    Dup = 0.02
    Ddown = -0.02
    b.setduration(start_time, end_time)
    entire_density0 = LDC.Local_Density_Calculator(b, Rc, [0])
    entire_density1 = LDC.Local_Density_Calculator(b, Rc, [-1])
    delta_ld = np.array(entire_density1) - np.array(entire_density0)
    LDC.show_heterogeneity(ax, b, delta_ld, norm, [0], [0,64,0,64], filled=True, colormap='RdBu', binarycolor=False, Dup=Dup, Ddown=Ddown)
    b.unwrap3()
    b.coarsening(20, False)
    b.showdisp(ax, lw=0.5, ms=0.5, nodot=True, showvoid=False,
               showpid=False, showforcepid=0, showforcedid=0, colorbar=False, radii=None, showradii=False)
    ax.set_xlim([0, b.L[0]])
    ax.set_ylim([0, b.L[1]])
    plt.xticks([])
    plt.yticks([])
    # ax.set_title(fr'grey for $\Delta \rho < {Ddown}$, white for $\Delta \rho \in [{Ddown}, {Dup}]$  and cyan for $\Delta \rho >{Dup}$')
    plt.savefig(f'{savepath}/0805_LD_Map_Dld-{start_time:05d}.png', dpi=600)
    print(f'draw ....')




if __name__ == "__main__":
    proj_path = str(get_sub_project_root())
    obj_path = str(get_sub_project_root() / 'data/input/0805/a_AVE_normalized.obj')
    # b.cal_area_of_fragments = ht.Particles.cal_area_of_fragments.__get__(b)
    rh = 0.8
    rs = 0.6
    dr_ht = 2
    stlength = 0
    start_time = 69500   #62448-10 #65760
    end_time = start_time+100 #32 #14
    if not os.path.exists(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj'):
        # initialize the data reading, this will save time to reload the data
        with open(obj_path, "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            b = custom_unpickler.load()
        b.setduration(start_time, end_time)
        b.frames0 = []
        with open(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj', 'wb') as f:
            pickle.dump(b, f)
        exit(2)
    else:
        with open(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj', "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            b = custom_unpickler.load()
        b.frames=b.frames[0:-1:2, :, :]
        start_time = 0#40  # 62448-10 #65760
        end_time = start_time + 100 # 20  # 32 #14
        start_time = 20
        end_time = start_time + 11
    mean_radius = np.min(b.radii)
    print(mean_radius)
    Rc = 6*mean_radius
    # b.shiftbox(0, 20)
    # b.pbc()
    # b.unwrap3()
    region = [23, 47, 55, 75] #[12, 25, 10, 23] #[15, 28, 20, 33]
    current_path = os.path.abspath(__file__)
    save_dir = os.path.dirname(current_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_string(b, start_time, end_time, 'm', region, Rc, rh, rs, dr_ht, stlength, savepath=f'{save_dir}/Fig1a.png', crange=[0, end_time-start_time-1])


    #Temporary plot for V head middle tail
    start_time = 69500
    with open(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj', "rb") as f:
        custom_unpickler = CustomUnpickler(f)
        b = custom_unpickler.load()
    start_time = 0#40  # 62448-10 #65760
    end_time = start_time + 100 # 20  # 32 #14
    plot_string(b, start_time, end_time, 'm', region, Rc, rh, rs, dr_ht, stlength, savepath=f'{save_dir}/Fig1b_traj.png', crange=[0, end_time-start_time-1])
    x_V(b, Rc, f"{save_dir}/Fig1b.png", L_data1=22, L_data2=23, fix_pos=False)






    # plot_string_tail_config(b, Rc, start_time, end_time, 'm', region)
    # plot_string_test(b, 0, 1000, None, 0.6, 0.8, 10, 5)
    # with open("a_cut_69500.obj", "rb") as f:
    #     b = pickle.load(f)
    # plot_regional_local_density(b, Rc, start_time, end_time, 'r', region, rh, rs, savepath=f'../figure/F{start_time}-c.png')
    # plot_regional_volume(b, Rc, start_time, end_time, 'r', region, rh, rs, savepath=f'../figure/F{start_time}-c.png')

# #push-off method
#     with open("../data/input/0805/a_cut_69500.obj", "rb") as f:
#         custom_unpickler = CustomUnpickler(f)
#         b = custom_unpickler.load()
#     quasivoid_structure_tail(b, start_time, end_time, 'm', region, Rc, rh, rs, dr_ht, stlength,
#                              savepath=f'{save_dir}/F{start_time}-d')
#
#     with open("../data/input/0805/a_cut_69500.obj", "rb") as f:
#         custom_unpickler = CustomUnpickler(f)
#         b = custom_unpickler.load()
#     quasivoid_structure_head(b, start_time, end_time, 'm', region, Rc, rh, rs, dr_ht, stlength,
#                              savepath=f'{save_dir}/F{start_time}-d')
#     # ld_map(ax, b, Rc, start_time, end_time, 'LDmap')
#
#     end_time = start_time + 11
#     region = [0, 40, 0, 40]
#     with open("../data/input/0805/L64/a_cut_65345.obj", "rb") as f:
#         custom_unpickler = CustomUnpickler(f)
#         b = custom_unpickler.load()
#     quasivoid_structure_tail(b, start_time, end_time, 'm', region, Rc, rh, rs, dr_ht, stlength,
#                              savepath=f'{save_dir}/L64-F{start_time}-d')
#
#     with open("../data/input/0805/L64/a_cut_65345.obj", "rb") as f:
#         custom_unpickler = CustomUnpickler(f)
#         b = custom_unpickler.load()
#     quasivoid_structure_head(b, start_time, end_time, 'm', region, Rc, rh, rs, dr_ht, stlength,
#                              savepath=f'{save_dir}/L64-F{start_time}-d')
#
    plt.show()


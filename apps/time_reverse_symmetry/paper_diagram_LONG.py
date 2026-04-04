import numpy as np
import matplotlib.pyplot as plt
from hd2d_src.HopTrack.core import *
import glob
import scipy.stats as stats
from scipy.stats import probplot
from scipy.stats import norm as nm
import matplotlib.font_manager as fm
import multiprocessing
import os
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 28,
    'axes.labelsize': 30,
    'axes.titlesize': 34,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 28,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'legend.frameon': False
})
from hd2d_src.HopTrack.utils.path import get_sub_project_root
project_root = get_sub_project_root()


from matplotlib.font_manager import FontProperties
font_prop = FontProperties(family='Times New Roman')
font_path = '/home/xiaochu/.fonts/times.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'

def Q_Q_plot(data, Rc, save_fig):

    for i,d in enumerate(data):
        # Q-Q plot
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        mu = np.mean(d)
        std = np.std(d)
        custom_dist = nm(loc=mu, scale=std)
        probplot(d, dist=custom_dist, plot=ax3, fit=False)


        lines = ax3.get_lines()
        lines[0].set_color('r')
        lines[0].set_marker('*')
        lines[0].set_markersize(3)
        lines[1].set_color('g')
        lines[1].set_marker('^')
        lines[1].set_markersize(3)
        # lines[1].set_color('b')
        # lines[1].set_linestyle('-')
        lines[2].set_color('b')
        lines[2].set_marker('s')
        lines[2].set_markerfacecolor('none')
        lines[2].set_markersize(3)
        lines[3].set_color('k')
        lines[3].set_linestyle('-')
        lines[4].set_marker('^')
        lines[4].set_color('k')
        lines[4].set_linestyle('-')
        # lines[6].set_marker('>')
        # lines[6].set_color('g')
        # lines[6].set_linestyle('-')
        # lines[7].set_color('g')
        # lines[7].set_linestyle('-')
        ax3.set_title(f'$R_c={Rc:.02f}$', fontsize=24, fontname='Times New Roman')
        ax3.legend(['string ends','head', 'Entire system', 'Internal particles', 'Gaussian fit'], loc='upper left', fontsize=14, prop=font_prop)
        # fig.text(0.01, 0.5, 'Relative frequency', ha='center', va='center', rotation='vertical', fontsize=24)
        ax3.tick_params(axis='x', labelsize=18)
        ax3.tick_params(axis='y', labelsize=18)
        ax3.xaxis.label.set_size(24)
        ax3.yaxis.label.set_size(24)
        ax3.xaxis.label.set_fontname('Times New Roman')
        ax3.yaxis.label.set_fontname('Times New Roman')
        ax3.grid()
        fig3.savefig(f'Rc_{Rc}_{i}.png', dpi=300)

def plot_rc(rc, rs, rho, save_LD, data_path, scale=True):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    save_flag = f"Rs_{rs:.02f}"
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')
    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')
    ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string.txt')

    shiftwidth = 2.0
    if local_density.ndim == 1:
        x1_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_a, x1_hp, x1_t]
            x2 = [x2_a, x2_hp, x2_t]
            local_tail_ini = np.array(1 - local_density[0]) * (Rc * 2) ** 2
            local_tail_final = np.array(1 - local_density[1]) * (Rc * 2) ** 2
            local_head_ini = np.array(1 - local_density[2]) * (Rc * 2) ** 2
            local_head_final = np.array(1 - local_density[3]) * (Rc * 2) ** 2
            entire_density = np.array(1 - entire_density) * (Rc * 2) ** 2
            ld_instring = np.array(1 - ld_instring) * (Rc * 2) ** 2
            ld_hopped = np.array(1 - ld_hopped) * (Rc * 2) ** 2
            ax.set_xlabel(fr'Unoccupied Volume(V/ $v_1$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"

        tail = []
        tail.append(local_tail_ini)
        tail.append(local_head_final)

        head = []
        head.append(local_tail_final)
        head.append(local_head_ini)

    else:

        x1_t = (1 - np.mean(local_density[:, [0, 3]]))*(Rc*2)**2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[:, [0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_a, x1_hp, x1_t]
            x2 = [x2_a, x2_hp, x2_t]
            local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc*2)**2
            local_tail_final = np.array(1 - local_density[:, 1]) * (Rc*2)**2
            local_head_ini = np.array(1- local_density[:, 2]) * (Rc*2)**2
            local_head_final = np.array(1- local_density[:, 3]) * (Rc*2)**2
            entire_density = np.array(1 - entire_density) * (Rc*2)**2
            ld_instring = np.array(1 - ld_instring) * (Rc*2)**2
            ld_hopped = np.array(1 - ld_hopped) * (Rc*2)**2
            ax.set_xlabel(fr'$Local\  Free\ Volume(V/ v_1)$', fontsize =24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"


        tail = []
        tail.extend(local_tail_ini)
        tail.extend(local_head_final)

        head = []
        head.extend(local_tail_final)
        head.extend(local_head_ini)

    colors = ['r', 'g', 'b', 'k']
    color1, color2, color3, color4 = colors
    # ax[i].set_ylim([0, 100])
    # n, ave, mu, std = LDD.plot_norm_distribution(ax, [tail, entire_density, ld_instring, entire_system_without_string_ends],
    #                                              lable=[f'String Ends', 'Entire system', 'LD in string', 'LD exclude string ends'],
    #                            color=[color1, color2, 'k', 'g'], x1=x1, x2=x2)
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                 [entire_density,  ld_hopped, tail],
                                                         [40,10,10],
                                                 lable=[r'$All$', r'$Hopped$', r'$Tail$'],
                                                 color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')
    y_position=height
    ax.annotate(
        '',
        xy=(ave[2], y_position), xycoords='data',
        xytext=(ave[0], y_position), textcoords='data',
        arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    )
    ax.text(ave[1] + 0.3, y_position + 0.04, fr'$\Delta V / v_1={ave[2]-ave[0]:.02f}$', ha='center', va='bottom', fontsize=17, fontname='Times New Roman')
    avelist.append(abs(ave[0]-ave[2]))
    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.legend(loc='best',prop={'family': 'Times New Roman', 'size': 18}, frameon=False)
    ax.set_ylabel(r"Probability Density Function", fontsize=24, fontname='Times New Roman')
    # ax.grid()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    std_string.append(std[0])
    std_system.append(std[1])
    if scale:
        flag = "sumarea"
    else:
        flag = "ld"
    save_path = f'{save_LD}/LD_Distribution'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png', dpi=300)
    print(f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png')
    plt.show()



    # plt.show()

def plot_6sets(rc, rs, rho, save_LD, data_path, scale=True):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    save_flag = f"Rs_{rs:.02f}"
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')
    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')
    ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string.txt')
    ld_unhopped = np.loadtxt(f'{obj_path}/ld_unhopped.txt')

    shiftwidth = 2.0
    if local_density.ndim == 1:
        x1_unhp = (1 - np.mean(ld_unhopped)) * (Rc * 2) ** 2 - shiftwidth
        x1_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_unhp = (1 - np.mean(ld_unhopped)) * (Rc * 2) ** 2 + shiftwidth
        x2_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_unhp, x1_a, x1_hp, x1_m, x1_hd, x1_t]
            x2 = [x2_unhp, x2_a, x2_hp, x2_m, x2_hd, x2_t]
            local_tail_ini = np.array(1 - local_density[0]) * (Rc * 2) ** 2
            local_tail_final = np.array(1 - local_density[1]) * (Rc * 2) ** 2
            local_head_ini = np.array(1 - local_density[2]) * (Rc * 2) ** 2
            local_head_final = np.array(1 - local_density[3]) * (Rc * 2) ** 2
            entire_density = np.array(1 - entire_density) * (Rc * 2) ** 2
            ld_instring = np.array(1 - ld_instring) * (Rc * 2) ** 2
            ld_hopped = np.array(1 - ld_hopped) * (Rc * 2) ** 2
            ld_unhopped = np.array(1 - ld_unhopped) * (Rc * 2) ** 2
            ax.set_xlabel(fr'$Unoccupied\ Volume(V/ v_1)$', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"

        tail = []
        tail.append(local_tail_ini)
        tail.append(local_head_final)

        head = []
        head.append(local_tail_final)
        head.append(local_head_ini)

    else:
        x1_unhp = (1 - np.mean(ld_unhopped)) * (Rc * 2) ** 2 - shiftwidth
        x1_t = (1 - np.mean(local_density[:, [0, 3]]))*(Rc*2)**2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_unhp = (1 - np.mean(ld_unhopped)) * (Rc * 2) ** 2 + shiftwidth
        x2_t = (1 - np.mean(local_density[:, [0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_unhp, x1_a, x1_hp, x1_m, x1_hd, x1_t]
            x2 = [x2_unhp, x2_a, x2_hp, x2_m, x2_hd, x2_t]
            local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc*2)**2
            local_tail_final = np.array(1 - local_density[:, 1]) * (Rc*2)**2
            local_head_ini = np.array(1- local_density[:, 2]) * (Rc*2)**2
            local_head_final = np.array(1- local_density[:, 3]) * (Rc*2)**2
            entire_density = np.array(1 - entire_density) * (Rc*2)**2
            ld_instring = np.array(1 - ld_instring) * (Rc*2)**2
            ld_hopped = np.array(1 - ld_hopped) * (Rc*2)**2
            ld_unhopped = np.array(1 - ld_unhopped) * (Rc * 2) ** 2
            ax.set_xlabel(fr'$Local\  Free\ Volume(V/ v_1)$', fontsize =24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"


        tail = []
        tail.extend(local_tail_ini)
        tail.extend(local_head_final)

        head = []
        head.extend(local_tail_final)
        head.extend(local_head_ini)

    colors = ['r', 'g', 'b', 'k','m', 'c']
    # ax[i].set_ylim([0, 100])
    # n, ave, mu, std = LDD.plot_norm_distribution(ax, [tail, entire_density, ld_instring, entire_system_without_string_ends],
    #                                              lable=[f'String Ends', 'Entire system', 'LD in string', 'LD exclude string ends'],
    #                            color=[color1, color2, 'k', 'g'], x1=x1, x2=x2)
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                 [ld_unhopped,  entire_density,  ld_hopped, ld_instring, head, tail],
                                                         [20, 40, 20, 10, 10, 10],
                                                 lable=[r'$Unhopped$' ,r'$All$', r'$Hopped$', r'$Middle$', r'$Head$', r'$Tail$'],
                                                 color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')
    y_position=height
    ax.annotate(
        '',
        xy=(ave[-1], y_position), xycoords='data',
        xytext=(ave[0], y_position), textcoords='data',
        arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    )
    ax.text(ave[1] + 0.3, y_position + 0.04, fr'$\Delta V / v_1={ave[-1]-ave[0]:.02f}$', ha='center', va='bottom', fontsize=17, fontname='Times New Roman')
    avelist.append(abs(ave[0]-ave[-1]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.legend(loc='best',prop={'family': 'Times New Roman', 'size': 18}, frameon=False)
    ax.set_ylabel(r"Probability Density Function", fontsize=24, fontname='Times New Roman')
    # ax.grid()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    std_string.append(std[0])
    std_system.append(std[1])
    if scale:
        flag = "sumarea"
    else:
        flag = "ld"
    save_path = f'{save_LD}/LD_Distribution'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png', dpi=300)
    print(f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png')
    plt.show()



    # plt.show()

def plot_rc_differential(rc, scale=True):
    ## plot one best sample to show the distribution
    Rc = rc
    save_LD = "local_density_rs_0.6_rh_0.8_Lst_10_LONG"
    save_fig = "Rs_0.60"
    save_flag = "Rs_0.60"
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/d*Rh0.80_{save_flag}_Rc_{Rc:.02f}_ri_0.00'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    obj_path1 = f'local_density/dt50_LDmap_config_Rh0.40_Rs_0.80_Rc_{Rc:.02f}_ri_0.00'

    # with open(f'{obj_path1}/local_density.obj', 'rb') as f:
    #     local_density = pickle.load(f)
    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')

    # ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string.txt')
    ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string_Crt.txt')
    entire_system_without_string_ends = np.loadtxt(f'{obj_path}/entire_local_density_exclude_string_ends.txt')
    mean_radius = 0.4993804085195311
    if scale:
        x1 = -0.8
        x2 = 0.8
        local_tail_ini = np.array(local_density[0]) * np.pi*(Rc**2) / (np.pi*mean_radius**2)
        local_tail_final = np.array(local_density[1]) * np.pi*(Rc**2) / (np.pi*mean_radius**2)
        local_head_ini = np.array(local_density[2]) * np.pi*(Rc**2) / (np.pi*mean_radius**2)
        local_head_final = np.array(local_density[3]) * np.pi*(Rc**2) / (np.pi*mean_radius**2)
        ax.set_xlabel(fr'$\Delta S / \bar{{V_i}}$ ($R_c={Rc:.02f}$)', fontsize =24, fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
    else:
        x1 = -0.03
        x2 = 0.03
        local_tail_ini = local_density[0]
        local_tail_final = local_density[1]
        local_head_ini = local_density[2]
        local_head_final = local_density[3]
        ax.set_xlabel(f'Local density ($R_c={Rc:.02f}$)', fontsize=24, fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)


    tail = local_tail_final - local_tail_ini

    head = local_head_final - local_head_ini

    color1 = 'red'
    color2 = 'green'
    colors = [color1, color2]
    # ax[i].set_ylim([0, 100])
    # n, ave, mu, std = LDD.plot_norm_distribution(ax, [tail, entire_density, ld_instring, entire_system_without_string_ends],
    #                                              lable=[f'String Ends', 'Entire system', 'LD in string', 'LD exclude string ends'],
    #                            color=[color1, color2, 'k', 'g'], x1=x1, x2=x2)
    n, ave, mu, std = plot_norm_distribution(ax,
                                                 [tail, head],
                                                 lable=[f'Tail', 'head'],
                                                 color=colors, x1=x1, x2=x2)
    #     ax[i].set_title(f'local density distribution')
    #     ax[i].set_xlabel('local density')
    avelist.append(abs(ave[0]-ave[1]))
    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.vlines(x=np.mean(tail), ymin=0, ymax=h[1], color=color1, ls='--')
    ax.vlines(x=np.mean(head), ymin=0, ymax=h[1], color=color2, ls='--')
    # ax.vlines(x=np.mean(entire_system_without_string_ends), ymin=0, ymax=h[1]-25, color='g', ls='--')
    ax.legend(fontsize=18, loc='upper left',prop=font_prop)
    ax.set_ylabel(r'Relative frequency of $(\Delta \rho_{local} \pi (Rc/\bar{R_i})^2))$', fontsize=24, fontname='Times New Roman')
    ax.grid()
    std_string.append(std[0])
    std_system.append(std[1])
    if scale:
        # fig.text(0.5, 0.01, 'Sum area in the region', ha='center', va='center', fontsize='large')
        # ax[-1].set_xlabel('Sum area in the region')
        flag = "sumarea"
    else:
        # fig.text(0.5, 0.01, 'Local density', ha='center', va='center', fontsize='large')
        # ax[-1].set_xlabel('Local density')
        flag = "ld"
    fig.savefig(f'P2_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png', dpi=300)

# def plot_deltaV_vs_Rc(ax, rc, rh, rs, rho, save_LD, scale=True):
#     ## plot one best sample to show the local density or unoccupied volume distribution
#     Rc = rc
#     save_fig = f"Rs_{rs:.02f}"
#     rho = rho
#     save_flag = f"Rs_{rs:.02f}"
#     ##--------------------------------------------------------
#     pattern = f'{save_LD}/d*Rh{rh:.02f}_{save_flag}_Rc_{Rc:.02f}_ri_0.00'
#     print(pattern)
#     obj_path = glob.glob(pattern)[0]
#     local_density = np.loadtxt(f'{obj_path}/local_density.txt')
#     entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')
#     if not os.path.exists(f'./{obj_path}/ld_hopped.txt'):
#         sp.run(f'pwd')
#         sp.run(f'cat ./{obj_path}/ld_hopped_0*.txt > ./{obj_path}/ld_hopped.txt', shell=True)
#     ld_instring = np.loadtxt(f'{obj_path}/ld_hopped.txt')
#
#
#     if scale:
#         x1 = (Rc*2)**2*(1- (rho + 0.022))
#         x2 = (Rc*2)**2*(1- (rho - 0.035))
#         local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc*2)**2
#         local_tail_final = np.array(1 - local_density[:, 1]) * (Rc*2)**2
#         local_head_ini = np.array(1- local_density[:, 2]) * (Rc*2)**2
#         local_head_final = np.array(1- local_density[:, 3]) * (Rc*2)**2
#         entire_density = np.array(1 - entire_density) * (Rc*2)**2
#         ld_instring = np.array(1 - ld_instring) * (Rc*2)**2
#         ax.set_xlabel(fr'Unoccupied volume($V/\nu_1$)', fontsize =24, fontname='Times New Roman')
#         ax.tick_params(axis='x', labelsize=20)
#         ax.tick_params(axis='y', labelsize=20)
#     else:
#         x1 = rho-0.04
#         x2 = rho+0.025
#         local_tail_ini = local_density[:, 0]
#         local_tail_final = local_density[:, 1]
#         local_head_ini = local_density[:, 2]
#         local_head_final = local_density[:, 3]
#         ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24, fontname='Times New Roman')
#         ax.tick_params(axis='x', labelsize=20)
#         ax.tick_params(axis='y', labelsize=20)
#
#
#     tail = []
#     tail.extend(local_tail_ini)
#     tail.extend(local_head_final)
#
#     mean_tail = np.mean(tail)
#     mean_all = np.mean(entire_density)
#
#     delta_V = mean_tail-mean_all
#
#     return delta_V

def plot_5sets(rc, rs, rho, save_LD, data_path, scale=True, show=False):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    print(obj_path)
    # if not os.path.exists(f'{obj_path}/ld_hopped.txt'):
    #     sp.run(f'pwd')
    #     sp.run(f'cat ./{obj_path}/ld_hopped_0*.txt > ./{obj_path}/ld_hopped.txt', shell=True)
    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')

    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')
    ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string.txt')
    # x1_t = (1 - np.max(local_density[:, [0, 3]]))*(Rc*2)**2
    # x1_hd = (1 - np.max(local_density[:, [1, 2]])) * (Rc * 2) ** 2
    # x1_a = (1 - np.max(entire_density)) * (Rc * 2) ** 2
    # x1_m = (1 - np.max(ld_instring)) * (Rc * 2) ** 2
    # x1_hp = (1 - np.max(ld_hopped)) * (Rc * 2) ** 2
    # x2_t = (1 - np.min(local_density[:, [0, 3]])) * (Rc * 2) ** 2
    # x2_hd = (1 - np.min(local_density[:, [1, 2]])) * (Rc * 2) ** 2
    # x2_a = (1 - np.min(entire_density)) * (Rc * 2) ** 2
    # x2_m = (1 - np.min(ld_instring)) * (Rc * 2) ** 2
    # x2_hp = (1 - np.min(ld_hopped)) * (Rc * 2) ** 2

    # x1_t = (1 - np.mean(local_density[:, [0, 3]]))*(Rc*2)**2*x1_scale
    # x1_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2*x1_scale
    # x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2*x1_scale
    # x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2*x1_scale
    # x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2*x1_scale
    # x2_t = (1 - np.mean(local_density[:, [0, 3]])) * (Rc * 2) ** 2*x2_scale
    # x2_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2*x2_scale
    # x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2*x2_scale
    # x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2*x2_scale
    # x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2*x2_scale


    # x1_t = (1 - np.mean(local_density[:, [0, 3]])*x1_scale)*(Rc*2)**2
    # x1_hd = (1 - np.mean(local_density[:, [1, 2]])*x1_scale) * (Rc * 2) ** 2
    # x1_a = (1 - np.mean(entire_density)*x1_scale) * (Rc * 2) ** 2
    # x1_m = (1 - np.mean(ld_instring)*x1_scale) * (Rc * 2) ** 2
    # x1_hp = (1 - np.mean(ld_hopped)*x1_scale) * (Rc * 2) ** 2
    # x2_t = (1 - np.mean(local_density[:, [0, 3]])*x2_scale) * (Rc * 2) ** 2
    # x2_hd = (1 - np.mean(local_density[:, [1, 2]])*x2_scale) * (Rc * 2) ** 2
    # x2_a = (1 - np.mean(entire_density)*x2_scale) * (Rc * 2) ** 2
    # x2_m = (1 - np.mean(ld_instring)*x2_scale) * (Rc * 2) ** 2
    # x2_hp = (1 - np.mean(ld_hopped)*x2_scale) * (Rc * 2) ** 2

    shiftwidth = 2.0
    if Rc == 11.0:
        print(Rc)
    if local_density.ndim == 1:
        x1_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_a, x1_hp, x1_m, x1_hd, x1_t]
            x2 = [x2_a, x2_hp, x2_m, x2_hd, x2_t]
            local_tail_ini = np.array(1 - local_density[0]) * (Rc * 2) ** 2
            local_tail_final = np.array(1 - local_density[1]) * (Rc * 2) ** 2
            local_head_ini = np.array(1 - local_density[2]) * (Rc * 2) ** 2
            local_head_final = np.array(1 - local_density[3]) * (Rc * 2) ** 2
            entire_density = np.array(1 - entire_density) * (Rc * 2) ** 2
            ld_instring = np.array(1 - ld_instring) * (Rc * 2) ** 2
            ld_hopped = np.array(1 - ld_hopped) * (Rc * 2) ** 2
            ax.set_xlabel(fr'Local Unoccupied Volume($V/ v_1$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"

        tail = []
        tail.append(local_tail_ini)
        tail.append(local_head_final)

        head = []
        head.append(local_tail_final)
        head.append(local_head_ini)


    else:
        x1_t = (1 - np.mean(local_density[:, [0, 3]]))*(Rc*2)**2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[:, [0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_a, x1_hp, x1_m, x1_hd, x1_t]
            x2 = [x2_a, x2_hp, x2_m, x2_hd, x2_t]
            local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc*2)**2
            local_tail_final = np.array(1 - local_density[:, 1]) * (Rc*2)**2
            local_head_ini = np.array(1- local_density[:, 2]) * (Rc*2)**2
            local_head_final = np.array(1- local_density[:, 3]) * (Rc*2)**2
            entire_density = np.array(1 - entire_density) * (Rc*2)**2
            ld_instring = np.array(1 - ld_instring) * (Rc*2)**2
            ld_hopped = np.array(1 - ld_hopped) * (Rc*2)**2
            ax.set_xlabel(fr'Local Unoccupied Volume($V/ v_1$)   $\Delta r_{{ht}} > 2$', fontsize =24, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"


        tail = []
        tail.extend(local_tail_ini)
        tail.extend(local_head_final)

        head = []
        head.extend(local_tail_final)
        head.extend(local_head_ini)

    # std2 = np.std(np.array(tail)-np.array(head))/np.sqrt(len(head))
    std2 = np.std(tail)/np.sqrt(len(tail))
    colors = ['r', 'g', 'b', 'k', 'm']
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                 [entire_density, ld_hopped, ld_instring, head,  tail],
                                                         [50, 20, 10, 10, 10],
                                                 lable=['All',  'Hopped', 'String Middle', 'String head',  'String tail'],
                                                 color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')
    y_position=height
    ax.annotate(
        '',
        xy=(ave[4], y_position), xycoords='data',
        xytext=(ave[0], y_position), textcoords='data',
        arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    )
    DeltaV = ave[4] - (1-rho)*(Rc*2)**2
    ax.text(ave[2] + 0.3, y_position + 0.04, fr'$\Delta V / v_1={DeltaV:.02f}$', ha='center', va='bottom', fontsize=17, fontname='Times New Roman')
    avelist.append(abs(ave[0]-ave[4]))
    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.legend(loc='best',prop={'family': 'Times New Roman', 'size': 18}, frameon=False)
    ax.set_ylabel("Probability Density", fontsize=24)#, fontname='Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    std_string.append(std[0])
    std_system.append(std[1])
    save_path = f'{obj_path}'
    savefile = f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png'
    fig.savefig(savefile, dpi=300)
    print(f'{savefile}')
    if show:
        plt.show()
    return Rc, DeltaV, std2

def plot_4sets(rc, rs, rho, save_LD, data_path, scale=True, show=False, bins=[50, 20, 10, 9], figname="F0.png"):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    print(obj_path)
    # if not os.path.exists(f'{obj_path}/ld_hopped.txt'):
    #     sp.run(f'pwd')
    #     sp.run(f'cat ./{obj_path}/ld_hopped_0*.txt > ./{obj_path}/ld_hopped.txt', shell=True)
    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')

    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')
    ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string.txt')


    LEN = save_LD.split("Len")[1]
    LEN = LEN.split("_")[0]
    LEN = float(LEN)
    LEN = int(LEN)

    if local_density.ndim == 1:
        print('Only one line in local_density!!')
    else:
        if scale:
            local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc*2)**2
            local_tail_final = np.array(1 - local_density[:, 1]) * (Rc*2)**2
            local_head_ini = np.array(1- local_density[:, 2]) * (Rc*2)**2
            local_head_final = np.array(1- local_density[:, 3]) * (Rc*2)**2
            entire_density = np.array(1 - entire_density) * (Rc*2)**2
            ld_instring = np.array(1 - ld_instring) * (Rc*2)**2
            ld_hopped = np.array(1 - ld_hopped) * (Rc*2)**2
            ax.set_xlabel(fr'$V/ v_1$', fontsize =24, fontname='Times New Roman') #($\Delta r_{{ht}} > {LEN} \sigma$)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"


        tail = []
        tail.extend(local_tail_ini)
        tail.extend(local_head_final)

        head = []
        head.extend(local_tail_final)
        head.extend(local_head_ini)

    data_in = [entire_density, ld_hopped, head, tail]
    X1 = []
    X2 = []
    for i in range(4):
        s = np.std(data_in[i], ddof=1) * 3
        x1 = np.mean(data_in[i]) - s
        X1.append(x1)
        x2 = np.mean(data_in[i]) + s
        X2.append(x2)

    # std2 = np.std(np.array(tail)-np.array(head))/np.sqrt(len(head))
    std2 = np.std(tail)/np.sqrt(len(tail))
    colors = ['r', 'g', 'b', 'k', 'm']
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                 [entire_density, ld_hopped, head,  tail],
                                                         bins,
                                                 lable=[r'$V_{{A}}$(All particles)',  r'$V_{{Hp}}$(Hopped particles)', r'$V_{{Hd}}$(String heads)',  r'$V_{{T}}$(String tails)'],
                                                 color=colors, x1=X1, x2=X2)
    print(f'mu is {mu}')
    print(f'std is {std}')
    y_position=height
    ax.annotate(
        '',
        xy=(ave[3], y_position), xycoords='data',
        xytext=(ave[0], y_position), textcoords='data',
        arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    )
    DeltaV = ave[3] - ave[0]
    ax.text(ave[0] +0.36, y_position + 0.01, fr'$\Delta \overline{{V}}_{{AT}} ={DeltaV:.02f} v_1$', ha='center', va='bottom',
            fontsize=17, fontname='Times New Roman')
    # ax.annotate(
    #     '',
    #     xy=(ave[3], y_position-0.1), xycoords='data',
    #     xytext=(ave[2], y_position-0.1), textcoords='data',
    #     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    # )
    # DeltaV2 = ave[3] -ave[2]
    # ax.text(ave[2] + 0.6, y_position -0.09, fr'$\Delta \overline{{V}}_{{HT}} ={DeltaV2:.02f} v_1$', ha='center', va='bottom',
    #         fontsize=17, fontname='Times New Roman')
    avelist.append(abs(ave[0]-ave[3]))
    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.legend(loc='best',prop={'family': 'Times New Roman', 'size': 18}, frameon=False)
    ax.set_ylabel(r"$p(V/v_1)$", fontsize=24)#, fontname='Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    plt.tight_layout()
    std_string.append(std[0])
    std_system.append(std[1])
    save_path = f'{obj_path}'
    savefile = f'{save_path}/{figname}'
    fig.savefig(savefile, dpi=300)
    print(f'{savefile}')
    if show:
        plt.show()
    return Rc, DeltaV, std2

def time_reverse_symertry(rc, rs, rho, save_LD, data_path, show=False, bins=[50, 20, 10, 9], figname="F0.png"):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 8))
    avelist = []
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    print(obj_path)


    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    # entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')
    local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc * 2) ** 2
    local_tail_final = np.array(1 - local_density[:, 1]) * (Rc * 2) ** 2
    local_head_ini = np.array(1 - local_density[:, 2]) * (Rc * 2) ** 2
    local_head_final = np.array(1 - local_density[:, 3]) * (Rc * 2) ** 2

    ax.set_xlabel(fr'$V/ v$', fontsize=48,
                  fontname='Times New Roman')  # ($\Delta r_{{ht}} > {LEN} \sigma$)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    flag = "sumarea"
    x1 = [26.6, 26.6]
    x2 = [31, 31]

    colors = ['r', 'b', 'b', 'k', 'm']
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                     [local_tail_ini, local_head_final],
                                                     bins,
                                                     label=[r'Tail(pre-jump configuration)', r'Head(post-jump configuration)'],
                                                     color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')

    # ax.vlines(0, 0, 1, colors='k', linestyles='-', lw=2)

    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    # ax.set_xlim([-1, 1])
    ax.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 28}, frameon=False, handletextpad=0, borderaxespad=-0.2, bbox_to_anchor=(0, 1))
    ax.set_ylabel(r"$p(V/v)$", fontsize=48)  # , fontname='Times New Roman')
    ax.set_ylim([0, 0.8])
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2, labelsize=36)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)

    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    std_string.append(std[0])
    std_system.append(std[1])
    save_path = f'{obj_path}'
    current_path = os.path.abspath(__file__)
    save_fig = os.path.dirname(current_path)
    fig.savefig(f'{save_fig}/{figname}', dpi=300)
    if show:
        plt.show()

def DeltaV_at_tail_and_head(rc, rs, rho, save_LD, data_path, show=False, bins=[50, 20, 10, 9], figname="F0.png"):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 8))
    avelist = []
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    print(obj_path)


    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    # entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')
    local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc * 2) ** 2
    local_tail_final = np.array(1 - local_density[:, 1]) * (Rc * 2) ** 2
    local_head_ini = np.array(1 - local_density[:, 2]) * (Rc * 2) ** 2
    local_head_final = np.array(1 - local_density[:, 3]) * (Rc * 2) ** 2

    ax.set_xlabel(fr'$\Delta V/ v_1$', fontsize=48,
                  fontname='Times New Roman')  # ($\Delta r_{{ht}} > {LEN} \sigma$)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    flag = "sumarea"
    # ave_all = np.mean(entire_density, axis=0)
    # ave_all = ( 1 - ave_all )* (Rc * 2) ** 2
    DeltaV_t = local_tail_final - local_tail_ini
    DeltaV_h = local_head_final - local_head_ini
    # x1 = [min(DeltaV_t), min(DeltaV_h)]
    # x2 = [max(DeltaV_t), max(DeltaV_h)]
    x1 = [-2, -2]
    x2 = [2, 2]

    colors = ['b', 'r', 'b', 'k', 'm']
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                     [DeltaV_t, DeltaV_h],
                                                     bins,
                                                     lable=['String tail', 'String head'],
                                                     color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')

    ax.vlines(0, 0, 1, colors='k', linestyles='-', lw=2)

    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    # ax.set_xlim([-1, 1])
    ax.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 28}, frameon=False)
    ax.set_ylabel(r"$p(\Delta V/v_1)$", fontsize=48)  # , fontname='Times New Roman')
    ax.set_ylim([0, 1])
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2, labelsize=36)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    plt.tight_layout()
    std_string.append(std[0])
    std_system.append(std[1])
    save_path = f'{obj_path}'
    savefile = f'{save_path}/{figname}'
    fig.savefig(savefile, dpi=300)
    print(f'{savefile}')
    if show:
        plt.show()

def loop_vs_4sets(rc, rs, rho, save_LD, data_path, scale=True, show=False):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    print(obj_path)
    # if not os.path.exists(f'{obj_path}/ld_hopped.txt'):
    #     sp.run(f'pwd')
    #     sp.run(f'cat ./{obj_path}/ld_hopped_0*.txt > ./{obj_path}/ld_hopped.txt', shell=True)
    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')

    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')
    ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string.txt')

    shiftwidth = 2.0
    loop_minal_density = np.loadtxt(f'/home/xiaochu/Public/project-LUV/data/output/0805/AVE_loop_dt_100000_rs_0.6_rh_0.8_Lst_0_Len1.00_nseg_1000/dt100000_Rc_6.00/local_loop.txt')
    if local_density.ndim == 1:
        x1_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth
        x1_l = (1 - np.mean(loop_minal_density)) * (Rc * 2) ** 2 - shiftwidth
        x2_l = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_a, x1_hp, x1_t, x1_l]
            x2 = [x2_a, x2_hp, x2_t, x2_l]
            local_tail_ini = np.array(1 - local_density[0]) * (Rc * 2) ** 2
            local_tail_final = np.array(1 - local_density[1]) * (Rc * 2) ** 2
            local_head_ini = np.array(1 - local_density[2]) * (Rc * 2) ** 2
            local_head_final = np.array(1 - local_density[3]) * (Rc * 2) ** 2
            entire_density = np.array(1 - entire_density) * (Rc * 2) ** 2
            ld_instring = np.array(1 - ld_instring) * (Rc * 2) ** 2
            ld_hopped = np.array(1 - ld_hopped) * (Rc * 2) ** 2
            ld_loop = np.array(1 - loop_minal_density) * (Rc * 2) ** 2
            ax.set_xlabel(fr'Local Unoccupied Volume($V/ v_1$)', fontsize=24, fontname='Times New Roman')  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"

        tail = []
        tail.append(local_tail_ini)
        tail.append(local_head_final)

        head = []
        head.append(local_tail_final)
        head.append(local_head_ini)


    else:
        x1_t = (1 - np.mean(local_density[:, [0, 3]]))*(Rc*2)**2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[:, [0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth
        x1_l = (1 - np.mean(loop_minal_density)) * (Rc * 2) ** 2 - shiftwidth
        x2_l = (1 - np.mean(loop_minal_density)) * (Rc * 2) ** 2 + shiftwidth
        if scale:
            x1 = [x1_a, x1_hp, x1_t, x1_l]
            x2 = [x2_a, x2_hp, x2_t, x2_l]
            local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc*2)**2
            local_tail_final = np.array(1 - local_density[:, 1]) * (Rc*2)**2
            local_head_ini = np.array(1- local_density[:, 2]) * (Rc*2)**2
            local_head_final = np.array(1- local_density[:, 3]) * (Rc*2)**2
            entire_density = np.array(1 - entire_density) * (Rc*2)**2
            ld_instring = np.array(1 - ld_instring) * (Rc*2)**2
            ld_hopped = np.array(1 - ld_hopped) * (Rc*2)**2
            ld_loop = np.array(1 - loop_minal_density) * (Rc * 2) ** 2
            ax.set_xlabel(fr'Local Unoccupied Volume($V/ v_1$)', fontsize =24, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"


        tail = []
        tail.extend(local_tail_ini)
        tail.extend(local_head_final)

        head = []
        head.extend(local_tail_final)
        head.extend(local_head_ini)

    # std2 = np.std(np.array(tail)-np.array(head))/np.sqrt(len(head))
    std2 = np.std(tail)/np.sqrt(len(tail))
    colors = ['r', 'g', 'k', 'm', 'b']
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                 [entire_density, ld_hopped,  tail, ld_loop],
                                                         [50, 20, 9, 30],
                                                 lable=['All',  'Hopped particles',  rf'String tail $(\Delta r_{{ht}} > {2*Rc} \sigma)$', 'Maximal LUV in Loops'],
                                                 color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')
    y_position=height
    ax.annotate(
        '',
        xy=(ave[2], y_position), xycoords='data',
        xytext=(ave[0], y_position), textcoords='data',
        arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    )
    DeltaV = ave[2] - (1-rho)*(Rc*2)**2
    ax.text(ave[0] +0.36, y_position + 0.01, fr'$\Delta V_{{AT}} ={DeltaV:.02f} v_1$', ha='center', va='bottom',
            fontsize=17, fontname='Times New Roman')
    # ax.annotate(
    #     '',
    #     xy=(ave[3], y_position-0.1), xycoords='data',
    #     xytext=(ave[2], y_position-0.1), textcoords='data',
    #     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    # )
    # DeltaV2 = ave[3] -ave[2]
    # ax.text(ave[2] + 0.6, y_position -0.09, fr'$\Delta V_{{HT}} ={DeltaV2:.02f} v_1$', ha='center', va='bottom',
    #         fontsize=17, fontname='Times New Roman')
    avelist.append(abs(ave[0]-ave[3]))
    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.legend(loc='best',prop={'family': 'Times New Roman', 'size': 16}, frameon=False)
    ax.set_ylabel("Probability Density Function", fontsize=24)#, fontname='Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    std_string.append(std[0])
    std_system.append(std[1])
    save_path = f'{obj_path}'
    savefile = f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_loop_4sets.png'
    fig.savefig(savefile, dpi=300)
    print(f'{savefile}')
    if show:
        plt.show()
    return Rc, DeltaV, std2


def plot_3sets(rc, rs, rho, save_LD, data_path, scale=True, show=False):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/{data_path}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    print(obj_path)
    # if not os.path.exists(f'{obj_path}/ld_hopped.txt'):
    #     sp.run(f'pwd')
    #     sp.run(f'cat ./{obj_path}/ld_hopped_0*.txt > ./{obj_path}/ld_hopped.txt', shell=True)
    local_density = np.loadtxt(f'{obj_path}/local_density.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')

    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')
    ld_instring = np.loadtxt(f'{obj_path}/local_density_in_string.txt')

    shiftwidth = 2.0
    if local_density.ndim == 1:
        x1_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[[0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[[1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_a, x1_hp, x1_m, x1_hd, x1_t]
            x2 = [x2_a, x2_hp, x2_m, x2_hd, x2_t]
            local_tail_ini = np.array(1 - local_density[0]) * (Rc * 2) ** 2
            local_tail_final = np.array(1 - local_density[1]) * (Rc * 2) ** 2
            local_head_ini = np.array(1 - local_density[2]) * (Rc * 2) ** 2
            local_head_final = np.array(1 - local_density[3]) * (Rc * 2) ** 2
            entire_density = np.array(1 - entire_density) * (Rc * 2) ** 2
            ld_instring = np.array(1 - ld_instring) * (Rc * 2) ** 2
            ld_hopped = np.array(1 - ld_hopped) * (Rc * 2) ** 2
            ax.set_xlabel(fr'Local Free Volume($V/ v_1$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)  # , fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"

        tail = []
        tail.append(local_tail_ini)
        tail.append(local_head_final)

        head = []
        head.append(local_tail_final)
        head.append(local_head_ini)


    else:
        x1_t = (1 - np.mean(local_density[:, [0, 3]]))*(Rc*2)**2 - shiftwidth
        x1_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 - shiftwidth
        x1_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 - shiftwidth
        x1_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 - shiftwidth
        x1_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 - shiftwidth
        x2_t = (1 - np.mean(local_density[:, [0, 3]])) * (Rc * 2) ** 2 + shiftwidth
        x2_hd = (1 - np.mean(local_density[:, [1, 2]])) * (Rc * 2) ** 2 + shiftwidth
        x2_a = (1 - np.mean(entire_density)) * (Rc * 2) ** 2 + shiftwidth
        x2_m = (1 - np.mean(ld_instring)) * (Rc * 2) ** 2 + shiftwidth
        x2_hp = (1 - np.mean(ld_hopped)) * (Rc * 2) ** 2 + shiftwidth

        if scale:
            x1 = [x1_a, x1_hp, x1_m, x1_hd, x1_t]
            x2 = [x2_a, x2_hp, x2_m, x2_hd, x2_t]
            local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc*2)**2
            local_tail_final = np.array(1 - local_density[:, 1]) * (Rc*2)**2
            local_head_ini = np.array(1- local_density[:, 2]) * (Rc*2)**2
            local_head_final = np.array(1- local_density[:, 3]) * (Rc*2)**2
            entire_density = np.array(1 - entire_density) * (Rc*2)**2
            ld_instring = np.array(1 - ld_instring) * (Rc*2)**2
            ld_hopped = np.array(1 - ld_hopped) * (Rc*2)**2
            ax.set_xlabel(fr'Lcal Free Volume($V/ v_1$)   $\Delta r_{{ht}} > 2$', fontsize =24, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "sumarea"
        else:

            local_tail_ini = local_density[:, 0]
            local_tail_final = local_density[:, 1]
            local_head_ini = local_density[:, 2]
            local_head_final = local_density[:, 3]
            ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)#, fontname='Times New Roman')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            flag = "ld"


        tail = []
        tail.extend(local_tail_ini)
        tail.extend(local_head_final)

        head = []
        head.extend(local_tail_final)
        head.extend(local_head_ini)

    # std2 = np.std(np.array(tail)-np.array(head))/np.sqrt(len(head))
    std2 = np.std(tail)/np.sqrt(len(tail))
    colors = ['r', 'g', 'b', 'k', 'm']
    n, ave, mu, std, height = plot_norm_distribution(ax,
                                                 [entire_density, ld_hopped, tail],
                                                         [50, 20, 9],
                                                 lable=['All',  'Hopped', 'String tail'],
                                                 color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')
    y_position=height
    ax.annotate(
        '',
        xy=(ave[2], y_position), xycoords='data',
        xytext=(ave[0], y_position), textcoords='data',
        arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    )
    DeltaV = ave[2] - (1-rho)*(Rc*2)**2
    ax.text(ave[0]+0.35, y_position+0.01, fr'$\Delta V / v_1={DeltaV:.02f}$', ha='center', va='bottom', fontsize=17, fontname='Times New Roman')
    avelist.append(abs(ave[0]-ave[2]))
    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.legend(loc='best',prop={'family': 'Times New Roman', 'size': 18}, frameon=False)
    ax.set_ylabel("Probability Density Function", fontsize=24)#, fontname='Times New Roman')
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    std_string.append(std[0])
    std_system.append(std[1])
    save_path = f'{obj_path}'
    savefile = f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_3set.png'
    fig.savefig(savefile, dpi=300)
    print(f'{savefile}')
    if show:
        plt.show()
    return Rc, DeltaV, std2


def plot_push(data_file):
    fig, ax = plt.subplots(figsize=(10, 8))
    push_area = np.loadtxt(data_file)

    color1 = 'red'
    color2 = 'green'
    colors = [color1, color2]
    tail = push_area[:, 0]/(np.pi*0.499380408519531**2)
    head = push_area[:, 1]/(np.pi*0.499380408519531**2)
    x1 = np.min(tail) - 0.1
    x2 = np.max(head) + 0.1

    n, ave, mu, std = plot_norm_distribution(ax,[tail, head], lable=[f'Tail', 'head'],
                                                 color=colors, x1=x1, x2=x2, fit=False, density=False)

    h = ax.get_ylim()
    ax.set_ylim([0, h[1] * 1.2])
    ax.vlines(x=np.mean(tail), ymin=0, ymax=h[1], color=color1, ls='--', label=f'{ave[0]}')
    ax.vlines(x=np.mean(head), ymin=0, ymax=h[1], color=color2, ls='--', label=f'{ave[1]}')

    ax.legend(fontsize=18, loc='upper left', prop=font_prop)
    ax.set_ylabel('Frequency', fontsize=24, fontname='Times New Roman')
    ax.grid()
    fig.savefig(f'P1_Rc_push.png', dpi=300)
    plt.show()



def plot_shift(rc, rh, rs, rho, save_LD, Ls, scale=True):
    ## plot one best sample to show the local density or unoccupied volume distribution
    Rc = rc
    save_fig = f"Rs_{rs:.02f}"
    rho = rho
    save_flag = f"Rs_{rs:.02f}"
    ##--------------------------------------------------------

    fig, ax = plt.subplots(figsize = (10,8))
    avelist =[]
    prb = []
    std_string = []
    std_system = []
    pattern = f'{save_LD}/dt*Rh{rh:.02f}_{save_flag}_Rc_{Rc:.02f}_ri_0.00_Lst_{Ls:02d}'
    print(pattern)
    obj_path = glob.glob(pattern)[0]
    # obj_path1 = f'local_density/dt50_LDmap_config_Rh0.40_Rs_0.80_Rc_{Rc:.02f}_ri_0.00'

    # with open(f'{obj_path1}/local_density.obj', 'rb') as f:
    #     local_density = pickle.load(f)
    local_tail_iandf = np.loadtxt(f'{obj_path}/local_tail_iandf.txt')
    local_head_iandf = np.loadtxt(f'{obj_path}/local_head_iandf.txt')
    entire_density = np.loadtxt(f'{obj_path}/entire_density.txt')
    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')

    ld_hopped = np.loadtxt(f'{obj_path}/ld_hopped.txt')
    x1_t = (1 - max(max(local_tail_iandf[:, i]) for i in range(2)))*(Rc*2)**2
    x1_hd = (1 - max(max(local_head_iandf[:, i]) for i in range(2))) * (Rc * 2) ** 2
    x1_a = (1 - np.max(entire_density)) * (Rc * 2) ** 2
    x1_hp = (1 - np.max(ld_hopped)) * (Rc * 2) ** 2
    x2_t = (1 - min(min(local_tail_iandf[:, i]) for i in range(2))) * (Rc * 2) ** 2
    x2_hd = (1 - min(min(local_head_iandf[:, i]) for i in range(2))) * (Rc * 2) ** 2
    x2_a = (1 - np.min(entire_density)) * (Rc * 2) ** 2
    x2_hp = (1 - np.min(ld_hopped)) * (Rc * 2) ** 2


    if scale:
        x1 = [x1_hd, x1_t,  x1_a, x1_hp]
        x2 = [x2_hd, x2_t,  x2_a, x2_hp]

        local_tail_ini = np.array(1 - local_tail_iandf[:, 0]) * (Rc*2)**2
        local_tail_final = np.array(1 - local_tail_iandf[:, 1]) * (Rc*2)**2
        local_head_ini = np.array(1- local_head_iandf[:, 0]) * (Rc*2)**2
        local_head_final = np.array(1- local_head_iandf[:, 1]) * (Rc*2)**2
        entire_density = np.array(1 - entire_density) * (Rc*2)**2
        ax.set_xlabel(fr'$Unoccupied\ Volume(V/ v_1) $ ', fontsize =24)#, fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ld_hopped = np.array(1 - ld_hopped) * (Rc*2)**2
        ax.set_xlabel(fr'$Unoccupied\ Volume(V/ v_1)$', fontsize =24)#, fontname='Times New Roman')
        flag = "sumarea"
    else:
        x1 = [rho - 0.04, rho - 0.04, rho - 0.04, rho - 0.04]
        x2 = [rho + 0.025, rho + 0.025, rho + 0.025, rho + 0.025]
        local_tail_ini = local_tail_iandf[:, 0]
        local_tail_final = local_tail_iandf[:, 1]
        local_head_ini = local_head_iandf[:, 0]
        local_head_final = local_head_iandf[:, 1]
        ax.set_xlabel(fr'Local density ($R_c={Rc:.02f}$)', fontsize=24)#, fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        flag = "ld"


    tail = []
    tail.extend(local_tail_ini)
    # tail.extend(local_head_final)

    head = []
    head.extend(local_tail_final)
    # head.extend(local_head_ini)

    colors = ['r', 'g', 'b', 'k']
    n, ave, mu, std = plot_norm_distribution(ax,
                                                 [head, tail, entire_density, ld_hopped], [30,30,100, 100],
                                                 lable=[r'$Head$', r'$Tail$', r'$All$', r'$Hopped$'],
                                                 color=colors, x1=x1, x2=x2)
    print(f'mu is {mu}')
    print(f'std is {std}')
    y_position=max(max(nl) for nl in n)*1.05
    ax.annotate(
        '',
        xy=(ave[1], y_position), xycoords='data',
        xytext=(ave[0], y_position), textcoords='data',
        arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', lw=2)
    )
    ax.text(ave[1] + 0.3, y_position + 0.04, fr'$\Delta V / v_1={ave[0]-ave[1]:.02f}$', ha='center', va='bottom', fontsize=17, fontname='Times New Roman')
    avelist.append(abs(ave[0]-ave[1]))
    prb.append(stats.norm.cdf(mu[1], loc=mu[0], scale=std[0]))
    h = ax.get_ylim()
    ax.set_ylim([0, h[1]*1.2])
    ax.legend(loc='best',prop={'family': 'Times New Roman', 'size': 18}, frameon=False)
    ax.set_ylabel(r'$p$', fontsize=24)
    # ax.grid()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    std_string.append(std[0])
    std_system.append(std[1])
    save_path = f'{save_LD}/LD_Distribution'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png', dpi=300)
    print(f'{save_path}/DISTRI_Rc_{rc:.02f}_impact_tool_{flag}_{save_fig}_exlude_string_LONG.png')
    plt.show()

def main(mode, rho):
    save_LD = f'{project_root}/data/output/0805/AVE_dt_100000_rs_0.6_rh_0.8_Lst_0_Len2.00_nseg_1000'
    data_path = 'Rc_6.00'
    case = ['MultiTasks', 'SingleTask', 'DV-Rc', '5sets', 'shift']
    if case[mode]=='MultiTasks':
        pool = multiprocessing.Pool(processes=80)
        tasks = np.arange(1.5, 9.5, 0.1)
        task_in = []
        for rc_i in tasks:
            task = (rc_i, 0.5, 0.5, rho, "dt_00016_local_density_rs_0.5_rh_0.8_Lst_0_Len2_LONG", True)
            task_in.append(task)
        pool.starmap(plot_rc, task_in)
        pool.close()
        pool.join()
    elif case[mode]=='SingleTask':
        # #plot differential local density
        Rc = 6
        time_reverse_symertry(rc=Rc, rs=0.6, rho=rho,
                save_LD=save_LD,
                data_path=data_path,
                bins=[40, 40],
                   figname=f"F3-c-rh2_Rc_{Rc:.02f}.png")
        # plot_rc_differential(rc=1.72, scale=True)
        # plot_push("local_density_rs_0.6_rh_0.8_Lst_10_LONG_2/V_push.txt")
    elif case[mode]=='5sets':
        current_dir = os.getcwd()
        ave_folders = [name for name in os.listdir(current_dir)
                       if name.startswith('AVE_noshift') and os.path.isdir(name)]
        data_save = []
        for folder in ave_folders:
            pattern = os.path.join(current_dir, folder, 'dt*/')
            dt_subfolders = glob.glob(pattern)
            dt_folder = os.path.basename(dt_subfolders[0].rstrip('/'))
            print(folder)
            print(dt_folder)
            ave_sp = folder.split("_")
            Rc = float(ave_sp[5])
            print(f'{ave_sp[4]} is {Rc}')
            rs = float(ave_sp[7])
            print(f'{ave_sp[6]} is {rs}')
            # out = plot_5sets(rc=Rc, rs=rs, rho=rho, show=False, x1_scale=1.02*(189.5-Rc)/187.5, x2_scale=0.98*(2-(189.5-Rc)/187.5),
            #                  save_LD=folder, data_path=dt_folder, scale=True)
            out = plot_5sets(rc=Rc, rs=rs, rho=rho, show=False, x1_scale=0.96, x2_scale=1.02,
                             save_LD=folder, data_path=dt_folder, scale=True)
            data_save.append(out)
            plt.close()
        data_save = np.array(data_save)
        data_indices = np.argsort(data_save[:, 0])
        data_save = data_save[data_indices]
        np.savetxt(f'data_save_{rho:.03f}.csv', data_save, delimiter=',')
        fig, ax = plt.subplots()
        ax.errorbar(data_save[:, 0], data_save[:, 1], yerr=data_save[:, 2],fmt='s-', markerfacecolor='none', capsize=2)
        # ax.plot(data_save[:, 0], data_save[:, 1])
        ax.set_ylabel(r'$\Delta V/\nu_1$', fontsize=18)
        ax.set_xlabel(r'$Rc$', fontsize=18)
        plt.show()
        fig.savefig(f'data_save_std.png', dpi=600)
        # Rc=7.0
        # plot_5sets(rc=7.0, rs=0.6, rho=0.805, show=True, x1_scale=1.05*(189.5-Rc)/187.5, x2_scale=0.95*(2-(189.5-Rc)/187.5),
        #            save_LD="AVE_noshift_dt_100000_Rc_7.00_rs_0.6_rh_0.8_Lst_0_Len14.00_nseg_1000",
        #            data_path="dt100000_Rh0.80_Rs_0.60_Rc_7.00_ri_0.00_Lst_00", scale=True)
    elif case[mode]=='shift':  # mode=4
        plot_shift(rc=3.20, rh=0.4, rs=0.6, rho=0.800,
                   save_LD="AVE_dt_100000_Rc_3.20_rs_0.6_rh_0.4_Lst_0_Len6.40_nseg_10000",
                Ls=0, scale=True)

if __name__ == '__main__':
    main(1, 0.805)
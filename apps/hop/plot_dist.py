import os
import sys
from hd2d_src.HopTrack.utils.path import get_sub_project_root
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

project_root = get_sub_project_root()
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 24,
    'axes.labelsize': 30,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 19,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'legend.frameon': False
})

dt_list = [10, 100, 500, 2000, 10000, 25000, 50000, 100000]
dt_list = dt_list[::-1]
Rc = 6.0
ld_hp = []
colors = []
lt = []
alpha_list = []
nbins = []
labels = []
markers = []
mdc = [ '<', 'o', '>',  'D', 'v', 'p', '^', '*', '+', 'x', '|', 'h', 's', 'd']
dc = 0.6 / len(dt_list)
for i in range(len(dt_list)):
    lt.append('--')
    colors.append('g')
    alpha_list.append(0.9 - i * dc)
    nbins.append(20)
    labels.append(f'$\delta t =${dt_list[i]*41}')
    markers.append(mdc[i])
lt.extend(['-', '-'])
colors.extend(['r', 'k'])
alpha_list.extend([1, 1])
nbins.extend([10, 50])
labels.extend([f'$V_T(\delta t={100*41})$', '$V_A$'])
markers.extend(['d', 's'])
x1 = []
x2 = []

local_density = np.loadtxt(
    f'{project_root}/data/output/0805/AVE_long_string_dt_100000_Rc_6.00_rs_0.6_rh_0.8_Lst_0_Len12.00_nseg_1000/Rc_6.00/local_density.txt')
local_tail_ini = np.array(1 - local_density[:, 0]) * (Rc * 2) ** 2
local_head_final = np.array(1 - local_density[:, 3]) * (Rc * 2) ** 2
tail = []
tail.extend(local_tail_ini)
tail.extend(local_head_final)

entire_density = np.loadtxt(
    f'{project_root}/data/output/0805/AVE_long_string_dt_100000_Rc_6.00_rs_0.6_rh_0.8_Lst_0_Len12.00_nseg_1000/Rc_6.00/entire_density.txt')
entire_density = np.array(1 - entire_density) * (Rc * 2) ** 2

for dt in dt_list:
    ld = np.loadtxt(f'{project_root}/apps/hop/ld_hopped_dt_{dt}.txt')
    ld_hopped = np.array(1 - ld) * (Rc * 2) ** 2
    ld_hp.append(ld_hopped)
ld_hp.append(tail)
ld_hp.append(entire_density)

X1 = []
X2 = []
for i in range(len(ld_hp)):
    s = np.std(ld_hp[i], ddof=1) * 3
    x1 = np.mean(ld_hp[i]) - s
    X1.append(x1)
    x2 = np.mean(ld_hp[i]) + s
    X2.append(x2)

fig, ax = plt.subplots(figsize=(10, 8))
n, ave, mu, std, height = plot_norm_distribution(ax,
                                                 ld_hp,
                                                 nbins=nbins,
                                                 label=labels,
                                                 color=colors, lt=lt, alpha_list=alpha_list,
                                                 markers=markers,
                                                 x1=X1, x2=X2, vline=False, ms=10, lw=4)
# ax.legend(loc='upper right', title='$\delta t$', frameon=False, handletextpad=0, borderaxespad=-0.1)
ax.legend(loc='upper right', frameon=False, handletextpad=0, borderaxespad=-0.1)
ax.set_xlabel('$V / v$')
ax.set_ylabel('$p(V / v)$')
ax.minorticks_on()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(which='both', width=2)
ax.tick_params(which='major', length=7)
ax.tick_params(which='minor', length=4)
ax.set_ylim([0, 0.8])
plt.tight_layout()
current_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_path)
fig.savefig(f'{save_path}/FigS2b_hop.jpg')
plt.show()

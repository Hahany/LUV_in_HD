import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import sys
from hd2d_src.HopTrack.utils.path import get_sub_project_root
project_root = get_sub_project_root()
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 17,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'legend.frameon': False
})
from hd2d_src.HopTrack.utils import *

current_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_path)
obj_a_path = f'{project_root}/data/input/0805/a_AVE_normalized.obj'

# plot figure
sisf = np.loadtxt(f'{save_path}/sisf.txt')
dt_out = sisf[0, :]*41
sisf_out = sisf[1, :]
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(dt_out, sisf_out, '-')
t_hop = np.array([10, 100, 500, 2000, 10000, 25000, 50000, 100000])*41
t_hop = t_hop.tolist()[::-1]
sisf_hop = np.interp(t_hop, dt_out, sisf_out)
tau = np.interp(1/np.e, sisf_out[::-1], dt_out[::-1])
print(f'the SISF relaxation time is {tau}')
mdc = ['<', 'o', '>',  'D', 'v', 'p', '^', '*', '+', 'x', '|', 'h', 's', 'd']
for i in range(len(t_hop)):
    ax.plot(t_hop[i], sisf_hop[i], marker=mdc[i], fillstyle='full',
            markeredgecolor='g', markerfacecolor='g', markersize=10)
# x = np.linspace(dt_out[-10], dt_out[-1], 100)
# ax.plot(x, x / max(dt_out) * msd_out[-1], '--', c='k', label='$t^1$')
ax.text(10, 1/np.e+0.05, r'$e^{-1}$')
ax.axhline(1, c='k', linestyle='-')
ax.axhline(1/np.e, c='k', linestyle='-')
# ax.legend(loc='upper left', fontsize=18, handletextpad=0, borderaxespad=-0.2, bbox_to_anchor=(0, 1), title='$\delta t$')
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel(r'$t$')
ax.set_ylabel('SISF')
ax.set_xlim([5, 150000*41])
ax.set_xticks([i for i in [100, 10000, 1e6]])
fig.tight_layout()
fig.savefig(f'{save_path}/Fig2b_sisf-SI.png', dpi=600)



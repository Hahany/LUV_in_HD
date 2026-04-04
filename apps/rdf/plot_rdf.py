import os
import sys
from hd2d_src.HopTrack.utils.path import get_sub_project_root
project_root=get_sub_project_root()
from matplotlib import pyplot as plt
import RDFpar as rdf
import numpy as np
from hd2d_src.HopTrack.utils import *
from scipy.signal import find_peaks
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'legend.frameon': False
})

with open(f"{project_root}/data/input/0805/a_AVE_test.obj", "rb") as f:
    custom_unpickler = CustomUnpickler(f)
    a = custom_unpickler.load()

edges = np.linspace(0, 5,200)
tlist = np.arange(0, 10)
fig, ax = plt.subplots()
x = 1/2*(edges[:-1]+edges[1:])
if not os.path.exists(f'{project_root}/apps/rdf/out.txt'):
    out, label = rdf.RDFpar(a, tlist, edges)
    x = np.array(x).reshape(-1, 1)
    plot_data = np.hstack((x, out))
    np.savetxt(f'{project_root}/apps/rdf/out.txt', plot_data)
    np.savetxt(f'{project_root}/apps/rdf/label.txt', label, fmt='%s')
else:
    data = np.loadtxt(f'{project_root}/apps/rdf/out.txt')
    label = np.loadtxt(f'{project_root}/apps/rdf/label.txt', dtype=str)
    out = data[:, 1:]
    x = data[:, 0]
    peaks1,_ = find_peaks(out[:, 0])
    peaks2,_ = find_peaks(out[:, 1])
    peaks3,_ = find_peaks(out[:, 2])

ax.vlines(x[peaks1[0]],0, 5, color='b', linestyles='--', lw=1)
ax.vlines(x[peaks2[0]],0, 5, color='r', linestyles='--', lw=1)
ax.vlines(x[peaks3[0]],0, 5, color='g', linestyles='--', lw=1)
ax.set_prop_cycle('color', ['b', 'r', 'g', 'k'])
ax.plot(x, out, label=label)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$g(r)$')
ax.legend(frameon=False)
plt.minorticks_on()
ax.minorticks_on()
ax.set_xticks([i for i in range(6)])
# ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='black')
# ax.grid(True, which='minor', linestyle=':', linewidth=0.25, color='grey')
plt.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
plt.tick_params(axis='y', which='both', left=True, right=True, direction='in')
plt.tight_layout()
current_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_path)
fig.savefig(f"{save_path}/FigS1a_RDFpar.png")
plt.show()


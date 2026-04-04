import os
import sys
project_root = '/home/xiaochu/Public/project-LUV'
sys.path.append(project_root)
import numpy as np
import matplotlib.pyplot as plt
from hd2d_src.HopTrack.utils import *
import glob
import scipy.stats as stats
from scipy.stats import probplot
from scipy.stats import norm as nm
import matplotlib.font_manager as fm
import multiprocessing
import os
import matplotlib as mpl
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 24,
    'axes.labelsize': 30,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'legend.frameon': False
})
def compute_sisf(a, k_opt, dt, interval=10, frinc=1.4):
    """
    positions: ndarray, shape (nframe, n, dim), unwrapped coordinates
    box_length: float, simulation box length
    k_opt: float, reduced wave number (in units of 2¦Ð/L)
    dt: float, time step
    interval: int, sampling interval in frames
    frinc: float, growth factor for time lags
    """
    nframe, n, dim = np.shape(a.frames)
    k = 2 * np.pi / np.array(k_opt)
    print(f'k is: {k}')
    # Generate delay times
    delays = []
    dfx = 1
    while dfx < nframe:
        delays.append(dfx)
        dfx = int(round(dfx * frinc + 1))
    delays.append(nframe-1)
    delays = np.array(delays)
    times = delays * dt

    # SISF results
    Fs = np.zeros(len(delays))

    for i, d in enumerate(delays):
        frsamp = np.arange(nframe - d - 1, -1, -interval)
        dr = a.frames[frsamp + d, :, 2:4] - a.frames[frsamp, :, 2:4]
        cos_term = np.cos(k * dr)
        Fs[i] = np.mean(cos_term)
        print(f'time: {d}')
    return times, Fs


if __name__ == '__main__':
    save_path = '/home/xiaochu/Public/project-LUV/apps/SISF'
    obj_a_path = '/home/xiaochu/Public/project-LUV/data/input/0805/a_AVE_normalized.obj'
    if not os.path.exists(f'{save_path}/sisf.txt'):
        with open(obj_a_path, "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            a = custom_unpickler.load()

        kn = [1, 1]
        t, Fs = compute_sisf(a, kn, dt=1)
        np.savetxt(f'{save_path}/sisf.txt', [t, Fs])
    else:
        print('sisf already computed')

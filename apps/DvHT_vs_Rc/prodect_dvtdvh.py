
from hd2d_src.HopTrack.utils.path import get_sub_project_root
from hd2d_src.HopTrack.core import *
import os
import numpy as np
from matplotlib import pyplot as plt
from  natsort import natsorted
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 28,
    'axes.labelsize': 30,
    'axes.titlesize': 34,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 22,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'legend.frameon': False
})

project_root = get_sub_project_root()
current_path = os.path.abspath(__file__)
data_dir = os.path.dirname(current_path)
data_path = f'{project_root}/data/output/0805/fix_cr_DHT_Nf_1000'


def get_deltaV_HT(dirData, prefix):
    ave_folders = natsorted([name for name in os.listdir(dirData)
                   if name.startswith(prefix)])  #and os.path.isdir(name)

    tail_ini = []
    tail_fin = []
    head_ini = []
    head_fin = []
    for folder_up in ave_folders:
        path = os.path.join(dirData, folder_up)
        folder = next(
            (name for name in os.listdir(path)
             if name.startswith('Rc_') and os.path.isdir(os.path.join(path, name))),
            None
        )
        Rc = float(folder.split('Rc_')[1])
        tail_file_path = os.path.join(dirData, folder_up, folder, 'local_density.txt')
        print(tail_file_path)
        all_file_path = os.path.join(dirData, folder_up, folder, 'entire_density.txt')
        print(all_file_path)
        i = float(folder.split('Rc_')[1])
        if os.path.isfile(all_file_path):
            ld = np.loadtxt(tail_file_path)
            all = np.loadtxt(all_file_path)
            if len(np.shape(ld)) > 1:
                local_tail_init = (1 - np.array(ld[:, 0]))*(Rc * 2) ** 2
                local_tail_final = (1 - np.array(ld[:, 1]))*(Rc * 2) ** 2
                local_head_init = (1 - np.array(ld[:, 2]))*(Rc * 2) ** 2
                local_head_final = (1 - np.array(ld[:, 3]))*(Rc * 2) ** 2

                tail_ini.extend(local_tail_init)
                head_ini.extend(local_head_init)
                tail_fin.extend(local_tail_final)
                head_fin.extend(local_head_final)

                return tail_ini, tail_fin, head_ini, head_fin

tail_ini, tail_fin, head_ini, head_fin = get_deltaV_HT(data_path, prefix='rh_0.8_rs_0.6_Lst_0_Len4.00')

dv_T = np.array(tail_ini) - np.array(tail_fin)
dv_H = np.array(head_ini) - np.array(head_fin)

product = dv_T * dv_H

fig, ax = plt.subplots()
ax.plot(product, '.')
ax.hlines(0, 0, len(product), 'k')
ax.set_xlim([0, len(product)])
ax.set_xlabel('string')
ax.set_ylabel(rf'$\Delta V_H \Delta V_T$')
fig.tight_layout()
ax.set_xticks([])
fig.savefig('/home/xiaochu/Public/local_free_volume_and_stringlike_motions_in_glasses/hard_disk_2d/apps/DvHT_vs_Rc/prodect_hd.png')


fig1, ax1 = plt.subplots()
hist = ax1.hist(product, bins=60, alpha=0.7, color='blue', edgecolor='black', density=True)
portion = len(product[product<0])/len(product)
print(rf'The ratio of $\Delta V_H * \Delta V_T < 0 is: $ {portion*100:.02f}% < 0')
ax1.set_xlim([-0.6, 0.6])
ax1.set_ylim([0, max(hist[0])])
ax1.vlines(0, 0, max(hist[0]), 'k')
ax1.set_xlabel(r'$\Delta V_H \Delta V_T$')
ax1.set_ylabel(r'$p(\Delta V_H \Delta V_T)$')
ax1.text(-0.5, 10, rf"{portion*100:.02f}$\% < 0$")
fig1.tight_layout()
fig1.savefig('/home/xiaochu/Public/local_free_volume_and_stringlike_motions_in_glasses/hard_disk_2d/apps/DvHT_vs_Rc/prodect_hist_hd.png', dpi=300)


fig, ax = plt.subplots()
dv_T = -dv_T
dv_H = -dv_H
ax.scatter(dv_T, dv_H)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel(r'$\Delta V_T$')
ax.set_ylabel(r'$\Delta V_H$')
fig.tight_layout()
ax.vlines(0, -1, 1, 'k')
ax.hlines(0, -1, 1, 'k')
n00 = np.sum((dv_T < 0) & (dv_H <= 0))
n01 = np.sum((dv_T < 0) & (dv_H > 0))
n10 = np.sum((dv_T >= 0) & (dv_H <= 0))
n11 = np.sum((dv_T >= 0) & (dv_H >0))
n = len(dv_T)
ax.text(-0.99, -0.99 , rf'{np.round(n00/n*100):.0f}$\%$')
ax.text(-0.99, 0.8 , rf'{np.round(n01/n*100):.0f}$\%$')
ax.text(0.55, -0.99 , rf'{np.round(n10/n*100):.0f}$\%$')
ax.text(0.55, 0.8 , rf'{np.round(n11/n*100):.0f}$\%$')
fig.savefig('/home/xiaochu/Public/local_free_volume_and_stringlike_motions_in_glasses/hard_disk_2d/apps/DvHT_vs_Rc/dvT_vs_dvH_hd.png')
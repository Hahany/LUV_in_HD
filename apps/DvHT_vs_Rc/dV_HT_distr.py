
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

    data_save = []
    R_list = []
    error = []
    data_all = []
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
                local_tail_ini = ld[:, 0]
                local_head_final = ld[:, 3]

                tail = []
                tail.extend(local_tail_ini)
                tail.extend(local_head_final)

                local_head_init = ld[:, 1]
                local_tail_final = ld[:, 2]

                head = []
                head.extend(local_head_init)
                head.extend(local_tail_final)

                head_spc = []
                head_spc.extend(local_tail_final)
                head_spc.extend(local_head_init)

                dv_spc = (np.array(head)-np.array(tail))*(Rc * 2) ** 2
                dv = (np.array(head_spc)-np.array(tail))*(Rc * 2) ** 2
                return dv_spc, dv

dV, dV_spc = get_deltaV_HT(data_path, prefix='rh_0.8_rs_0.6_Lst_0_Len4.00')
dV = -dV
n0 = len(dV[dV>=0])
n1 = len(dV[dV<0])
print(f'number of dV lager than 0 : {n0}')
print(f'number of dV smaller than 0 : {n1}')
np.savetxt(f'{data_dir}/deltaV_HT.csv', dV)
fig, ax = plt.subplots(figsize=(10, 8))
data = [dV]
nbins = [16]
label = [r'']
color = ['k']
x1 = [min(dV)]
x2 = [max(dV)]
_, _, mu_list, sigma_list, h=plot_norm_distribution(ax, data, nbins, label, color, x1, x2, mcolor=None,  pset=None, lt=None, alpha_list=None,
                           markers=None, fit=True, density=True, vline=True, ms=12, lw=4)
# ax.set_xlabel(r'$\Delta V_{HT}/v$')
print(f'mean values: {mu_list}')
print(f'stds: {sigma_list}')
ax.set_xlabel(r'$\Delta V_{T}/v$')
ax.set_ylabel(r'$p(\Delta V_{T}/v)$')
ax.text(0.3, h, f'$\Delta V_{{T}}<0$({n1/(n0+n1)*100:.02f}$\%$)')
ax.vlines(0, 0, h*1.1, 'k')
ax.set_ylim([0, h*1.1])
plt.tight_layout()
fig.savefig(f'{data_dir}/dV_T_distr.png')




n0 = len(dV_spc[dV_spc>=0])
n1 = len(dV_spc[dV_spc<0])
print(f'number of dV lager than 0 : {n0}')
print(f'number of dV smaller than 0 : {n1}')
np.savetxt(f'{data_dir}/deltaV_HT.csv', dV_spc)
fig, ax = plt.subplots(figsize=(10, 8))
data = [dV_spc]
nbins = [16]
label = [r'']
color = ['k']
x1 = [min(dV_spc)]
x2 = [max(dV_spc)]
_, _,  mu_list, sigma_list, h=plot_norm_distribution(ax, data, nbins, label, color, x1, x2, mcolor=None,  pset=None, lt=None, alpha_list=None,
                           markers=None, fit=True, density=True, vline=True, ms=12, lw=4)
# ax.set_xlabel(r'$\Delta V_{HT}/v$')
print(f'mean values: {mu_list}')
print(f'stds: {sigma_list}')
# ax.legend(['Data', 'Gaussian fit'])
ax.set_xlabel(r'$\Delta V_{HT}/v$')
ax.set_ylabel(r'$p(\Delta V_{HT}/v)$')
ax.text(0.3, h, f'$V_{{T}}>V_{{H}}$({n0/(n0+n1)*100:.02f}$\%$)')
ax.vlines(0, 0, h*1.1, 'k')
ax.set_ylim([0, h*1.1])
plt.tight_layout()
fig.savefig(f'{data_dir}/dV_HT_distr.png')
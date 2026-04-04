import numpy as np
import matplotlib.pyplot as plt
import os
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
current_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_path)
plt.subplots(figsize=(8, 7))
data = np.loadtxt(f'{save_path}/hist_dr.txt')
dt = np.loadtxt(f'{save_path}/dt.txt')*41
n, c = np.shape(data)
nh = n // 2
for i in range(nh):
    plt.plot(data[i*2, :], data[2*i+1, :], label=f'{int(dt[i])}', lw=3)
plt.legend(ncol=2, title=r'Time($\tau$)', frameon=False, handletextpad=0.5, borderaxespad=-0.1 )
plt.minorticks_on()
# plt.grid(which='major', color='gray', linewidth=0.8)
# plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
plt.axvline(x=0.8, color='r', linestyle='--', label='Vertical Line')
plt.xlabel(r'$\Delta r /\sigma$')
plt.ylabel('Probability density')
plt.ylim([1e-4, 20])
plt.xlim([0, 5])
plt.yscale('log')
plt.tight_layout()
current_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_path)
plt.savefig(f'{save_path}/FigS2a_hist_dr.jpg', dpi=600)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate as intp
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
current_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_path)

sisf = np.loadtxt(f'{save_path}/sisf.txt')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sisf[0, :]*41, sisf[1, :], 's-', label='2D HD')

##Calculate the $\alpha$ relaxation time
sisf_tg = 1/np.e
x_tgf = intp.interp1d(sisf[1, :], sisf[0, :]*41, kind="linear", bounds_error=True)
x_tg = x_tgf(sisf_tg)


ax.set_xscale('log')
ax.set_ylim([0, 1])
fig.tight_layout()
ax.legend()
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$F_{s}(\vec{k}, t)$')
ax.hlines(1/np.e, 0, 1e7, linestyle='dashed', color='k')
plt.tight_layout()
ax.minorticks_on()
ax.set_xticks([100, 1e3, 1e4, 1e5, 1e6, 1e7])
ax.text(50, 0.45, r'$1/e$')
ax.vlines(x_tg, 0, 1, colors='k', linestyles='dashdot')
x_tg_label = x_tg/(10**6)
ax.text(x_tg/1100, 1/np.e+0.1, rf'$\tau_{{\alpha}}={x_tg_label:.02f} \times 10^{{6}}$')
ax.set_xlim([40, 5e6])

# plt.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
plt.show()

fig.savefig(f'{save_path}/FigS6_sisf_HD.png')
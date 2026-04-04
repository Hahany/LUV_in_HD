import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import curve_fit
from hd2d_src.HopTrack.utils.path import get_sub_project_root
mpl.rcParams['text.usetex'] = True
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

def saturate_func(x, A, x0, beta, C):
    return A * (1 - np.exp(-(x / x0)**beta))+C

def get_deltaV(dirData, fix_rho, prefix):
    current_dir = dirData
    ave_folders = [name for name in os.listdir(current_dir)
                   if name.startswith(prefix)]  #and os.path.isdir(name)
    data_save = []
    R_list = []
    error = []
    std_out = []
    data_all = []
    for folder in ave_folders:
        Rc = float(folder.split('Rc_')[1])
        tail_file_path = os.path.join(current_dir, folder, 'local_density.txt')
        print(tail_file_path)
        all_file_path = os.path.join(current_dir, folder, 'entire_density.txt')
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

                ave_tail = np.mean(tail)
                if fix_rho:
                    ave_all = 0.8050357324284557
                else:
                    ave_all = np.mean(all)
                print(ave_tail)
                print(ave_all)
                deltaV = (ave_all - ave_tail) * (Rc * 2) ** 2
                data_save.append(deltaV)
                data_all.append(ave_all)
                R_list.append(Rc)
                import random
                v_all = (1-all) * (Rc * 2) ** 2
                v_tail = (1-np.array(tail))* (Rc * 2) ** 2
                #-----------------------(考虑协方差)-------------------------
                # if len(v_all) > len(v_tail):
                #     rand_A = random.sample(v_all.tolist(), len(v_tail))
                #     cov_AT = np.cov(rand_A, v_tail, ddof=1)
                #     std_T = np.std(v_tail, ddof=1)
                #     std_A = np.std(rand_A, ddof=1)
                #     std2 = np.sqrt(std_T ** 2 + std_A ** 2 - 2 * cov_AT[0, 1]) / np.sqrt(len(v_tail))
                # else:
                #     rand_T = random.sample(v_tail.tolist(), len(v_all.tolist()))
                #     cov_AT = np.cov(v_all, rand_T, ddof=1)
                #     std_T = np.std(rand_T, ddof=1)
                #     std_A = np.std(v_all, ddof=1)
                #     std2 = np.sqrt(std_T ** 2 + std_A ** 2 - 2 * cov_AT[0, 1]) / np.sqrt(len(v_all))
                # -----------------------(近似独立不考虑协方差)-------------------------
                std_A = np.std(v_all, ddof=1)
                std_T = np.std(v_tail, ddof=1)
                std2 = np.sqrt(std_A ** 2 / len(v_all) + std_T ** 2 / len(v_tail))
                std_out.append(np.sqrt(std_A ** 2 + std_T ** 2))
                error.append(std2)
    data = np.array([R_list, data_save, error, data_all, std_out])
    data = data.T
    sort = np.argsort(data[:,0])
    data = data[sort,:]
    print(sort)
    return data



def DeltaV_vs_Rc_3D():
    # Create a new figure and a 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    data_files = [name for name in os.listdir('./') if name.startswith('data_save') and os.path.isfile(name)]
    data_files = sorted(data_files)

    # Define a color map to differentiate each dataset with gradient colors
    colors = plt.cm.plasma(np.linspace(0.7, 0, len(data_files)))

    # Define different markers for each dataset
    markers = ['o', 's', '^', 'v', 'D', 'p']  # Add more if you have more datasets

    for idx, d in enumerate(data_files):
        label = d.split("_")
        l = float(label[2][0:-4])

        # Load data from file
        s = np.loadtxt(d, delimiter=',')

        # Plot the errorbar plot on the 3D axes with unique marker and color
        line, = ax.plot(s[:, 0], [l] * len(s), s[:, 1], linestyle='-', color=colors[idx],
                        marker=markers[idx % len(markers)],
                        markerfacecolor='none', markeredgecolor=colors[idx], label=fr'$\rho={l:.03f}$')

        # Add error bars with less prominent lines
        cap_width = 0.1
        for i in range(len(s)):
            ax.plot([s[i, 0], s[i, 0]], [l, l], [s[i, 1] - s[i, 2], s[i, 1] + s[i, 2]],
                    '-', color=colors[idx], alpha=0.5)  # Use light gray and semi-transparent lines for error bars

            ax.plot([s[i, 0] - cap_width, s[i, 0] + cap_width], [l, l], [s[i, 1] - s[i, 2], s[i, 1] - s[i, 2]],
                    '-', color=colors[idx], alpha=0.5)
            ax.plot([s[i, 0] - cap_width, s[i, 0] + cap_width], [l, l], [s[i, 1] + s[i, 2], s[i, 1] + s[i, 2]],
                    '-', color=colors[idx], alpha=0.5)

        verts = []
        min_z = 0
        zs = s[:, 1]
        xs = s[:, 0]
        for i in range(len(s)):
            x = xs[i]
            z_top = zs[i]
            verts.append((x, l, z_top))  # Top of the polygon

        # Close the polygon by connecting the last point to the first
        verts.append((xs[-1], l, min_z))
        verts.insert(0, (xs[0], l, min_z))

        poly = Poly3DCollection([verts], alpha=0.3)
        poly.set_facecolor(line.get_color())
        ax.add_collection3d(poly)

    # Set labels for the axes
    ax.set_xlabel(r'$R/\sigma$', fontsize=20, fontname='Times New Roman')
    ax.set_ylabel(r'$\rho$', fontsize=20, fontname='Times New Roman')
    ax.set_zlabel(r'$\Delta V/v_1$', fontsize=20, fontname='Times New Roman')

    # Adjust view angle for better visibility
    ax.view_init(elev=30, azim=135)  # You can change these values to find the best viewing angle

    # Place the legend outside of the plot area if needed
    # ax.legend(loc='best', bbox_to_anchor=(1.05, 1))
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='z', which='major', labelsize=16)
    # Show the plot and save it to a file
    plt.tight_layout()  # Adjust layout so that everything fits nicely
    plt.show()
    fig.savefig('DeltaV_vs_Rc_3D_distinct.svg', bbox_inches='tight', dpi=600)


def DeltaV_vs_Rc_2D(file_path, plot_all=False, fix_rho=False, xa=0, xb=100):
    fig, ax = plt.subplots(figsize=(8, 6))
    data = np.loadtxt(file_path)
    data = np.array(data)

    # fit
    p0 = [1.0, 3.0, 1.0, 0.2]  # 初始猜测参数 [A, x0, beta]
    R = data[xa:xb, 0]
    ave_dv = data[xa:xb, 1]
    ste_dv = data[xa:xb, -1]
    popt, pcov = curve_fit(saturate_func, R, ave_dv, p0=p0, sigma=ste_dv, absolute_sigma=True, maxfev=20000)

    A_fit, x0_fit, beta_fit, C_fit = popt
    A_err, x0_err, beta_err, C_err = np.sqrt(np.diag(pcov))

    # plot fit
    x_fit = np.linspace(min(R), max(R), 100)
    y_fit = saturate_func(x_fit, *popt)

    # ax.plot(data[xa:xb, 0], data[xa:xb, 1], color='c', lw=4, zorder=1)
    ax.errorbar(data[xa:xb, 0], data[xa:xb, 1], yerr=data[xa:xb, 2], color='c', fmt='o', marker=None, capsize=6, ecolor='b',
                markerfacecolor='b', markeredgecolor='b', zorder=0, lw=1, label=r'$\Delta \bar{V}_\mathrm{AT}/v$')
    plt.plot(x_fit, y_fit, 'r--', label=fr'${A_fit+C_fit:.02f}-{A_fit:.02f}e^{{-(R/{x0_fit:.2f})^{{{beta_fit:.2f}}}}}$', lw=2)
    # ax.plot(data[:,0], data[:, 1], '-sk')
    # yh = np.mean(data[9:17, 1])
    # ax.hlines(yh, data[9, 0], data[16, 0], color='k', linestyles='--',
    #           label=rf'$\Delta  \overline{{V}}/v_1={yh:.02f}$')

    ax.set_xlabel(r'$R$', fontsize=20)
    ax.set_ylabel(r'$\Delta \bar{{V}}_\mathrm{AT}/v$', fontsize=20)
    if plot_all:
        for fi in np.linspace(0.78, 0.80, 5):
            filename = f"data_save_{fi:.03f}.csv"
            data = np.loadtxt(filename, delimiter=',')
            ax.plot(data[:, 0], data[:, 1])
    ax.grid(False)
    # ax.hlines(y=0, xmin=data[xa, 0], xmax=data[xb, 0], linestyle='--')
    ax.legend(loc='lower right')
    ax.set_ylim([0.18, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    xma = 25
    x_insert = 6
    y_insert = np.interp(x_insert, x_fit, y_fit)
    ax.hlines(A_fit+C_fit, 5, 20, ls='--', color='k', lw=2)
    ax.text(5, 0.92, fr'Asymptotic value: {A_fit+C_fit:.02f}')
    # ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '1.0'])
    # ax.vlines(data[xma, 0], 0, data[xma, 1], ls='--', color='k')
    mask=data[:, 0] == 6
    ax.vlines(x_insert, 0, y_insert, ls='--', color='k')
    text = data[mask, 1].tolist()[0]
    ax.hlines(y_insert, 0, x_insert, ls='--', color='k')
    ax.text(3, 0.68, fr'{y_insert:.02f}')
    ax.set_xlim([1, 19])
    fig.tight_layout()
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.tick_params(which='major', top=True, right=True, direction='in')
    ax.tick_params(which='minor', top=True, right=True, direction='in')
    plt.show()
    current_path = os.path.abspath(__file__)
    save_fig = os.path.dirname(current_path)
    if fix_rho:
        fig.savefig(f"{save_fig}/Fig3_Dv_vs_Rc_2D_fix.png", bbox_inches='tight', dpi=600)
    else:
        fig.savefig(f"{save_fig}/Fig3_Dv_vs_Rc_2D_measure.png", bbox_inches='tight', dpi=600)

def rho_vs_Rc_(file_path, fix_rho=False, xa=0, xb=100):
    fig, ax = plt.subplots()
    data = np.loadtxt(file_path)
    data = np.array(data)
    ax.plot(data[xa:xb,0], data[xa:xb,3])
    plt.show()
    folder_path = os.path.dirname(file_path)
    if fix_rho:
        fig.savefig(f"rho_vs_R_fix.png", bbox_inches='tight', dpi=600)
    else:
        fig.savefig(f"rho_vs_R_measure.png", bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    #==========overlap=========================
    # data_path = '/home/xiaochu/Public/project-LUV/data/output/0805/AVE_dt_100000_rs_0.6_rh_0.8_Lst_0_Len2.00_nseg_1000'
    # deltaV = get_deltaV(data_path, fix_rho=False, prefix='Rc_')
    # np.savetxt(f'{data_path}/deltaV.csv', deltaV)
    # DeltaV_vs_Rc_2D(f'{data_path}/deltaV.csv', plot_all=False, fix_rho=False, xa=0, xb=34)
    # rho_vs_Rc_(f'{data_path}/deltaV.csv', xa=0, xb=34)
    #================no overlap================
    #=======0.805======
    data_path = get_sub_project_root() / "data/output/0805/AVE_dt_100000_rs_0.6_rh_0.8_Lst_0_Len2.00_nseg_1000"
    deltaV = get_deltaV(data_path, fix_rho=False, prefix='Rc_')
    np.savetxt(f'{data_path}/deltaV.csv', deltaV)
    DeltaV_vs_Rc_2D(f'{data_path}/deltaV.csv', plot_all=False, fix_rho=False, xa=0, xb=34)
    # rho_vs_Rc_(f'{data_path}/deltaV.csv', xa=0, xb=34)
    
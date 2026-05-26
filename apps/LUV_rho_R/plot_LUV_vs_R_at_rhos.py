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
                v_all = (1-np.array(all)) * (Rc * 2) ** 2
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


def DeltaV_vs_Rc_2D(dict, app_path, fix_rho=False, xa=0, xb=100):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors  = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple', 'saddlebrown']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    
    for i, key in enumerate(dict.keys()):
        color  = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        data = np.loadtxt(f'{app_path}/deltaV_{key}.csv')
        data = np.array(data)

        # fit
        p0 = [1.0, 3.0, 1.0, 0.2]
        R      = data[xa:xb, 0]
        ave_dv = data[xa:xb, 1]
        ste_dv = data[xa:xb, -1]
        popt, pcov = curve_fit(
            saturate_func, R, ave_dv,
            p0=p0, sigma=ste_dv, absolute_sigma=True, maxfev=20000
        )

        A_fit, x0_fit, beta_fit, C_fit = popt
        A_err, x0_err, beta_err, C_err = np.sqrt(np.diag(pcov))

        # fit curve
        x_fit = np.linspace(min(R), max(R), 100)
        y_fit = saturate_func(x_fit, *popt)
        phi_val = int(key) / 1000
        if phi_val <= 0.785:
            fit_label = fr'$\phi={phi_val:.3f}$, non-saturating'
        else:
            fit_label = fr'$\phi={phi_val:.3f},\ {A_fit+C_fit:.02f}-{A_fit:.02f}e^{{-(R/{x0_fit:.2f})^{{{beta_fit:.2f}}}}}$'
        # plot data points (no legend entry)
        # plot data points (no legend entry)
        ax.errorbar(
            data[xa:xb, 0], data[xa:xb, 1], yerr=data[xa:xb, 2],
            fmt=marker, color=color, ecolor=color,
            capsize=6, markerfacecolor='none', markeredgecolor=color,
            zorder=2, lw=1
        )

        # plot fit line
        ax.plot(
            x_fit, y_fit,
            linestyle='--', color=color, lw=2,
            marker=marker, markevery=10, markersize=6,
            markerfacecolor='none', markeredgecolor=color,
            label=fit_label
        )

    ax.set_xlabel(r'$R$', fontsize=20)
    ax.set_ylabel(r'$\Delta \bar{{V}}_\mathrm{AT}/v$', fontsize=20)
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=20)
    ax.set_ylim([0.18, 1.4])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_xlim([1, 19])
    fig.tight_layout()
    ax.minorticks_on()
    ax.tick_params(which='both',  width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.tick_params(which='major', top=True, right=True, direction='in')
    ax.tick_params(which='minor', top=True, right=True, direction='in')
    plt.show()
    current_path = os.path.abspath(__file__)
    save_fig = os.path.dirname(current_path)
    if fix_rho:
        fig.savefig(f"{save_fig}/Fig3_Dv_vs_Rc_2D_fix.png",     bbox_inches='tight', dpi=600)
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
    app_path = '/home/xiaochu/Public/LUV_in_HD/apps/LUV_rho_R'
    dict = {'0780': '/home/xiaochu/Public/LUV_in_HD/data/output/0780/no_overlap/rh_0.8_rs_0.6_Lst_2_Len2.00_nseg_100',
            '0785': '/home/xiaochu/Public/LUV_in_HD/data/output/0785/no_overlap/rh_0.8_rs_0.6_Lst_2_Len2.00_nseg_100',
            # '0790': '/home/xiaochu/Public/LUV_in_HD/data/output/0790/no_overlap/rh_0.8_rs_0.6_Lst_2_Len2.00_nseg_100',
            '0795': '/home/xiaochu/Public/LUV_in_HD/data/output/0795/no_overlap/rh_0.8_rs_0.6_Lst_2_Len2.00_nseg_1000',
            '0800': '/home/xiaochu/Public/LUV_in_HD/data/output/0800/no_overlap/rh_0.8_rs_0.6_Lst_2_Len2.00_nseg_1000',
            '0805': '/home/xiaochu/Public/LUV_in_HD/data/output/0805/no_overlap/rh_0.8_rs_0.6_Lst_2_Len2.00_nseg_1000'}
    for key, value in dict.items():
        data_path = value
        deltaV = get_deltaV(data_path, fix_rho=False, prefix='Rc_')
        np.savetxt(f'{app_path}/deltaV_{key}.csv', deltaV)
    DeltaV_vs_Rc_2D(dict, app_path=app_path, fix_rho=False, xa=0, xb=34)

        
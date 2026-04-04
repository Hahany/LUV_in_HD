import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from natsort import natsorted
from hd2d_src.HopTrack.utils.path import get_sub_project_root
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
def get_deltaV(dirData, fix_rho, prefix):
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



                ave_tail = np.mean(tail)
                ave_head = np.mean(head)
                if fix_rho:
                    ave_all = 0.8050357324284557
                else:
                    ave_all = np.mean(all)
                print(f'mean tail is {ave_tail}')
                print(f'mean all is {ave_all}')
                deltaV = (ave_head - ave_tail) * (Rc * 2) ** 2
                data_save.append(deltaV)
                data_all.append(ave_all)
                R_list.append(Rc)
                import random
                v_head = (1-np.array(head)) * (Rc * 2) ** 2
                v_tail = (1-np.array(tail))* (Rc * 2) ** 2
                #-----------------------(考虑协方差)-------------------------
                cov_HT = np.cov(v_head, v_tail, ddof=1)
                std_T = np.std(v_tail, ddof=1)
                std_H = np.std(v_head, ddof=1)
                std2 = np.sqrt(std_T ** 2 + std_H ** 2 - 2 * cov_HT[0, 1]) / np.sqrt(len(v_tail))
                error.append(std2)
    data = np.array([R_list, data_save, error, data_all])
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
    fig, ax = plt.subplots(figsize=(10, 8))
    data = np.loadtxt(file_path)
    data = np.array(data)

    # ax.plot(data[xa:xb, 0], data[xa:xb, 1], color='c', lw=4, zorder=1)
    ax.errorbar(data[xa:xb, 0], data[xa:xb, 1], yerr=data[xa:xb, 2], color='c', fmt='o-', markersize=8, marker=None, capsize=6, ecolor='b',
                markerfacecolor='b', markeredgecolor='k', zorder=2, lw=2, label=r'$\Delta \bar{{V}}_\mathrm{{HT}}$')
    # ax.plot(data[:,0], data[:, 1], '-sk')
    # yh = np.mean(data[9:17, 1])
    # ax.hlines(yh, data[9, 0], data[16, 0], color='k', linestyles='--',
    #           label=rf'$\Delta  \overline{{V}}/v_1={yh:.02f}$')
    # ax.legend(loc='upper left')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$\Delta \bar{{V}}_\mathrm{{HT}}/v$')
    if plot_all:
        for fi in np.linspace(0.78, 0.80, 5):
            filename = f"data_save_{fi:.03f}.csv"
            data = np.loadtxt(filename, delimiter=',')
            ax.plot(data[:, 0], data[:, 1])
    ax.grid(False)
    ax.hlines(y=0, xmin=data[xa, 0], xmax=data[-1, 0], linestyle='--')
    # ax.legend()
    # ax.set_ylim([0.18, 1.00])
    # ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # xma = 25
    # ax.hlines(np.mean(data[xma:xb, 1]), data[xma, 0], data[xb-1, 0], ls='-', color='k', lw=2)
    # ax.text(data[xma, 0]+1, 0.8, fr'{np.mean(data[xma:xb, 1]):.02f}')
    # ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '1.0'])
    # ax.vlines(data[xma, 0], 0, data[xma, 1], ls='--', color='k')
    # mask=data[:, 0] == 6
    # ax.vlines(data[mask,0], 0, data[mask, 1], ls='--', color='k')
    # text = data[mask, 1].tolist()[0]
    # ax.hlines(text, 1, data[mask, 0], ls='--', color='k')
    # ax.text(2, 0.7, fr'{text:.02f}$v$')
    # ax.set_xlim([1, 19])
    fig.tight_layout()
    ax.minorticks_on()
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)
    ax.tick_params(which='minor', length=4)
    ax.tick_params(which='major', top=True, right=True, direction='in')
    ax.tick_params(which='minor', top=True, right=True, direction='in')
    ax.set_xlim([1.9, 8.1])
    ax.set_ylim([0, 0.6])
    plt.show()

    if fix_rho:
        fig.savefig(f"fix_{figname}", bbox_inches='tight', dpi=600)
    else:
        fig.savefig(f"{figname}", bbox_inches='tight', dpi=600)

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
        fig.savefig(f"rho_vs_R_measure_delta_t_200.png", bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    project_root = get_sub_project_root()
    current_path = os.path.abspath(__file__)
    data_dir = os.path.dirname(current_path)
    data_path = f'{project_root}/data/output/0805/DHT_ge_2R_nseg_1000_time_shifting'
    deltaV = get_deltaV(data_path, fix_rho=False, prefix='rh_0.8_rs_0.6_Lst_0_Len')
    np.savetxt(f'{data_path}/deltaV.csv', deltaV)
    n, _ = np.shape(deltaV)
    figname = f'{data_dir}/FigS8a.png'
    DeltaV_vs_Rc_2D(f'{data_path}/deltaV.csv', plot_all=False, fix_rho=False, xa=0, xb=n-4)

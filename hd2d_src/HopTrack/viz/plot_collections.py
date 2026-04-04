import numpy as np
import matplotlib.cm as cm
from matplotlib.collections import LineCollection

#plot trajectory by using the line collections method
def plot_collections(ax, mask, CalRegion, ts0, ts1, ts, ti, title):
    ParticlesInRegion = CalRegion[ts0, mask, :]
    cmap = cm.get_cmap('rainbow')
    cmap = cmap.reversed()
    line_collections = []
    glc = []
    colors = cmap(np.linspace(0, 1, ts))
    colors = colors[ts0:ts1]
    for i in range(len(ParticlesInRegion)):
        p_id_index = np.where(CalRegion[0, :, 0] == ParticlesInRegion[i, 0])
        cline = np.zeros((int(ts1 - ts0), 2, 2))
        cline[:, 0, :] = CalRegion[ts0:ts1, p_id_index, 2:4]
        cline[:, 1, :] = CalRegion[int(ts0 + 1): ts1 + 1, p_id_index, 2:4]
        line_collections.append(LineCollection(cline, linewidths=1, color=colors, zorder=5))
    for lc in line_collections:
        ax.add_collection(lc)
    ax.set_aspect(1)
    ax.set_title(f'{title} No.{ti} config')
    ax.scatter(ParticlesInRegion[:, 2], ParticlesInRegion[:, 3], c='r',
               s=0.5)  # , vmin=0, vmax=1)
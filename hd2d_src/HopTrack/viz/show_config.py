import matplotlib.pyplot as plt
import numpy as np
def show_config(frames, id_list, r, savefile):
    fig0, ax0 = plt.subplots()
    N, n = frames.shape
    for i in range(N):
        if i in id_list:
            ax0.plot(frames[i, 2], frames[i, 3], 'go')

        circle = plt.Circle((frames[i,2], frames[i, 3]), r[int(frames[i, 0]-1)], color='k', fill=False, lw=0.5)
        ax0.add_patch(circle)
    plt.axis('equal')

    plt.savefig(f'{savefile}', dpi=900)



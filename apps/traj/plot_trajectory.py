import os
import sys
from hd2d_src.HopTrack.utils.path import get_sub_project_root
from matplotlib import pyplot as plt
import time
import os
from hd2d_src.HopTrack.utils import *
##############<< start plot setting parameters >> ################
project_root = get_sub_project_root()
obj_path = f'{project_root}/data/input/0805/a_withstring.obj'
current_path = os.path.abspath(__file__)
savepath = os.path.dirname(current_path)
if not os.path.exists(savepath):
    os.makedirs(savepath)
###############<< end >>################
def trj(mode, dt, nseg):
    t0  = 0
    starttime = time.time()
    modes=['soley interval', 'total']
    mode=modes[mode]
    with open(f"{obj_path}", "rb") as f:
        custom_unpickler = CustomUnpickler(f)
        a = custom_unpickler.load()
    t1 = a.tmax
    if mode == 'soley interval':
        start = 0
        end =  start + dt - 1
        print(f'plotting {start}, {end}')
        a.setduration(start, end)
        a.unwrap3()
        fig, ax = plt.subplots()
    #   a.coarsening(nseg, NoCoarsenOnTopandTail=False)
    #     a.chooseWithoutCoarsening(nseg)
        # a.hop(0.8)
        a.showdisp(fig, ax, t_start=start, t_end=end, showradii=False,
                    lw=1, ms=4, nodot=True, showpid=False, showforcepid=0,showforcedid=12, crange=[0, nseg-1])
        # ax.set_xlim([50, 70])
        # ax.set_ylim([60, 80])
        fig.savefig(f'{savepath}/FigS3a.png', dpi=600)
        plt.close(fig)
    elif mode == 'total':
        fig, ax = plt.subplots()
        a.unwrap3()

        # a.hops(0.8)
        t_start = 0
        t_end = nseg-1
        a.chooseWithoutCoarsening(nseg)
        a.showdisp(fig, ax, t_start=t_start, t_end=t_end, showradii=False, lw=1, ms=4, nodot=True, showpid=False,
                   showforcepid=0,showforcedid=12, crange=[0, 99])
        # ax.tight_layout()
        fig.savefig(f'{savepath}/FigS3b.png', dpi=600)
        endtime = time.time()
        print(f'cost {endtime - starttime} seconds')
if __name__ == '__main__':
    # trj(1, 1000, 100)
    trj(0, 10, 10)
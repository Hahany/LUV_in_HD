import os.path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from hd2d_src.HopTrack.utils.CustomUnpickler import CustomUnpickler
from hd2d_src.HopTrack.core.Local_Density_Calculator import LD_one_particle
import hd2d_src.HopTrack.core.Local_Density_Calculator as LDC
from hd2d_src.HopTrack.core.Local_Density_Calculator import show_heterogeneity
from hd2d_src.HopTrack.utils.path import get_sub_project_root
from hd2d_src.HopTrack.viz import show_config

def plot_tracj(b, start_time, end_time, color, region):
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    b.showdisp(ax, lw=1, ms=1.5, nodot=True, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False)
def plot_string(b, start_time, end_time, region, savepath):
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    b.chooseWithoutCoarsening(2)
    mask = []
    frames = b.frames[0, :, :]
    r = b.radii*0.95
    show_config(frames, mask, r, savepath)
    plt.xticks([])
    plt.yticks([])
    plt.box(on=True)





    # ax.set_xlim([b.frames[0, b.connected_components[0][-1], 2] - Rc - 1,
    #              b.frames[0, b.connected_components[0][-1], 2] + Rc + 1])
    # ax.set_ylim([b.frames[0, b.connected_components[0][-1], 3] - Rc - 1,
    #              b.frames[0, b.connected_components[0][-1], 3] + Rc + 1])
    # fig.savefig(f"{savepath}_head.svg", dpi=600)



if __name__ == "__main__":
    proj_path = str(get_sub_project_root())
    obj_path = str(get_sub_project_root() / 'data/input/0805/a_AVE_normalized.obj')
    # b.cal_area_of_fragments = ht.Particles.cal_area_of_fragments.__get__(b)
    rh = 0.8
    rs = 0.6
    dr_ht = 2
    stlength = 0
    start_time = 69500   #62448-10 #65760
    end_time = start_time+100 #32 #14
    if not os.path.exists(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj'):
        # initialize the data reading, this will save time to reload the data
        with open(obj_path, "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            b = custom_unpickler.load()
        b.setduration(start_time, end_time)
        b.frames0 = []
        with open(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj', 'wb') as f:
            pickle.dump(b, f)
        exit(2)
    else:
        with open(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj', "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            b = custom_unpickler.load()
        b.frames=b.frames[0:-1:2, :, :]
        start_time = 0#40  # 62448-10 #65760
        end_time = start_time + 100 # 20  # 32 #14
        start_time = 20
        end_time = start_time + 11
    mean_radius = np.min(b.radii)
    print(mean_radius)
    Rc = 6*mean_radius
    # b.shiftbox(0, 20)
    # b.pbc()
    # b.unwrap3()
    region = [23, 47, 55, 75] #[12, 25, 10, 23] #[15, 28, 20, 33]
    current_path = os.path.abspath(__file__)
    save_dir = os.path.dirname(current_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Temporary plot for V head middle tail
    start_time = 69500
    with open(f'{proj_path}/data/input/0805/a_cut_{start_time}.obj', "rb") as f:
        custom_unpickler = CustomUnpickler(f)
        b = custom_unpickler.load()
    start_time = 0#40  # 62448-10 #65760
    end_time = start_time + 100 # 20  # 32 #14
    plot_string(b, start_time, end_time, region, savepath=f'{save_dir}/1.png')





#
    plt.show()


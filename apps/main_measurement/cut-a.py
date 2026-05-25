import pickle
import matplotlib as mpl
import os
import sys
mpl.rcParams['text.usetex'] = True
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from hd2d_src.HopTrack.utils import *
font_prop = FontProperties(family='Times New Roman', size=20)
font_path = '/home/xiaochu/.fonts/times.ttf'
fm.fontManager.addfont(font_path)

start_time = 0
for end_time in [500, 2000, 10000]:
    with open("/home/xiaochu/Public/LUV_in_HD/data/input/0805/a_AVE_normalized.obj", "rb") as f:
        custom_unpickler = CustomUnpickler(f)
        b = custom_unpickler.load()
    b.setduration(start_time, end_time)
    b.chooseWithoutCoarsening(nchoose=2)
    b.frames0 = []
    with open(f'/home/xiaochu/Public/LUV_in_HD/data/input/0805/a_cut_{start_time}_{end_time}.obj', 'wb') as f:
        pickle.dump(b, f)
exit(2)
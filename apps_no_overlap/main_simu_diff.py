import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_hopped_no_overlap
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_unhopped
from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator
from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator_Nooverlap
import matplotlib.pyplot as plt
import time
import pickle
import resource
from hd2d_src.HopTrack.utils import  CustomUnpickler

#trace string


def trace_and_measure(obj_a_path, nseg, t1, dt, Rc, rh, rs, Lst, dht, savepath, childfolder, savefig=False):
    #rho------packing fraction
    #nseg-----selected frame number
    #dt-------time duration
    #Rc-------measurement radius
    #rh-------hop threshold
    #rs-------connected threshold
    #Lst------length of string
    #dht------distance between string head and tail
    #ri-------insert threshold at the string ends


    # calculate local density
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    distr_save_path = f'{savepath}/{childfolder}'
    if not os.path.exists(distr_save_path):
        os.makedirs(distr_save_path)


    if not os.path.exists(f'{savepath}/a_withstring.obj'):
        with open(obj_a_path, "rb") as f:
            custom_unpickler = CustomUnpickler(f)
            a = custom_unpickler.load()
        a.chooseWithoutCoarsening(nseg)
        a.unwrap3()
        a.frames0 = []
        a.findstring(HopThreshold=rh, ConnectThreshold=rs, microstring=True, ignoreLoop=False, stlength=Lst,
                 dr_ht=dht, hop_whole=False)
        with open(f'{savepath}/a_withstring.obj', 'wb') as f:
            pickle.dump(a, f)
    else:
        with open(f'{savepath}/a_withstring.obj', 'rb') as f:
            a = pickle.load(f)
    print(f'string number is {len(a.connected_components)}')
    print(f'box size is {a.L}')
    print(f'minimal particle size is {min(a.radii)}')
    if len(a.connected_components) != 0:
        if savefig:
            for si, s in enumerate(a.connected_components):
                fig, ax = plt.subplots()
                a.showstring(ax, nodot=False, showid=False, select=True, sid=si,
                             findquasivoid=True, size=0.5, show_length_distribusion=False, showtraj=False, stringID=None,
                             SSC=None, WL=False, show_localdensity_region=False, Rc=2.0, mode=1)
                a.showdisp(fig, ax, t_start=a.starend_of_string[si][0], t_end=a.starend_of_string[si][1], sid=si, lw=1,
                           ms=2, nodot=True, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
                           colorbar=False, showradii=False)
                savefile = f'{savepath}/d{dt}_{t1:05d}'
                fig.savefig(f'{savefile}_Sid_{si:04d}_ts_{a.starend_of_string[si][0]}_tn_{a.starend_of_string[si][1]}.png', dpi=300)
                plt.close(fig)

        ld_str, ld_all = a.showdensity_simu_diff(Rc, dr_ht=dht)

    else:
        print('No string is found!!!')
    np.savetxt(f'{distr_save_path}/ld_str.txt', ld_str.T, delimiter='\t', fmt='%.6f')
    np.savetxt(f"{distr_save_path}/ld_all.txt", ld_all.T, delimiter='\t', fmt='%.6f')



def print_memory_usage():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == 'posix':
        print(f"Memory usage: {mem / 1024:.2f} MB")
    else:
        print(f"Memory usage: {mem / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    # start options
    start = time.time()
    Rc = float(sys.argv[1])  ### measurement region size
    rh = 0.8        # hop criteria
    rs = 0.6        #connect criteria
    Lst = 0         # length of sting , particle number
    ri = 0          # Minimum distance between other particles and the string tail particle
    t0 = 0          # first frame, befault from 0
    t1 = int(sys.argv[2]) # total frame number
    dt = int(t1-t0)
    nseg = int(sys.argv[3])    # number of frames after coarsening
    dht = 2  # the distance between head and tail , previous value: 2*Rc
    rho = float(sys.argv[4])   # density
    s_rho = int(1000*rho)
    data_dir = f'../data/input/{s_rho:04d}'
    obj_a_path = f'{data_dir}/a_AVE_normalized.obj'
    save_path= f"../data/output/{s_rho:04d}/no_overlap_simu_diff/rh_{rh:.01f}_rs_{rs:.01f}_Lst_{Lst}_Len{dht:.02f}_nseg_{nseg}"
    child_folder = f'Rc_{Rc:.02f}'
    trace_and_measure(obj_a_path, nseg=nseg, t1=t1, dt=dt, Rc=Rc,
                      rh=rh, rs=rs, Lst=Lst, dht=dht, savepath=save_path,
                      childfolder=child_folder, savefig=False)
    end = time.time()
    print_memory_usage()
    print(f'totally cost {end - start}s')
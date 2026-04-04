import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_hopped
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_unhopped
from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator
import matplotlib.pyplot as plt
import time
import pickle
import resource
from hd2d_src.HopTrack.utils import *

#trace string


def trace_and_measure(obj_a_path, rho, nseg, t1, dt, Rc, rh, rs, Lst, dht, ri, savepath, childfolder, savefig=False):
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

        if t1 > a.tmax:
            t1 = a.tmax
    else:
        with open(f'{savepath}/a_withstring.obj', 'rb') as f:
            a = pickle.load(f)
        if t1 > a.tmax:
            t1 = a.tmax
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


        local_tail_ini = []
        local_head_ini = []
        local_tail_final = []
        local_head_final = []
        ld_instring = []
        sl = []
        entire_density = []

        stringlength = a.stringlength
        sl.extend(stringlength)
        init, finalt, inih, finalh, time_select = a.showdensity(Rc, ri, rho, figshow=False,
                                                                savefile=f'{savepath}/d{dt}_{t1:05d}', mode=0,
                                                                Length_of_string=Lst, dr_ht=dht)

        #calculate the LUV of the middle particle in the string--------------set:1---------------
        # ld_string = a.showdensity(Rc, ri, rho, figshow=False, savefile=None, mode=1, Length_of_string=Lst,
        #                           crt_in=False,
        #                           choose_middle=True, dr_ht=ri)

        local_tail_ini.extend(init)
        local_tail_final.extend(finalt)
        local_head_ini.extend(inih)
        local_head_final.extend(finalh)
        e_d = Local_Density_Calculator(a, Rc, [0])  # only measure the first frame
        entire_density.extend(e_d)
        # ld_instring.extend(ld_string)

        local_density = np.array([local_tail_ini, local_tail_final, local_head_ini, local_head_final])
        np.savetxt(f'{distr_save_path}/entire_density.txt', entire_density, delimiter='\t', fmt='%.6f')
        np.savetxt(f"{distr_save_path}/local_density.txt", local_density.T, delimiter='\t', fmt='%.6f')
        # np.savetxt(f"{distr_save_path}/local_density_in_string.txt", ld_instring, delimiter='\t', fmt='%.6f')

        # # hopped particle----------------------------------------set:2----------------------
        # # a.setduration(t0, t1-1)
        # a.chooseWithoutCoarsening(2)
        # ld_hp = ld_hopped(a, 2, Rc, rh)
        # ld_hp = np.array(ld_hp)
        # np.savetxt(f"{distr_save_path}/ld_hopped.txt",
        #            ld_hp.T, delimiter='\t', fmt='%.6f')

        # # unhopped----------------------------------------------set:3----------------------
        # # a.setduration(t0, t1-1)
        # a.chooseWithoutCoarsening(2)
        # ld_uhp = ld_unhopped(a, 2, Rc, rh)
        # ld_uhp = np.array(ld_uhp)
        # np.savetxt(f"{distr_save_path}/ld_unhopped.txt",
        #            ld_uhp.T, delimiter='\t', fmt='%.6f')

    else:
        print('stringlenth is None')


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
    nseg = int(sys.argv[3])    # number of frames after coarsening
    dt = int((t1-t0)/nseg)
    dht = 10  # the distance between head and tail , previous value: 2*Rc
    rho = float(sys.argv[4])   # density
    s_rho = int(1000*rho)
    data_dir = f'../data/input/{s_rho:04d}'
    obj_a_path = f'{data_dir}/a_AVE_normalized.obj'
    save_path= f"/home/xiaochu/Public/project-LUV/data/output/{s_rho:04d}/DHT_ge_{dht:.02f}_Nf_{nseg}/rh_{rh:.01f}_rs_{rs:.01f}_Lst_{Lst}_Len{dht:.02f}_nseg_{nseg}"
    child_folder = f'Rc_{Rc:.02f}'
    trace_and_measure(obj_a_path, rho=rho, nseg=nseg, t1=t1, dt=dt, Rc=Rc,
                      rh=rh, rs=rs, Lst=Lst, dht=dht, ri=ri, savepath=save_path,
                      childfolder=child_folder, savefig=False)
    end = time.time()
    print(f'totally cost {end - start}s')
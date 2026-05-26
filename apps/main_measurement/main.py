import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
from hd2d_src.HopTrack.core.Local_Density_Calculator import ld_hopped
from hd2d_src.HopTrack.core.Local_Density_Calculator import Local_Density_Calculator
import time
import pickle
import resource
from hd2d_src.HopTrack.utils import CustomUnpickler
from multiprocessing import Pool
import fcntl
from hd2d_src.HopTrack.core.share.Deduplicate import *


def prepare_string_obj(obj_a_path, savepath, t1, nseg, rh, rs, Lst, dht):
    os.makedirs(savepath, exist_ok=True)
    lock_path = f'{savepath}/a_withstring.lock'
    obj_path = f'{savepath}/a_withstring.obj'
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not os.path.exists(obj_path):
                with open(obj_a_path, "rb") as f:
                    a = CustomUnpickler(f).load()
                a.setduration(begin=0, finish=t1)
                a.chooseWithoutCoarsening(nseg)
                a.unwrap_vec()
                a.frames0 = []
                a.findstring(HopThreshold=rh, ConnectThreshold=rs, microstring=True,
                             ignoreLoop=True, stlength=Lst, dr_ht=dht, hop_whole=False)
                Deduplicate(a)
                deepDeduplicate(a)
                with open(obj_path, 'wb') as f:
                    pickle.dump(a, f)
                print('a_withstring.obj generated')
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def trace_and_measure(distr_save_path, rho, Rc, rh, Lst, dht, ri, savepath, savefile=None):
    with open(f'{savepath}/a_withstring.obj', 'rb') as f:
        a = pickle.load(f)
    print(f'string number is {len(a.connected_components)}')
    print(f'box size is {a.L}')
    print(f'minimal particle size is {min(a.radii)}')

    local_tail_ini = []
    local_tail_final = []
    local_head_ini = []
    local_head_final = []
    entire_density = []

    if len(a.connected_components) != 0:
        init, finalt, inih, finalh, time_select = a.showdensity(
            Rc, ri, rho, savefile=savefile, figshow=False,
            pid=None, select=False, mode=0, Length_of_string=Lst,
            crt_in=False, choose_middle=False, dr_ht=dht)

        local_tail_ini.extend(init)
        local_tail_final.extend(finalt)
        local_head_ini.extend(inih)
        local_head_final.extend(finalh)
        e_d = Local_Density_Calculator(a, Rc, [0])
        entire_density.extend(e_d)
    else:
        print('No string is found!!!')

    local_density = np.array([local_tail_ini, local_tail_final, local_head_ini, local_head_final])
    np.savetxt(f'{distr_save_path}/entire_density.txt', entire_density, delimiter='\t', fmt='%.6f')
    np.savetxt(f"{distr_save_path}/local_density.txt", local_density.T, delimiter='\t', fmt='%.6f')

    a.chooseWithoutCoarsening(2)
    ld_hp = ld_hopped(a, 2, Rc, rh)
    ld_hp = np.array(ld_hp)
    np.savetxt(f"{distr_save_path}/ld_hopped.txt", ld_hp.T, delimiter='\t', fmt='%.6f')


def worker(args):
    Rc, kwargs = args
    distr_save_path = f"{kwargs['savepath']}/Rc_{Rc:.02f}"
    os.makedirs(distr_save_path, exist_ok=True)
    trace_and_measure(distr_save_path=distr_save_path, Rc=Rc, **kwargs)


def print_memory_usage():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == 'posix':
        print(f"Memory usage: {mem / 1024:.2f} MB")
    else:
        print(f"Memory usage: {mem / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    start = time.time()
    rh = 0.8
    rs = 0.6
    Lst = 2
    ri = 0
    t1 = int(sys.argv[1])
    nseg = int(sys.argv[2])
    rho = float(sys.argv[3])
    dht = 2
    s_rho = int(1000 * rho)
    obj_a_path = f'/home/xiaochu/Public/project-LUV/data/input/{s_rho:04d}/a_AVE_normalized.obj'
    save_path = f"/home/xiaochu/Public/project-LUV/data/output/{s_rho:04d}/overlap/rh_{rh:.01f}_rs_{rs:.01f}_Lst_{Lst}_Len{dht:.02f}_nseg_{nseg}"

    prepare_string_obj(obj_a_path, save_path, t1, nseg, rh, rs, Lst, dht)

    Rc_list = [2.0 + i * 0.5 for i in range(37)]
    shared_kwargs = dict(rho=rho, rh=rh, Lst=Lst, dht=dht, ri=ri, savepath=save_path)

    N_JOBS = 40
    with Pool(processes=N_JOBS) as pool:
        pool.map(worker, [(Rc, shared_kwargs) for Rc in Rc_list])

    print_memory_usage()
    print(f'totally cost {time.time() - start:.1f}s')
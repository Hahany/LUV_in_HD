import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from hd2d_src.HopTrack.core.share.distPBC2D import distPBC2D

#find string in glass

def findstring(glass, HopThreshold, ConnectThreshold, microstring=True, ignoreLoop=False,
               stlength=0, dr_ht=-1, hop_whole=False, silent=True, Dt=1, dt=1):
    # Dt: 每隔多少帧检查一次，默认1（逐帧）
    # dt: 判断hop的时间窗口长度，默认1（相邻帧）
    # deep first search, output string and level
    # hop_whole : check a particle is hopped or not at the total simulation time
    # stlength is the length of string
    # the distance between string head and tail
    if dr_ht < 0:
        dr_ht = 2
        print('dr_ht is set to 2 for default value when tracing string, but will be soly set to 2Rc when calculating LUV......')
    class hop:
        def __init__(glass, id):
            glass.id = id
            glass.next = None
            glass.readed = None

    class String:
        def __init__(glass, particle_id, time):
            glass.id = particle_id
            glass.time = time

    def dfs(hop):
        level = 1
        chain = [hop.id]
        if hop.next:
            if not hop.readed:
                hop.readed = 1
                le, cha = dfs(hop.next)
                level = level + le
                chain.extend(cha)
        return level, chain

    if hop_whole:
        T, N, _ = np.shape(glass.frames)
        DispEachFrame = np.linalg.norm(glass.frames[dt:, :, 2:4] - glass.frames[0, :, 2:4],
                                       axis=2)
        HopList = np.where(DispEachFrame >= HopThreshold)
    else:
        T, N, _ = np.shape(glass.frames)
        # 只在 0, Dt, 2Dt, ... 这些帧上计算，每次跨越 dt 帧的位移
        selected_frames = np.arange(0, T - dt, Dt)

        hop_times = []
        hop_particles = []
        for t in selected_frames:
            disp = np.linalg.norm(
                glass.frames[t, :, 2:4] - glass.frames[t + dt, :, 2:4], axis=1)
            particles = np.where(disp >= HopThreshold)[0]
            hop_times.extend([t] * len(particles))
            hop_particles.extend(particles.tolist())
        HopList = (np.array(hop_times), np.array(hop_particles))



    if len(HopList[0]) == 0:
        print('No hops found')
        return 0
    TimeList = HopList[0]  # get frames when hop
    norepeatTimeList = list(OrderedDict.fromkeys(TimeList))
    TimeList = norepeatTimeList  # update hop time list
    strings = []
    T = len(TimeList)
    i_t = 0
    while i_t < T:
        t1_frame = TimeList[i_t]
        t2_frame = t1_frame + dt  # 原来是 t1_frame + 1，现在改为 t1_frame + dt
        print(f'processing frame {t2_frame}', end='\r')
        string = []
        particlelist = HopList[1][np.where(HopList[0] == TimeList[i_t])]
        Pn = len(particlelist)
        n = 0

        # find all two-hops connected system
        while n < Pn:
            string.append([particlelist[n]])
            ini_pn = glass.frames[t1_frame, particlelist[n], 2:4]
            pm = np.delete(particlelist, n)
            if not pm.any():
                n += 1
                continue
            # print(f"t2_frame= {t2_frame}, pm={pm}, {np.shape(glass.frames)} \n")
            final_pm = glass.frames[t2_frame, pm, 2:4]
            dis_matrix = ini_pn - final_pm
            dis_matrix = distPBC2D(dis_matrix, glass.L[0], glass.L[1])
            dis_connect = np.linalg.norm(dis_matrix, axis=1)
            min_index = np.argmin(dis_connect)
            dis_b = dis_connect[min_index]
            if min_index >= n:
                min_index = min_index + 1  # initial position in glass.frames
            parcon = particlelist[min_index]
            if dis_b <= ConnectThreshold:
                string[-1].append(parcon)
            n += 1

        # find hop branch
        l = len(string)
        string_out = string.copy()
        for i in range(l):
            elem1 = string[i]
            for j in range(i + 1, l):
                elem2 = string[j]
                if len(elem1) == 2 and len(elem2) == 2:
                    if elem1[1] == elem2[1]:  # if one hop connect with two different hops
                        va = distPBC2D(glass.frames[t1_frame, elem1[0], 2:4] - glass.frames[t2_frame, elem1[1], 2:4],
                                       glass.L[0], glass.L[1])
                        vb = distPBC2D(glass.frames[t1_frame, elem2[0], 2:4] - glass.frames[t2_frame, elem2[1], 2:4],
                                       glass.L[0], glass.L[1])
                        dis_a = np.linalg.norm(va)
                        dis_b = np.linalg.norm(vb)
                        if dis_a > dis_b:  # deal with hop branch
                            string_out[i] = string_out[i][0:-1]

                        else:
                            string_out[j] = string_out[j][0:-1]

                    elif elem2[1] == elem1[0] and elem2[0] == elem1[1]:
                        string_out[i] = []
                        string_out[j] = []

                elif len(elem1) == 1:
                    if elem2[-1] == elem1[0]:
                        string_out[i] = []
        string_out = [ele for ele in string_out if ele]
        string = string_out.copy()

        # find connective string
        plist = [paii for pai in string for paii in pai]
        plist = list(set(plist))
        hops = [None] * len(plist)
        for i in range(len(plist)):
            hops[i] = hop(plist[i])
        if not silent:
            print(f'string particle id is {string}')
            print(f'string particle id set is {plist}')
        for i in range(len(string)):
            if len(string[i]) == 2:
                plist = np.array(plist)  #保证后面的判断不出故障
                hops[int(np.where(plist == string[i][0])[0])].next = hops[int(np.where(plist == string[i][-1])[0])]
                if not silent:
                    print(
                        f'next node of node h{hops[int(np.where(plist == string[i][0])[0])].id} is {hops[int(np.where(plist == string[i][0])[0])].next.id}')

        hops_string = []
        for h in hops:
            for hini in hops:
                hini.readed = 0
            level, chain = dfs(h)
            hops_string.append(chain)
            if not silent:
                print(f'level of node h{h.id} is {level}, chain is {chain}')
        string = []
        for hsi in range(len(hops_string)):
            a = hops_string[hsi]
            for hsj in range(len(hops_string)):
                if hsi != hsj:
                    b = hops_string[hsj]
                    if set(b).issuperset(set(a)):
                        a = b
            string.append(a)
        string = [list(item) for item in set(tuple(row) for row in string)]
        if not silent:
            print(f'dfs string is {string}')

        strings.append(string)
        i_t += 1

    # compute connectivity between frames

    startandendtime = []
    for iframe in range(len(strings)):
        string_startend = []
        for sub_string in range(len(strings[iframe])):
            string_startend.append([TimeList[iframe], TimeList[iframe] + dt])  # 原来是 +1
        startandendtime.append(string_startend)
    # startandendtime_out = startandendtime.copy()
    # ####find connection between frames
    # print('start processing')
    # reconstruct the string as an object.
    Strings = []
    for skindex, sk in enumerate(strings):
        stime = []
        for skiindex, ski in enumerate(sk):
            string = String(ski, startandendtime[skindex][skiindex])
            stime.append(string)
        Strings.append(stime)

    # find connective microstring or not, if microstring is true, then don't  find it
    if not microstring:
        St1 = Strings[0]
        t2_index = 1
        while t2_index < len(Strings):
            St2 = Strings[t2_index]
            for si_index, si in enumerate(St1):
                pm = si.id[-1]  # get the final particle index of string 1
                dis_matrix = []
                countS = []
                for sj_index, sj in enumerate(St2):
                    pn = sj.id[0]  # the first particle index in the later string
                    t1 = si.time[0]
                    t2 = sj.time[-1]
                    drP_mn = np.linalg.norm(distPBC2D(glass.frames[t1, pm, 2:4] - glass.frames[t2, pn, 2:4],
                                                      glass.L[0], glass.L[1]))
                    if drP_mn <= ConnectThreshold:
                        dis_matrix.append(drP_mn)
                        countS.append(sj_index)
                        # save distance when it smaller than connectThreshold
                if dis_matrix:
                    psest = np.argmin(dis_matrix)
                    sj = St2[countS[psest]]
                    si.id.extend(sj.id)
                    si.time = [si.time[0], sj.time[-1]]
                    St2.remove(sj)
            St1.extend(St2)
            Strings.remove(St2)

    # reorganize strings
    new_string = []
    new_startend = []
    stringlength = []
    for ti, si in enumerate(Strings):
        for tii, sii in enumerate(si):
            new_startend.append(sii.time)
            new_string.append(sii.id)
            stringlength.append(len(sii.id))
    if len(new_string) != 0:
        sortlist = sorted(zip(stringlength, new_string, new_startend), reverse=True)
        glass.stringlength, glass.connected_components, glass.starend_of_string = zip(*sortlist)
        if ignoreLoop:
            stringList = []
            lengthList = []
            startendList = []
            for i, particleInchain in enumerate(glass.connected_components):
                if len(particleInchain) == len(set(particleInchain)) and len(particleInchain) >= stlength:
                    #  When exactly one complete loop is found
                    dist_headtail_vec = distPBC2D(glass.frames[glass.starend_of_string[i][1], particleInchain[0], 2:4] -
                                                  glass.frames[glass.starend_of_string[i][0], particleInchain[-1], 2:4],
                                                  glass.L[0], glass.L[1])
                    dist_headtail = np.linalg.norm(dist_headtail_vec)
                    if dist_headtail > dr_ht:
                        stringList.append(particleInchain)
                        lengthList.append(glass.stringlength[i])
                        startendList.append(glass.starend_of_string[i])
            glass.stringlength = lengthList
            glass.connected_components = stringList
            glass.starend_of_string = startendList
            # print(stringList)
            if len(stringList) != 0:
                return 1
            else:
                return 0
        else:
            stringList = []
            lengthList = []
            startendList = []
            for i, particleInchain in enumerate(glass.connected_components):
                dist_headtail_vec = distPBC2D(
                    glass.frames[glass.starend_of_string[i][1], particleInchain[0], 2:4] -
                    glass.frames[glass.starend_of_string[i][0], particleInchain[-1], 2:4], glass.L[0], glass.L[1])
                dist_headtail = np.linalg.norm(dist_headtail_vec)
                if dist_headtail > dr_ht:
                    stringList.append(particleInchain)
                    lengthList.append(glass.stringlength[i])
                    startendList.append(glass.starend_of_string[i])
            glass.stringlength = lengthList
            glass.connected_components = stringList
            glass.starend_of_string = startendList
            if len(stringList) != 0:
                return 1
            else:
                return 0



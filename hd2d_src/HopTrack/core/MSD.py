import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import numpy as np
import matplotlib.pyplot as plt



def MSD(glass, fr1=0, interval=10, COM=False, frinc=1.4, realdt=1, spf=1):
    T, N, d = np.shape(glass.frames)
    dt = []
    r2mean = []
    if fr1 == 0:
        frame = np.arange(0, T)
    else:
        frame = np.arange(fr1, T)
    dfr = 1
    framestart = 0
    frameend = T-1
    flag_iter = 0
    while flag_iter < 2:
        framesample = np.arange(frameend - dfr, framestart - 1, -interval)
        r2mean1 = []
        for fr in framesample:
            dr = glass.frames[fr + dfr, :, 2:4] - glass.frames[fr, :, 2:4]
            if COM:
                drmean = np.mean(dr, axis=0)
                dr -= drmean
            dr2 = np.linalg.norm(dr, axis=1)
            r2mean1.append(np.mean(dr2))
        r2mean.append(np.mean(r2mean1))
        dt.append(realdt * dfr * spf)
        dfr = int(round(dfr * frinc) + 1)
        if frameend - framestart < dfr:
            flag_iter += 1
            dfr = frameend - framestart
    dt_out = dt
    msd_out = r2mean
    out = np.array([dt_out, msd_out])
    return out.T



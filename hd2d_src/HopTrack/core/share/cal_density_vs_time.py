import numpy as np
from hd2d_src.HopTrack.core.share.circle_intersection_area import circle_intersection_area
import math

# purely calculate local density along time
def cal_density_vs_time(glass, CalRegion, x_rlow, x_rup, y_rlow, y_rup, x1, y1, r1):
    nt, _, _ = np.shape(CalRegion)
    ld = []
    for t in range(nt):
        ParticlesInRegion = CalRegion[t,
                            (CalRegion[t, :, 2] > x_rlow) & (CalRegion[t, :, 2] < x_rup) & (
                                    CalRegion[t, :, 3] > y_rlow) & (CalRegion[t, :, 3] < y_rup), :]
        S_area = 0
        for ci, CalParticle in enumerate(ParticlesInRegion):
            x2 = CalParticle[2]
            y2 = CalParticle[3]
            r2 = glass.radii[int(CalParticle[0]) - 1]
            S_area += circle_intersection_area(x1, y1, r1, x2, y2, r2)
        density = S_area / (math.pi * r1 ** 2)
        ld.append(density)
    return ld
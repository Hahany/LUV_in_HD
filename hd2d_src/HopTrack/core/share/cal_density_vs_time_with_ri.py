import numpy as np
from hd2d_src.HopTrack.core.share.circle_intersection_area import circle_intersection_area
from hd2d_src.HopTrack.core.share.distPBC2D import distPBC2D
import math

# calculate the local density sround point (x y) with radius r and return the minal distance
# around the particle (x y)
def cal_density_vs_time_with_ri(glass, CalRegion, x_rlow, x_rup, y_rlow, y_rup, x1, y1, r1):
    nt, _, _ = np.shape(CalRegion)
    ld = []
    mindr = []
    for t in range(nt):
        ParticlesInRegion = CalRegion[t,
                            (CalRegion[t, :, 2] > x_rlow) & (CalRegion[t, :, 2] < x_rup) & (
                                    CalRegion[t, :, 3] > y_rlow) & (CalRegion[t, :, 3] < y_rup), :]
        S_area = 0
        S_area0 = 0
        dr = []  # record the distance of local region center and particle center
        for ci, CalParticle in enumerate(ParticlesInRegion):
            x2 = CalParticle[2]
            y2 = CalParticle[3]
            r2 = glass.radii[int(CalParticle[0]) - 1]
            S_area += circle_intersection_area(x1, y1, r1, x2, y2, r2)
            dr.append(np.linalg.norm(distPBC2D([x1 - x2, y1 - y2], glass.L[0], glass.L[1])))
        density = S_area / (math.pi * r1 ** 2)
        ld.append(density)
        mindr.append(np.min(dr))
    return ld, mindr

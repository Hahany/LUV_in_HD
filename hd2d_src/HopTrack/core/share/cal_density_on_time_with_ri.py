import numpy as np
from hd2d_src.HopTrack.core.share.circle_intersection_area import circle_intersection_area
import math
from hd2d_src.HopTrack.core.share.distPBC2D import distPBC2D

def cal_density_on_time_with_ri(glass, frame, x_rlow, x_rup, y_rlow, y_rup, x1, y1, r1):
    """单帧版本，替代 cal_density_vs_time_with_ri 只算单个时间点的情况"""
    ParticlesInRegion = frame[
        (frame[:, 2] > x_rlow) & (frame[:, 2] < x_rup) &
        (frame[:, 3] > y_rlow) & (frame[:, 3] < y_rup), :]
    S_area = 0
    dr = []
    for CalParticle in ParticlesInRegion:
        x2 = CalParticle[2]
        y2 = CalParticle[3]
        r2 = glass.radii[int(CalParticle[0]) - 1]
        S_area += circle_intersection_area(x1, y1, r1, x2, y2, r2)
        dr.append(np.linalg.norm(distPBC2D([x1 - x2, y1 - y2], glass.L[0], glass.L[1])))
    density = S_area / (math.pi * r1 ** 2)
    min_dr = np.min(dr) if dr else 0.0
    return density, min_dr
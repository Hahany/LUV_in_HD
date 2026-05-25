import numpy as np
from hd2d_src.HopTrack.core.share.circle_intersection_area import circle_intersection_area
import math

def cal_density_on_time(glass, frame, x_rlow, x_rup, y_rlow, y_rup, x1, y1, r1):
    # frame shape: (N, 4)，不再需要 t 参数
    ParticlesInRegion = frame[
        (frame[:, 2] > x_rlow) & (frame[:, 2] < x_rup) &
        (frame[:, 3] > y_rlow) & (frame[:, 3] < y_rup), :]
    S_area = 0
    for CalParticle in ParticlesInRegion:
        x2 = CalParticle[2]
        y2 = CalParticle[3]
        r2 = glass.radii[int(CalParticle[0]) - 1]
        S_area += circle_intersection_area(x1, y1, r1, x2, y2, r2)
    return S_area / (math.pi * r1 ** 2)
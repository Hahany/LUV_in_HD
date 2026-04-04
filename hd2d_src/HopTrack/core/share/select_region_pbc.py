import numpy as np
# select  a region and use the periodic condition when the region at the edge
def select_region_pbc(glass, x1, y1, regionradius):
    r_region = regionradius + np.max(glass.radii) + 1.0
    x_rlow = x1 - r_region
    x_rup = x1 + r_region
    y_rlow = y1 - r_region
    y_rup = y1 + r_region
    CalRegion = glass.frames.copy()

    # PBC
    if x_rlow < glass.x_low:
        x_rlowp = (glass.L[0]) + x_rlow
        CalRegion[CalRegion[:, :, 2] > x_rlowp, 2] = CalRegion[CalRegion[:, :, 2] > x_rlowp, 2] - \
                                                     glass.L[0]
    elif x_rup > glass.x_up:
        x_rupp = x_rup - (glass.L[0])
        CalRegion[CalRegion[:, :, 2] < x_rupp, 2] = CalRegion[CalRegion[:, :, 2] < x_rupp, 2] + glass.L[
            0]
    if y_rlow < glass.y_low:
        y_rlowp = (glass.L[1]) + y_rlow
        CalRegion[CalRegion[:, :, 3] > y_rlowp, 3] = CalRegion[CalRegion[:, :, 3] > y_rlowp, 3] - \
                                                     glass.L[1]
    elif y_rup > glass.y_up:
        y_rupp = y_rup - (glass.L[1])
        CalRegion[CalRegion[:, :, 3] < y_rupp, 3] = CalRegion[CalRegion[:, :, 3] < y_rupp, 3] + glass.L[
            1]
    return CalRegion, x_rlow, x_rup, y_rlow, y_rup

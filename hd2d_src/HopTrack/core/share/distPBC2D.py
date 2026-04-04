#periodic boundary condition for 2D
def distPBC2D(vec, Lx, Ly):
    hLx = Lx / 2
    hLy = Ly / 2
    if vec[0] > hLx:
        vec[0] = vec[0] - Lx
    elif vec[0] < -hLx:
        vec[0] = vec[0] + Lx

    if vec[1] > hLy:
        vec[1] = vec[1] - Ly
    elif vec[1] < -hLy:
        vec[1] = vec[1] + Ly

    return vec
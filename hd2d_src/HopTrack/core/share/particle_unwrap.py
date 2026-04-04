import numpy as np
# unwrap value b from a and l
def particle_unwrap(a, b, l):  # unwrap a specific value
    r_ab = np.abs(a - b)
    hl = l / 2
    if r_ab > hl:
        b = b + (b / (a - b)) * l
    return b
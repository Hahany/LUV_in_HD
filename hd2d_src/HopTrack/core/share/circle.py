import numpy as np
# get x y coordinate of a circle
def circle(x0, y0, r):
    theta = np.linspace(0, 2 * np.pi, 100)
    a = r * np.cos(theta) + x0
    b = r * np.sin(theta) + y0
    return a, b
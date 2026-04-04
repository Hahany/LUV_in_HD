import math
# calse the intersection of two circle
def circle_intersection_area(x1, y1, r1, x2, y2, r2):
    # distance between circle center
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # To determine the positional relationship between two circles and calculate area
    if d > r1 + r2:
        return 0
    elif d <= abs(r1 - r2):
        r = min(r1, r2)
        return math.pi * r ** 2
    else:
        l = (-r1 ** 2 + r2 ** 2 + d ** 2) / (2 * d)
        theta1 = math.acos(l / r2)
        theta2 = math.acos((d - l) / r1)
        s = math.sqrt(r2 ** 2 - l ** 2)
        return theta1 * (r2 ** 2) + theta2 * (r1 ** 2) - d * s


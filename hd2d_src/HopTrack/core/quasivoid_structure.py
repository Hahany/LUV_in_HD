import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate, scale
from shapely.geometry.base import BaseGeometry
import sys
sys.setrecursionlimit(1000)

def rotate_segment_90_degrees(p1, p2):
    # Calculate the midpoint of the segment
    x1, y1 = p1
    x2, y2 = p2
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2

    # Translate the points so that the midpoint becomes the origin
    x1_translated = x1 - midpoint_x
    y1_translated = y1 - midpoint_y
    x2_translated = x2 - midpoint_x
    y2_translated = y2 - midpoint_y

    # Rotate the points 90 degrees counter-clockwise
    x1_rotated = -y1_translated
    y1_rotated = x1_translated
    x2_rotated = -y2_translated
    y2_rotated = x2_translated

    # Translate back to the original position
    x1_final = x1_rotated + midpoint_x
    y1_final = y1_rotated + midpoint_y
    x2_final = x2_rotated + midpoint_x
    y2_final = y2_rotated + midpoint_y

    p1_rotated = [x1_final, y1_final]
    p2_rotated = [x2_final, y2_final]
    return p1_rotated, p2_rotated


def reflect_geometry(geometry: BaseGeometry, point1, point2):
    """
    Reflect a geometry along an arbitrary line.
    :param geometry: The Shapely geometry object to reflect
    :param point1: Start point of the reflection axis (x1, y1)
    :param point2: End point of the reflection axis (x2, y2)
    :return: The reflected geometry object
    """
    # Calculate the angle of the reflection axis relative to the x-axis
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = np.degrees(np.arctan2(dy, dx))  # Angle between the reflection axis and x-axis

    # Translate the geometry so that the reflection axis passes through the origin
    geometry_translated = translate(geometry, xoff=-point1[0], yoff=-point1[1])

    # Rotate the geometry so that the reflection axis aligns with the x-axis
    geometry_rotated = rotate(geometry_translated, -angle, origin=(0, 0))

    # Reflect the geometry along the x-axis
    geometry_reflected = scale(geometry_rotated, xfact=1, yfact=-1, origin=(0, 0))

    # Rotate the geometry back to its original orientation
    geometry_restored = rotate(geometry_reflected, angle, origin=(0, 0))

    # Translate the geometry back to its original position
    geometry_final = translate(geometry_restored, xoff=point1[0], yoff=point1[1])

    return geometry_final


def find_and_reflect_intersection(fragment, b, ax, t, l=1, V_fs=[]):
    """
    Recursively find the overlapping region with circles, reflect the region,
    and update the fragment until no further intersection occurs.

    Parameters:
    - fragment: Current overlapping region
    - b: The system (with frames and radii)
    - ax: The plotting axis for visualization
    - i: Frame index (used for labeling)
    - j: Index for looping over the circles (default is 0)
    """
    # Stop condition: If j exceeds the number of circles, stop recursion
    if l == 0:
        return V_fs

    intersections = []
    fragments = []
    for j in range(len(b.frames[0, :, 0])):
        # Get the radius and center of the circle
        r = b.radii[int(b.frames[t, j, 0] - 1)]
        circle = Point(b.frames[t, j, 2:4]).buffer(r)
        # Calculate intersection
        intersection = circle.intersection(fragment)
        if ( not intersection.is_empty) and (intersection.geom_type == "Polygon"):
            intersections.append(intersection)
            # If intersection exists, reflect the intersection region
            p_t0 = b.frames[0, j, 2:4]
            p_t1 = b.frames[-1, j, 2:4]
            point1, point2 = rotate_segment_90_degrees(p_t0, p_t1)

            # Reflect the intersection and update fragment
            fragment1 = reflect_geometry(intersection, point1, point2)
            fragments.append(fragment1)


            # # Visualize the intersection and the reflected region
            # x, y = intersection.exterior.xy
            # ax.fill(x, y, alpha=0.5, fc='m')
            #
            # x, y = fragment1.exterior.xy
            # ax.fill(x, y, alpha=0.5, fc='y')
            # plt.pause(1)
    l = len(intersections)
    if l >= 1:
        V_f = fragment
        for intersection in intersections:
            V_f = V_f.difference(intersection)
        if V_f.geom_type == "Polygon" and not V_f.is_empty:
            x, y = V_f.exterior.xy
            ax.fill(x, y, alpha=0.4, fc='c')
            V_fs.append(V_f)
    else:
        x, y = fragment.exterior.xy
        ax.fill(x, y, alpha=0.4, fc='c')
        V_fs.append(fragment)
    # Recursive step: move to the next circle and repeat
    for fragment in fragments:
        find_and_reflect_intersection(fragment, b, ax, t, l=l, V_fs=V_fs)
    return V_fs

def calculate_area_in_circle(shapes, circle_center, circle_radius, ax, color='k'):
    """
    Calculate the total area of shapes that are within a circular region.

    Parameters:
    - shapes: List of Shapely geometry objects (e.g., Polygon, MultiPolygon).
    - circle_center: Tuple (x, y) representing the center of the circle.
    - circle_radius: Radius of the circle.

    Returns:
    - Total area of shapes within the circular region.
    """
    # Create the circular region
    circular_region = Point(circle_center).buffer(circle_radius)
    circle = Circle(circle_center, circle_radius, facecolor='none', edgecolor=color, linestyle='--', lw=1.5, zorder=10)
    ax.add_artist(circle)

    # Compute the intersection of each shape with the circular region
    total_area = 0
    for shape in shapes:
        intersection = shape.intersection(circular_region)
        total_area += intersection.area

    return total_area



def quasivoid_structure_tail(b, start_time, end_time, color, region, Rc, rh, rs, dr_ht, stlength, savepath):
    fig, ax = plt.subplots()
    plt.ion()
    mean_radius = np.min(b.radii)
    # b.chbox(20, 40, 50, 60, update_bak=True)
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    # b.short_burst(1, 1, 10, 0)
    b.chooseWithoutCoarsening(2)
    b.findstring(HopThreshold=rh*2*mean_radius, ConnectThreshold=rs*2*mean_radius,
                 ignoreLoop=True, stlength=stlength, dr_ht=dr_ht*2*mean_radius, hop_whole=True)
    b.showstring(ax=ax, showid=False, showtraj=False, stringID=[], SSC=color,
                 size=10, WL=True, show_localdensity_region=True, Rc = Rc, mode=0, findquasivoid=True, nodot=True)
    # plot trajectory
    b.setduration(start_time, end_time)
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time-start_time), lw=2, ms=2.5,
               nodot=False, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False, overlap=3)
    ax.set_xlim([b.frames[0, b.connected_components[0][0], 2] - Rc - 1,
                 b.frames[0, b.connected_components[0][0], 2] + Rc + 1])
    ax.set_ylim([b.frames[0, b.connected_components[0][0], 3] - Rc - 1,
                 b.frames[0, b.connected_components[0][0], 3] + Rc + 1])

    ## plot quasivoid at string tail initial time
    quasivoid_tail_final = [b.frames[-1, b.connected_components[0][0], 2],
                            b.frames[-1, b.connected_components[0][0], 3]]
    r_q = b.radii[int(b.frames[0, b.connected_components[0][0], 0] - 1)]
    circle_q = Point(quasivoid_tail_final).buffer(r_q)
    circle = Circle(quasivoid_tail_final, r_q, facecolor='none', edgecolor='r',alpha=0.4, linestyle='--', lw=1.5, zorder=10)
    ax.add_artist(circle)
    quasivoid_tail_init = [b.frames[0, b.connected_components[0][0], 2], b.frames[0, b.connected_components[0][0], 3]]
    circle_q0 = Point(quasivoid_tail_init).buffer(r_q)
    inter_qq = circle_q.intersection(circle_q0)
    fragment = circle_q.difference(inter_qq)
    if not fragment.is_empty:
        V_fs = find_and_reflect_intersection(fragment, b, ax, 0, 1, [])  # Recursively process the intersection
    plt.ioff()
    V_s = calculate_area_in_circle(V_fs, quasivoid_tail_init, Rc, ax, 'red')
    v_frag = V_s/(np.pi*mean_radius**2)
    print(f'{v_frag}')
    fig.savefig(f"{savepath}_tail.png", dpi=300)
    return v_frag

def quasivoid_structure_tail_statis(b, si, Rc, savepath):
    fig, ax = plt.subplots()
    plt.ion()
    mean_radius = np.min(b.radii)
    # b.chbox(20, 40, 50, 60, update_bak=True)
    start_time = b.starend_of_string[si][0]
    end_time = b.starend_of_string[si][1]
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time-start_time), lw=2, ms=2.5,
               nodot=False, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=True, overlap=3)
    ax.set_xlim([b.frames[0, b.connected_components[si][0], 2] - Rc - 1,
                 b.frames[0, b.connected_components[si][0], 2] + Rc + 1])
    ax.set_ylim([b.frames[0, b.connected_components[si][0], 3] - Rc - 1,
                 b.frames[0, b.connected_components[si][0], 3] + Rc + 1])

    ## plot quasivoid at string tail initial time
    quasivoid_tail_final = [b.frames[end_time, b.connected_components[si][0], 2],
                            b.frames[end_time, b.connected_components[si][0], 3]]
    r_q = b.radii[int(b.frames[start_time, b.connected_components[si][0], 0] - 1)]
    circle_q = Point(quasivoid_tail_final).buffer(r_q)
    quasivoid_tail_init = [b.frames[start_time, b.connected_components[si][0], 2], b.frames[start_time, b.connected_components[si][0], 3]]
    circle_q0 = Point(quasivoid_tail_init).buffer(r_q)
    inter_qq = circle_q.intersection(circle_q0)
    fragment = circle_q.difference(inter_qq)
    if not fragment.is_empty:
        V_fs = find_and_reflect_intersection(fragment, b, ax, 0, 1, [])  # Recursively process the intersection
    plt.ioff()
    V_s = calculate_area_in_circle(V_fs, quasivoid_tail_init, Rc, ax, 'red')
    v_frag = V_s/(np.pi*mean_radius**2)
    print(f'{v_frag}')
    fig.savefig(f"{savepath}_tail.png", dpi=300)
    return v_frag

def quasivoid_structure_head(b, start_time, end_time, color, region,
                             Rc, rh, rs, dr_ht, stlength, savepath):
    fig, ax = plt.subplots()
    plt.ion()
    mean_radius = np.min(b.radii)
    # b.chbox(20, 40, 50, 60, update_bak=True)
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    # b.short_burst(1, 1, 10, 0)
    b.chooseWithoutCoarsening(2)
    b.findstring(HopThreshold=rh*2*mean_radius, ConnectThreshold=rs*2*mean_radius,
                 ignoreLoop=True, stlength=stlength, dr_ht=dr_ht*2*mean_radius, hop_whole=True)
    b.showstring(ax=ax, showid=False, showtraj=False, stringID=[], SSC=color,
                 size=10, WL=True, show_localdensity_region=True, Rc = Rc, mode=0,
                 findquasivoid=True, nodot=True, reverse=True)
    # plot trajectory
    b.setduration(start_time, end_time)
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time-start_time), lw=2, ms=2.5,
               nodot=False, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False, overlap=0)
    ## plot quasivoid at string head at final time
    ax.set_xlim([b.frames[-1, b.connected_components[0][-1], 2] - Rc - 1,
                 b.frames[-1, b.connected_components[0][-1], 2] + Rc + 1])
    ax.set_ylim([b.frames[-1, b.connected_components[0][-1], 3] - Rc - 1,
                 b.frames[-1, b.connected_components[0][-1], 3] + Rc + 1])

    quasivoid_head_init = [b.frames[0, b.connected_components[0][-1], 2],
                            b.frames[0, b.connected_components[0][-1], 3]]
    r_q = b.radii[int(b.frames[0, b.connected_components[0][-1], 0] - 1)]
    circle_q = Point(quasivoid_head_init).buffer(r_q)
    quasivoid_head_final = [b.frames[-1, b.connected_components[0][-1], 2],
                            b.frames[-1, b.connected_components[0][-1], 3]]
    circle_q0 = Point(quasivoid_head_final).buffer(r_q)
    circle = Circle(quasivoid_head_init, r_q, facecolor='none', edgecolor='b',alpha=0.4, linestyle='--', lw=1.5, zorder=10)
    ax.add_artist(circle)
    inter_qq = circle_q.intersection(circle_q0)
    fragment = circle_q.difference(inter_qq)
    if not fragment.is_empty:
        V_ss = find_and_reflect_intersection(fragment, b, ax, -1, 1, [])  # Recursively process the intersection
    plt.ioff()
    V_s = calculate_area_in_circle(V_ss, quasivoid_head_final, Rc, ax, 'blue')
    v_frag = V_s / (np.pi * mean_radius ** 2)
    print(f'{v_frag}')
    fig.savefig(f"{savepath}_head.png", dpi=300)
    return v_frag

def quasivoid_structure_head_statis(b, si, Rc, savepath):
    fig, ax = plt.subplots()
    plt.ion()
    mean_radius = np.min(b.radii)
    start_time = b.starend_of_string[si][0]
    end_time = b.starend_of_string[si][1]
    # plot trajectory
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time-start_time), lw=2, ms=2.5,
               nodot=False, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=True, overlap=0)
    ## plot quasivoid at string head at final time
    ax.set_xlim([b.frames[end_time, b.connected_components[si][-1], 2] - Rc - 1,
                 b.frames[end_time, b.connected_components[si][-1], 2] + Rc + 1])
    ax.set_ylim([b.frames[end_time, b.connected_components[si][-1], 3] - Rc - 1,
                 b.frames[end_time, b.connected_components[si][-1], 3] + Rc + 1])

    quasivoid_head_init = [b.frames[start_time, b.connected_components[si][-1], 2],
                            b.frames[start_time, b.connected_components[si][-1], 3]]
    r_q = b.radii[int(b.frames[start_time, b.connected_components[si][-1], 0] - 1)]
    circle_q = Point(quasivoid_head_init).buffer(r_q)
    quasivoid_head_final = [b.frames[end_time, b.connected_components[si][-1], 2],
                            b.frames[end_time, b.connected_components[si][-1], 3]]
    circle_q0 = Point(quasivoid_head_final).buffer(r_q)
    inter_qq = circle_q.intersection(circle_q0)
    fragment = circle_q.difference(inter_qq)
    if not fragment.is_empty:
        V_ss = find_and_reflect_intersection(fragment, b, ax, -1, 1, [])  # Recursively process the intersection
    plt.ioff()
    V_s = calculate_area_in_circle(V_ss, quasivoid_head_final, Rc, ax, 'blue')
    v_frag = V_s / (np.pi * mean_radius ** 2)
    print(f'{v_frag}')
    fig.savefig(f"{savepath}_head.png", dpi=300)
    return v_frag

def head_and_tial(b, start_time, end_time, color, region, Rc, rh, rs, dr_ht, stlength, savepath):
    fig, ax = plt.subplots()
    plt.ion()
    mean_radius = np.min(b.radii)
    # b.chbox(20, 40, 50, 60, update_bak=True)
    x1, x2, y1, y2 = region
    b.chbox(x1, x2, y1, y2, update_bak=True)
    b.setduration(start_time, end_time)
    # b.short_burst(1, 1, 10, 0)
    b.chooseWithoutCoarsening(2)
    b.findstring(HopThreshold=rh * 2 * mean_radius, ConnectThreshold=rs * 2 * mean_radius,
                 ignoreLoop=True, stlength=stlength, dr_ht=dr_ht * 2 * mean_radius, hop_whole=True)
    # b.showstring(ax=ax, showid=False, showtraj=False, stringID=[], SSC=color,
    #              size=10, WL=True, show_localdensity_region=True, Rc=Rc, mode=0, findquasivoid=True, nodot=True)
    # plot trajectory
    b.setduration(start_time, end_time)
    b.hops(0.8)
    b.showdisp(fig, ax, t_start=0, t_end=int(end_time - start_time), lw=2, ms=2.5,
               nodot=True, showvoid=False, showpid=False, showforcepid=0, showforcedid=0,
               colorbar=False, showradii=False, overlap=3)

    b.setduration(start_time, end_time)
    quasivoid_tail_final = [b.frames[-1, b.connected_components[0][0], 2],
                            b.frames[-1, b.connected_components[0][0], 3]]
    r_q = b.radii[int(b.frames[0, b.connected_components[0][0], 0] - 1)]
    circle_q = Point(quasivoid_tail_final).buffer(r_q)
    circle = Circle(quasivoid_tail_final, r_q, facecolor='none', edgecolor='r',alpha=0.4, linestyle='--', lw=1.5, zorder=10)
    ax.add_artist(circle)
    quasivoid_tail_init = [b.frames[0, b.connected_components[0][0], 2], b.frames[0, b.connected_components[0][0], 3]]
    circle_q0 = Point(quasivoid_tail_init).buffer(r_q)
    inter_qq = circle_q.intersection(circle_q0)
    fragment = circle_q.difference(inter_qq)
    if not fragment.is_empty:
        V_fs = find_and_reflect_intersection(fragment, b, ax, 0, 1, [])  # Recursively process the intersection
    plt.ioff()
    V_s = calculate_area_in_circle(V_fs, quasivoid_tail_init, Rc, ax, 'red')
    v_frag = V_s / (np.pi * mean_radius ** 2)
    print(f'{v_frag}')

    quasivoid_head_init = [b.frames[0, b.connected_components[0][-1], 2],
                            b.frames[0, b.connected_components[0][-1], 3]]
    r_q = b.radii[int(b.frames[0, b.connected_components[0][-1], 0] - 1)]
    circle_q = Point(quasivoid_head_init).buffer(r_q)
    quasivoid_head_final = [b.frames[-1, b.connected_components[0][-1], 2],
                            b.frames[-1, b.connected_components[0][-1], 3]]
    circle_q0 = Point(quasivoid_head_final).buffer(r_q)
    circle = Circle(quasivoid_head_init, r_q, facecolor='none', edgecolor='b',alpha=0.4, linestyle='--', lw=1.5, zorder=10)
    ax.add_artist(circle)
    inter_qq = circle_q.intersection(circle_q0)
    fragment = circle_q.difference(inter_qq)
    if not fragment.is_empty:
        V_ss = find_and_reflect_intersection(fragment, b, ax, -1, 1, [])  # Recursively process the intersection
    plt.ioff()
    V_s = calculate_area_in_circle(V_ss, quasivoid_head_final, Rc, ax, 'blue')
    v_frag = V_s / (np.pi * mean_radius ** 2)
    print(f'{v_frag}')
    fig.savefig(f"{savepath}_head_and_tail.png", dpi=300)
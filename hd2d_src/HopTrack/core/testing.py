
import matplotlib.pyplot as plt


def plot_test(ax1, center1, radius1, center2, radius2, center3, radius3):


    c1 = plt.Circle(center1, radius1, facecolor='none', edgecolor='black', alpha=1, zorder=0)
    c2 = plt.Circle(center2, radius2, facecolor='none', edgecolor='r', alpha=1, zorder=0)
    c3 = plt.Circle(center3, radius3, facecolor='none', edgecolor='b', alpha=1, zorder=0)
    ax1.add_artist(c1)
    ax1.add_artist(c2)
    ax1.add_artist(c3)
    radius1 = 2*radius1
    radius2 = 2*radius2
    radius3 = 2*radius3
    # ax1.set_xlim(min(center1[0] - radius1, center2[0] - radius2, center3[0] - radius3),
    #              max(center1[0] + radius1, center2[0] + radius2, center3[0] + radius3))
    # ax1.set_ylim(min(center1[1] - radius1, center2[1] - radius2, center3[1] - radius3),
    #              max(center1[1] + radius1, center2[1] + radius2, center3[1] + radius3))
    ax1.set_aspect('equal')
if __name__ == '__main__':
    plot_test((0, 0), 2, (0, 1), 2, (1, 0), 2)


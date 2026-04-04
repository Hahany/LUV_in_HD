import hd2d_src.HopTrack.core.share.circle as circle
def set_ax(glass, ax, x1, y1, r1, x_low, x_up, y_low, y_up, x_qt=0, y_qt=0, x_qh=0, y_qh=0):
    #ax, (x1, y1, r1), (x_low, x_up, y_low, y_up), (x_qt, y_qt, x_qh, y_qh)
    a, b = circle(x1, y1, r1)
    ax.set_xlim(x_low, x_up)
    ax.set_ylim(y_low, y_up)
    if sum([x_qt, y_qt, x_qh, y_qh]) > 0:
        ax.scatter(x_qt, y_qt, c='g', marker='s')
        ax.scatter(x_qh, y_qh, c='g', marker='o')
    ax.plot(a, b, c='k')
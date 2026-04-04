from hd2d_src.HopTrack.core.share import circle


#plot particles in the system with precise size

def plot_config(glass, ax, CalRegion, mask, t, ld, c):
    ParticlesInRegion = CalRegion[t, mask, :]
    for ci, CalParticle in enumerate(ParticlesInRegion):
        x2 = CalParticle[2]
        y2 = CalParticle[3]
        r2 = glass.radii[int(CalParticle[0]) - 1]
        a, b = circle(x2, y2, r2)
        ax.plot(a, b, c=c, alpha=1.0, linewidth=0.5)  # plot the particle with precise size
    ax.text(0.1, 0.9, f'local density = {ld:.04f}', transform=ax.transAxes)

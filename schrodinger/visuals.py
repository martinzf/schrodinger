"""
visuals.py: Collection of functions for plotting and animating wavefunctions.
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation
from matplotlib import animation
matplotlib.use('Qt5Agg')
plt.style.use('fast')
matplotlib.rcParams.update({
    'font.size': 18,
    'figure.figsize': (10, 8)
})


# 3D PLOT QUALITY
# - Downsampling number of wavefunction datapoints by factor 1 / FACTOR
# - Maximum grid points = MAXGRID
FACTOR = 4
MAXGRID = 100

# WAVEFUNCTION COLOURMAP
CMAP = 'hsv'


def colored_line(x, psi, cmap=CMAP, linewidth=2):
    '''Returns a colored LineCollection object.'''

    # Create segments for the line
    points = np.array([x, np.abs(psi)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection
    norm = plt.Normalize(vmin=-1, vmax=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.angle(psi)/np.pi)
    lc.set_linewidth(linewidth)

    return lc


def colorbar():
    '''Adds colorbar to the plot.'''

    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(cmap='hsv', norm=norm)
    cbar = plt.gcf().colorbar(sm, ax=plt.gca(), orientation='vertical', pad=.15, shrink=0.6)
    cbar.set_label(r'Phase ($\pi$ rad)')


def plot(x, y, xlabel, title):
    '''Plots a wavefunction at a given time or position.'''

    lc = colored_line(x, y)
    plt.gca().add_collection(lc)
    colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(r'$|\psi|$')
    plt.title(title)
    ymax = 1.1 * np.max(np.abs(y))
    plt.xlim(x[0], x[-1])
    plt.ylim(0, ymax)


def trisurf(T, X, PSI):
    '''Triangular mesh plot of a wavefunction.'''

    tri = Triangulation(T, X)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    colorbar()
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel(r'$|\psi|$')
    surf = ax.plot_trisurf(tri, np.abs(PSI))
    norm = plt.Normalize(vmin=-1, vmax=1)
    colors = plt.get_cmap('hsv')(norm(np.angle(PSI) / np.pi))
    surf.set_facecolors(colors[tri.triangles].mean(axis=1))


def downsample(psi, t, factor=FACTOR, maxgrid=MAXGRID):
    '''Reduces the number of points for trisurf plot.'''

    Nt, Nx = psi.shape
    Nt_ds = min(maxgrid, Nt // factor)
    Nx_ds = min(maxgrid, Nx // factor)
    ratio_t = Nt // Nt_ds
    ratio_x = Nx // Nx_ds
    t_ds = t[::ratio_t]
    psi_ds = psi[::ratio_t, ::ratio_x]

    return ratio_x, t_ds, psi_ds


def update_colored_line(lc, x, psi):
    '''Updates the shape and colour of a colored_line LineCollection.'''

    points = np.array([x, np.abs(psi)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc.set_segments(segments)
    lc.set_array(np.angle(psi)/np.pi)


def update_fill(patch, xnew, y0new, y1new):
    '''Updates the patch of an ax.fill_between().'''

    v_x = np.hstack([xnew[0],xnew,xnew[-1],xnew[::-1],xnew[0]])
    v_y = np.hstack([y1new[0],y0new,y0new[-1],y1new[::-1],y1new[0]])
    vertices = np.vstack([v_x,v_y]).T
    codes = np.array([1]+(2*len(xnew)+1)*[2]+[79]).astype('uint8')
    path = patch.get_paths()[0]
    path.vertices = vertices
    path.codes = codes


def animate_wf(wf):
    '''Animates a Wavefunction object.'''

    Nt, Nx = wf.amplitude.shape
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel(r'$|\psi|$')
    title = ax.text(.02, .95, '', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha='left')
    xmin = np.min(np.vectorize(wf.x1)(wf.t))
    xmax = np.max(np.vectorize(wf.x2)(wf.t))
    ymax = 1.1 * np.max(np.abs(wf.amplitude))
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    colorbar()
    x = np.linspace(wf.x1(wf.t[0]), wf.x2(wf.t[0]), Nx)
    lc = colored_line(x, wf.amplitude[0])
    ax.add_collection(lc)
    left_shade  = ax.fill_between([xmin, x[0]], [0, 0], [ymax, ymax], color='grey', alpha=0.5)
    right_shade = ax.fill_between([x[-1], xmax], [0, 0], [ymax, ymax], color='grey', alpha=0.5)
    def fun(i):
        x = np.linspace(wf.x1(wf.t[i]), wf.x2(wf.t[i]), Nx)
        update_colored_line(lc, x, wf.amplitude[i])
        update_fill(left_shade, [xmin, x[0]], [0, 0], [ymax, ymax])
        update_fill(right_shade, [x[-1], xmax], [0, 0], [ymax, ymax])
        title.set_text(rf't={wf.t[i]:.2f}, E={wf.E[i]:.2f}$\pm${wf.dE[i]:.2f}')
        return lc, left_shade, right_shade, title,
    return animation.FuncAnimation(fig, fun, frames=Nt, interval=50, blit=True)

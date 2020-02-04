#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:42:06 2018

@author: yulkang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union, Iterable

import numpy_groupies as npg

from . import np2

def ____Subplots____():
    pass

def subplotRC(nrow, ncol, row, col, **kwargs):
    iplot = (row - 1) * ncol + col
    ax = plt.subplot(nrow, ncol, iplot, **kwargs)
    return ax

def subplotRCs(nrow, ncol, **kwargs):
    ax = np.empty([nrow, ncol], dtype=object)
    for row in range(1, nrow+1):
        for col in range(1, ncol+1):
            ax[row-1, col-1] = subplotRC(nrow, ncol, row, col, **kwargs)
    return ax

def coltitle(cols, axes):
    h = []
    for ax, col in zip(axes[0,:], cols):
        h.append(ax.set_title(col))
    return np.array(h)

def rowtitle(rows, axes, pad=5):
    """
    :param rows: list of string row title
    :param axes: 2-D array of axes, as from subplotRCs()
    :param pad: in points.
    :return: n_rows array of row title handles
    adapted from: https://stackoverflow.com/a/25814386/2565317
    """
    from matplotlib.transforms import offset_copy

    labels = []
    for ax, row in zip(axes[:, 0], rows):
        label = ax.annotate(
            row,
            xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center')
        labels.append(label)

    fig = axes[0,0].get_figure()
    fig.tight_layout()

    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)

    return np.array(labels)

def ____Axes_Limits____():
    pass

def sameaxes(ax, ax0=None, xy='xy'):
    """
    Match the chosen limits of axes in ax to ax0's (if given) or the max range.
    Also consider: ax1.get_shared_x_axes().join(ax1, ax2)
    Optionally followed by ax1.set_xticklabels([]); ax2.autoscale()
    See: https://stackoverflow.com/a/42974975/2565317
    :param ax: np.ndarray (as from subplotRCs) or list of axes.
    :param ax0: a scalar axes to match limits to. if None (default),
    match the maximum range among axes in ax.
    :param xy: 'x'|'y'|'xy'(default)
    :return: [[min, max]] of limits. If xy='xy', contains two pairs.
    """
    if type(ax) is np.ndarray:
        ax = ax.reshape(-1)
    def cat_lims(lims):
        return np.concatenate([np.array(v1).reshape(1,2) for v1 in lims])
    lims_res = []
    for xy1 in xy:
        if ax0 is None:
            if xy1 == 'x':
                lims = cat_lims([ax1.get_xlim() for ax1 in ax])
            else:
                lims = cat_lims([ax1.get_ylim() for ax1 in ax])
            lims0 = [np.min(lims[:,0]), np.max(lims[:,1])]
        else:
            if xy1 == 'x':
                lims0 = ax0.get_xlim()
            else:
                lims0 = ax0.get_ylim()
        if xy1 == 'x':
            for ax1 in ax:
                ax1.set_xlim(lims0)
        else:
            for ax1 in ax:
                ax1.set_ylim(lims0)
        lims_res.append(lims0)
    return lims_res


def lim_symmetric(xy='y', lim=None, ax=None):
    """
    @type xy: str
    @type lim: float
    @type ax: plt.Axes
    @return: None
    """
    assert xy == 'x' or xy == 'y', 'xy must be "x" or "y"!'
    if ax is None:
        ax = plt.gca()
    if lim is None:
        if xy == 'x':
            lim = np.amax(np.abs(ax.get_xlim()))
        else:
            lim = np.amax(np.abs(ax.get_ylim()))
    if xy == 'x':
        ax.set_xlim(-lim, +lim)
    else:
        ax.set_ylim(-lim, +lim)


def same_clim(images, ax0=None):
    if type(images) is np.ndarray:
        images = images.reshape(-1)

    clims = np.array([im.get_clim() for im in images])
    clim = [np.amin(clims[:,0]), np.amax(clims[:,1])]
    for im in images:
        im.set_clim(clim)

def beautify_psychometric(ax=None, 
                          ylim=[0, 1],
                          y_margin=0.05,
                          axvline=False,
                          axhline=False):
    if ax is None:
        ax = plt.gca()
        
    dylim = ylim[1] - ylim[0]
    ylim_actual = [ylim[0] - dylim * y_margin, ylim[1] + dylim * y_margin]
    plt.ylim(ylim_actual)
    detach_yaxis(ymin=ylim[0], ymax=ylim[1])
    if ylim[0] == 0.5 and ylim[1] == 1:
        plt.yticks([0.5, 0.75, 1])
    elif ylim[0] == 0 and ylim[1] == 1:
        plt.yticks([0, 0.5, 1])
    
    box_off()
    if axvline:
        plt.axvline(x=0, 
                    color=[.9, .9, .9], 
                    linestyle='-', zorder=-1,
                    linewidth=0.5)
    if axhline:
        plt.axhline(y=0.5, 
                    color=[.9, .9, .9], 
                    linestyle='-', zorder=-1,
                    linewidth=0.5)

def detach_axis(xy='xy', amin=0, amax=None, ax=None, spine=None):
    if xy == 'xy':
        for xy1 in ['x', 'y']:
            detach_axis(xy1, amin, amax, ax)
        return
            
    if ax is None:
        ax = plt.gca()
        
    if xy == 'x':
        if spine is None:
            spine = 'bottom'
        lim = list(plt.xlim())
        if amin is not None:
            lim[0] = amin
        if amax is not None:
            lim[1] = amax
        ax.spines[spine].set_bounds(lim[0], lim[-1])
    elif xy == 'y':
        if spine is None:
            spine = 'left'
        lim = list(plt.ylim())
        if amin is not None:
            lim[0] = amin
        if amax is not None:
            lim[1] = amax
        ax.spines[spine].set_bounds(lim[0], lim[-1])
    else:
        raise ValueError("xy must be 'x', 'y', or 'xy'!")

def detach_yaxis(ymin=0, ymax=None, ax=None):
    detach_axis('y', ymin, ymax, ax)

def hide_ticklabels(xy='xy', ax=None):
    if ax is None:
        ax = plt.gca()
    if 'x' in xy:
        plt.setp(ax.get_xticklabels(), visible=False)
    if 'y' in xy:        
        plt.setp(ax.get_yticklabels(), visible=False)

def box_off(remove_spines=('right', 'top'),
            ax=None):
    if ax is None:
        ax = plt.gca()
    for r in remove_spines:
        ax.spines[r].set_visible(False)
        
def axis_off(xy, ax=None):
    if ax is None:
        ax = plt.gca()
    if 'x' in xy:
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
    if 'y' in xy:        
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)

def tick_color(xy, ticks, labels, colors):
    def set_tick_colors(ticks):
        for tick, color in zip(ticks, colors):
            tick.set_color(color)

    if 'x' in xy:
        _, labels = plt.xticks(ticks, labels)
        set_tick_colors(labels)

    if 'y' in xy:
        _, labels = plt.yticks(ticks, labels)
        set_tick_colors(labels)

def ____Heatmaps____():
    pass

def cmap(name, **kw):
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap    
    
    if name == 'BuRd':
        cmap = ListedColormap(np.flip(mpl.cm.RdBu(range(256)), axis=0))
    else:
        cmap = plt.cmap(name, **kw)
        
    return cmap

def imshow_discrete(x, shade=None, 
                    colors=[[1,0,0],[0,1,0],[0,0,1]], 
                    color_shade=[1,1,1],
                    **kw):
    if shade is None:
        shade = np.ones(list(x.shape[:-1]) + [1])
    else:
        shade = shade[:,:,np.newaxis]
    
    n_color = len(colors)
    c = np.zeros(list(x.shape) + [len(colors[0])])
    for i_color in range(n_color):
        incl = np.float32(x == i_color)[:,:,np.newaxis]
        c += incl * shade * np.float32(np2.vec_on(colors[i_color], 2, 3)) \
            + incl * (1. - shade) * np.float32(np2.vec_on(color_shade, 2, 3))
        
    plt.imshow(c, **kw)


def plot_pcolor(x, y, c=None, norm=None, **kwargs):
    """
    Parametric color line.

    Based on https://scipy-cookbook.readthedocs.io/items/Matplotlib_MulticoloredLine.html
    :param x: 1D array
    :type x: np.ndarray
    :param y: 1D array
    :type y: np.ndarray
    :param c: 1D array
    :type c: np.ndarray
    :type cmap: mpl.colors.Colormap
    :param kwargs:
    :rtype: mpl.collections.LineCollection
    """
    from matplotlib.collections import LineCollection

    n = x.shape[0]
    if c is None:
        c = np.arange(n)
    if norm is None:
        norm = plt.Normalize(c.min(), c.max())

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, norm=norm, **kwargs)
    lc.set_array(c)
    plt.gca().add_collection(lc)
    return lc


def plotmulti(xs, ys, cmap='coolwarm', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    n = ys.shape[0]
    if xs.ndim == 1:
        xs = np.tile(xs[None, :], [n, 1])
    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, n))
    h = []
    for x, y, color in zip(xs, ys, colors):
        h.append(ax.plot(x, y, color=color, **kwargs))
    return h


def multiline(xs, ys, c=None, ax=None, **kwargs):
    """
    Plot lines with different colorings

    Adapted from: https://stackoverflow.com/a/50029441/2565317

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    EXAMPLE:
        xs = [[0, 1],
              [0, 1, 2]]
        ys = [[0, 0],
              [1, 2, 1]]
        c = [0, 1]

        lc = multiline(xs, ys, c, cmap='bwr', lw=2)

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """
    from matplotlib.collections import LineCollection

    # find axes
    ax = plt.gca() if ax is None else ax

    n = len(xs)
    if c is None:
        c = np.linspace(0, 1, n)

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def ____Errorbar____():
    pass

def errorbar_shade(x, y, yerr=None, **kw):
    if yerr is None:
        y1 = y[0,:]
        y2 = y[1,:]
    else:
        if yerr.ndim == 1:
            yerr = np.concatenate([-yerr[np.newaxis,:],
                                   yerr[np.newaxis,:]], axis=0)
        elif yerr.ndim == 2:
            # assume yerr[0,:] = err_low, yerr[1,:] = err_high
            # (both positive), as in plt.errorbar
            raise NotImplementedError()

        else:
            raise ValueError()
        
        y1 = y + yerr[0,:]
        y2 = y + yerr[1,:]
        
        # print([x, y, y1, y2])
    
    if not ('alpha' in kw.keys()):
        kw['alpha'] = 0.2
    
    h = plt.fill_between(x, y1, y2, **kw)
    return h

def ____Psychophysics____():
    pass

def plot_binned_ch(x0, ch, n_bin=9, **kw):
    ix, x = np2.quantilize(x0, n_quantile=n_bin, return_summary=True)
    p = npg.aggregate(ix, ch, func='mean')
    sd = npg.aggregate(ix, ch, func='std')
    n = npg.aggregate(ix, 1, func='sum')
    se = sd / np.sqrt(n)
    
    h = plt.errorbar(x, p, yerr=se, **kw)
    
    return h, x, p, se

def ____Stats_Probability____():
    pass

def ecdf(x0, *args, **kw):
    p, x = np2.ecdf(x0)
    return plt.step(np.concatenate([x[:1], x], 0),
                    np.concatenate([np.array([0.]), p], 0),
                    *args, **kw)

def ____Gaussian____():
    pass

def plot_centroid(mu=np.zeros(2), sigma=np.eye(2),
                  add_axis=True, *args, **kwargs):
    th = np.linspace(0, 2*np.pi, 100)[np.newaxis,:]
    u, s, _ = np.linalg.svd(sigma)
    x = np.concatenate((np.cos(th), np.sin(th)), axis=0)
    us = u @ np.diag(np.sqrt(s))
    x = us @ x + mu[:,np.newaxis]
    h = plt.plot(x[0,:], x[1,:], *args, **kwargs)
    res = {
        'u':u,
        's':s,
        'us':us,
        'x':x,
    }

    if add_axis:
        axis0 = us @ np.array([1, 0]) + mu
        axis1 = us @ np.array([0, 1]) + mu
        h0 = plt.plot([mu[0], axis0[0]], [mu[1], axis0[1]], *args, **kwargs)
        h1 = plt.plot([mu[0], axis1[0]], [mu[1], axis1[1]], *args, **kwargs)
        res['axis0'] = axis0
        res['axis1'] = axis1
        res['h_axis0'] = h0
        res['h_axis1'] = h1

    return h, res

def ____Window_Management____():
    pass

def use_interactive():
    mpl.use('Qt5Agg')

def get_screen_size():
    from PyQt5 import QtGui
    return QtGui.QGuiApplication.screens()[0].geometry().getRect()[2:4]

def subfigureRC(nr, nc, r, c, set_size=False, fig=None):
    from PyQt5.QtCore import QRect
    siz = np.array(get_screen_size())
    siz1 = siz / np.array([nc, nr])
    st = siz1 * np.array([c-1, r-1])
    if fig is None:
        fig = plt.gcf()
    mgr = fig.canvas.manager
    if set_size:
        mgr.window.setGeometry(QRect(st[0], st[1], siz1[0], siz1[1]))
    else:
        c_size = mgr.window.geometry().getRect()
        mgr.window.setGeometry(QRect(st[0], st[1], c_size[2], c_size[3]))

def ____FILE_IO____():
    pass


def fig2array(fig, dpi=150):
    """
    Returns an image as numpy array from figure

    Adapted from: https://stackoverflow.com/a/58641662/2565317
    :type fig: plt.Figure
    :type dpi: int
    :rtype: np.ndarray
    """
    import io
    import cv2
    import numpy as np

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def arrays2gif(arrays, file='ani.gif', duration=100, loop=0,
                     **kwargs):
    """
    Convert numpy arrays, as from fig2array(), into gif.
    :param arrays: arrays[i]: an [height,width,color]-array, e.g.,
    from fig2array()
    :type arrays: Union[Iterable, np.ndarray]
    :param duration: of each frame, in ms
    :param loop: 0 to loop forever, None not to loop
    :rtype: Iterable[PIL.Image]
    """
    from PIL import Image, ImageDraw

    height, width, n_channel = arrays[0].shape
    images = []
    for arr in arrays:
        images.append(Image.fromarray(arr))

    kwargs.update({
        'duration': duration,
        'loop': loop
    })
    if kwargs['loop'] is None:
        kwargs.pop('loop')

    images[0].save(
        file,
        save_all=True,
        append_images=images[1:],
        **kwargs
    )
    return images


def convert_movie(src_file, ext_new='.mp4'):
    """
    Adapted from: https://stackoverflow.com/a/40726572/2565317
    :param src_file: path to the source file, including extension
    :type src_file: str
    :type ext_new: str
    """
    import moviepy.editor as mp
    import os

    clip = mp.VideoFileClip(src_file)
    pth, _ = os.path.splitext(src_file)
    clip.write_videofile(pth + ext_new)

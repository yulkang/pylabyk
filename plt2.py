#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:42:06 2018

@author: yulkang
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

from typing import Union, List, Iterable, Callable, Sequence, Mapping, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
from matplotlib.colors import ListedColormap
from typing import Union, Iterable
from copy import deepcopy, copy

import numpy_groupies as npg

from . import np2


def ____Subplots____():
    pass


AxesArray = Mapping[Tuple[Union[int, slice], ...], plt.Axes]


def supxy(axs: AxesArray, xprop=0.5, yprop=0.5) -> Tuple[float, float]:
    rect_nw = axs[0, 0].get_position().bounds
    rect_ne = axs[0, -1].get_position().bounds
    rect_sw = axs[-1, 0].get_position().bounds

    x0 = rect_nw[0]
    y0 = rect_sw[1]
    x1 = rect_ne[0] + rect_ne[2]
    y1 = rect_ne[1] + rect_ne [3]

    return (x1 - x0) * xprop + x0, (y1 - y0) * yprop + y0


AxesSlice = Union[plt.Axes, Sequence[plt.Axes], np.ndarray, AxesArray]


class GridAxes:
    def __init__(self,
                 nrows: int, ncols: int,
                 left=0.5, right=0.1,
                 bottom=0.5, top=0.5,
                 wspace: Union[float, Iterable[float]] = 0.25,
                 hspace: Union[float, Iterable[float]] = 0.25,
                 widths: Union[float, Iterable[float]] = 1.,
                 heights: Union[float, Iterable[float]] = 0.75,
                 kw_fig=(),
                 close_on_del=True,
                 ):
        """
        Give all size arguments in inches. top and right are top and right
        margins, rather than top and right coordinates.

        Figure is deleted when the GridAxes object is garbage-collected to
        prevent memory leak from opening too many figures.
        So the object needs to be returned for the figure to be saved,
        unless close_on_del=False.

        2D slices of a GridAxes object is itself a GridAxes within the same
        figure. This allows, e.g., column title across multiple columns with
        axs[:, 2:4].suptitle('Titles for columns 2-3')

        TODO: create GridAxes without creating a new figure by anchoring to
          a specified point in the provided figure. This will allow multiple
          grids to coexist in the same figure. (e.g., 2- and 3-column grids.)

        :param nrows:
        :param ncols:
        :param left:
        :param right:
        :param bottom:
        :param top:
        :param wspace:
        :param hspace:
        :param widths: widths of columns in inches.
        :param heights: heights of rows in inches.
        :param kw_fig:
        :return: axs[row, col] = plt.Axes
        """

        wspace = np.zeros([ncols - 1]) + wspace
        hspace = np.zeros([nrows - 1]) + hspace

        w = np.zeros([ncols * 2 + 1])
        h = np.zeros([nrows * 2 + 1])

        w[2:-1:2] = wspace
        h[2:-1:2] = hspace

        widths = np.zeros([ncols]) + widths
        heights = np.zeros([nrows]) + heights

        w[1::2] = widths
        h[1::2] = heights

        w[0] = left
        w[-1] = right
        h[-1] = bottom
        h[0] = top

        self._close_on_del = close_on_del

        fig = plt.figure(**{
            **dict(kw_fig),
            'figsize': [w.sum(), h.sum()]
        })
        gs = plt.GridSpec(
            nrows=nrows * 2 + 1, ncols=ncols * 2 + 1,
            left=0, right=1, bottom=0, top=1,
            wspace=0, hspace=0,
            width_ratios=w, height_ratios=h,
            figure=fig
        )
        self.gs = gs  # for backward compatibility

        axs = np.empty([nrows, ncols], dtype=np.object)

        for row in range(nrows):
            for col in range(ncols):
                axs[row, col] = plt.subplot(gs[row * 2 + 1, col * 2 + 1])

        self.axs = axs

    @property
    def w(self) -> np.array:
        w = [0.]
        for ax in self.axs[0, :]:
            bounds = ax.get_position().bounds
            w += [bounds[0], bounds[0] + bounds[2]]
        w.append(1.)
        return np.diff(w)

    @property
    def h(self) -> np.array:
        h = [0.]
        for ax in self.axs[:, 0]:
            bounds = ax.get_position().bounds
            h += [bounds[1], bounds[1] + bounds[3]]
        h.append(1.)
        # coord from the top
        return np.flip(np.diff(h))

    def copy(self):
        gridaxes = copy(self)
        gridaxes._close_on_del = self._close_on_del
        return gridaxes

    @property
    def top(self):
        return self.h[0] * self.figure.get_size_inches()[1]

    @property
    def bottom(self):
        return self.h[-1] * self.figure.get_size_inches()[1]

    @property
    def left(self):
        return self.w[0] * self.figure.get_size_inches()[0]

    @property
    def right(self):
        return self.w[-1] * self.figure.get_size_inches()[0]

    @property
    def hspace(self):
        return self.h[2:-2:2] * self.figure.get_size_inches()[1]

    @property
    def wspace(self):
        return self.w[2:-2:2] * self.figure.get_size_inches()[0]

    @property
    def widths(self):
        return self.w[1::2] * self.figure.get_size_inches()[0]

    @property
    def heights(self):
        return self.h[1::2] * self.figure.get_size_inches()[1]

    @property
    def nrows(self):
        return self.axs.shape[0]

    @property
    def ncols(self):
        return self.axs.shape[1]

    def __getitem__(self, key):
        axs = self.axs[key]

        if isinstance(axs, np.ndarray) and axs.ndim == 2:
            gridaxes = self.copy()
            gridaxes.axs = axs
            return gridaxes
        else:
            return axs

    def __setitem__(self, key, data: AxesSlice):
        self.axs[key] = data

    def flatten(self) -> Sequence[plt.Axes]:
        return self.axs.flatten()

    @property
    def figure(self) -> plt.Figure:
        return self.axs[0, 0].figure

    def __del__(self):
        """Close figure to prevent memory leak"""
        if self._close_on_del:
            fig = self.axs[0, 0].figure
            import sys
            if sys.getrefcount(fig) == 0:
                plt.close(fig)
                print('Closed figure %d!' % id(fig))  # CHECKING

    def supxy(self, xprop=0.5, yprop=0.5):
        return supxy(self.axs[:], xprop=xprop, yprop=yprop)

    @property
    def supheight(self):
        return self.supxy(yprop=1)[1] - self.supxy(yprop=0)[1]

    @property
    def supwidth(self):
        return self.supxy(xprop=1)[0] - self.supxy(xprop=0)[0]

    def suptitle(self, txt: str,
                 xprop=0.5, pad=0.05, fontsize=12, yprop=None,
                 va='bottom', ha='center',
                 **kwargs):
        if yprop is None:
            yprop = 1. + pad

        return plt.figtext(
            *self.supxy(xprop=xprop, yprop=yprop), txt,
            ha=ha, va=va, fontsize=fontsize, **kwargs)

    @property
    def shape(self):
        return self.axs.shape


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


def coltitle(col_titles, axes):
    """
    :param col_titles: list of string row title
    :type col_titles: Iterable[str]
    :param axes: 2-D array of axes, as from subplotRCs()
    :type axes: Iterable[Iterable[plt.Axes]]
    :return: array of title handles
    """
    h = []
    for ax, col in zip(axes[0,:], col_titles):
        h.append(ax.set_title(col))
    return np.array(h)


def rowtitle(row_titles, axes, pad=5, ha='right', **kwargs):
    """
    :param row_titles: list of string row title
    :type row_titles: Iterable[str]
    :param axes: 2-D array of axes, as from subplotRCs()
    :type axes: Iterable[Iterable[plt.Axes]]
    :param pad: in points.
    :type pad: float
    :return: n_rows array of row title handles
    adapted from: https://stackoverflow.com/a/25814386/2565317
    """
    from matplotlib.transforms import offset_copy

    labels = []
    for ax, row in zip(axes[:, 0], row_titles):
        label = ax.annotate(
            row,
            xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha=ha, va='center', **kwargs)
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


def break_axis(amin, amax=None, xy='x', ax=None, fun_draw=None):
    """
    @param amin: data coordinate to start breaking from
    @type amin: Union[float, int]
    @param amax: data coordinate to end breaking at
    @type amax: Union[float, int]
    @param xy: 'x' or 'y'
    @type ax: plt.Axes
    @param fun_draw: if not None, fun_draw(ax1) and fun_draw(ax2) will
    be run to recreate ax. Use the same function as that was called for
    with ax. Use, e.g., fun_draw=lambda ax: ax.plot(x, y)
    @type fun_draw: function
    @return: axs: a list of axes created
    @rtype: List[plt.Axes, plt.Axes]
    """
    from copy import copy
    from matplotlib.transforms import Bbox

    if amax is None:
        amax = amin

    if ax is None:
        ax = plt.gca()

    axs = []
    if xy == 'x':
        rect = ax.get_position().bounds
        lim = ax.get_xlim()
        prop_min = (amin - lim[0]) / (lim[1] - lim[0])
        prop_max = (amax - lim[0]) / (lim[1] - lim[0])
        rect1 = np.array([
            rect[0],
            rect[1],
            rect[2] * prop_min,
            rect[3]
        ])
        rect2 = [
            rect[0] + rect[2] * prop_max,
            rect[1],
            rect[2] * (1 - prop_max),
            rect[3]
        ]

        fig = ax.figure  # type: plt.Figure
        ax1 = fig.add_axes(plt.Axes(fig=fig, rect=rect1))
        ax1.update_from(ax)
        if fun_draw is not None:
            fun_draw(ax1)
        ax1.set_xticks(ax.get_xticks())
        ax1.set_xlim(lim[0], amin)
        ax1.spines['right'].set_visible(False)

        ax2 = fig.add_axes(plt.Axes(fig=fig, rect=rect2))
        ax2.update_from(ax)
        if fun_draw is not None:
            fun_draw(ax2)
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xlim(amax, lim[1])
        ax2.spines['left'].set_visible(False)
        ax2.set_yticks([])

        ax.set_visible(False)
        # plt.show()  # CHECKED
        axs = [ax1, ax2]

    elif xy == 'y':
        raise NotImplementedError()
    else:
        raise ValueError()

    return axs


def sameaxes(ax: Union[AxesArray, GridAxes],
             ax0: plt.Axes = None, xy='xy'):
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
    if type(ax) is np.ndarray or type(ax) is GridAxes:
        ax = ax.flatten()

    def cat_lims(lims):
        return np.concatenate([np.array(v1).reshape(1,2) for v1 in lims])

    lims_res = []
    for xy1 in xy:
        if ax0 is None:
            if xy1 == 'x':
                lims = cat_lims([ax1.get_xlim() for ax1 in ax])
                is_inverted = ax[0].get_xaxis().get_inverted()
            else:
                lims = cat_lims([ax1.get_ylim() for ax1 in ax])
                is_inverted = ax[0].get_yaxis().get_inverted()
            if is_inverted:
                lims0 = [np.max(lims[:,0]), np.min(lims[:,1])]
            else:
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


def same_clim(images: Union[mpl.image.AxesImage, Iterable[plt.Axes]],
              img0: Union[mpl.image.AxesImage, plt.Axes] = None,
              clim=None):
    try:
        images = images.flatten()
    except:
        pass
    if isinstance(images[0], plt.Axes):
        axes = images
        images = []
        for ax in axes:  # type: plt.Axes
            im = ax.findobj(mpl.image.AxesImage)
            images += im

    if clim is None:
        if img0 is None:
            clims = np.array([im.get_clim() for im in images])
            clim = [np.amin(clims[:,0]), np.amax(clims[:,1])]
        else:
            if isinstance(img0, plt.Axes):
                img0 = img0.findobj(mpl.image.AxesImage)
            clim = img0.get_clim()
    for img in images:
        img.set_clim(clim)


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

def detach_axis(xy='xy', amin=0., amax=None, ax=None, spine=None):
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


def box_off(remove_spines: Union[str, Iterable[str]] = ('right', 'top'),
            remove_ticklabels=True,
            ax=None):
    """
    :param remove_spines: 'all': remove all spines and ticks; or a list
    of some/all of 'left', 'right', 'top', and/or 'bottom'.
    :type remove_spines: Union[str, Iterable[str]]
    :type ax: plt.Axes
    :return:
    """
    if ax is None:
        ax = plt.gca()  # plt.Axes
    if remove_spines == 'all':
        remove_spines = ['left', 'right', 'top', 'bottom']
        ax.set_xticks([])
        ax.set_yticks([])

    if 'left' in remove_spines:
        ax.tick_params(axis='y', length=0)
        if remove_ticklabels:
            ax.set_yticklabels([])
    if 'bottom' in remove_spines:
        ax.tick_params(axis='x', length=0)
        if remove_ticklabels:
            ax.set_xticklabels([])

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


def ____Ticks____():
    pass


def ticks(ax=None, xy='y',
          major=True,
          interval=None, format=None, length=None, **kwargs):

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator, NullFormatter)

    if ax is None:
        ax = plt.gca()

    if xy == 'x':
        axis = ax.xaxis
    elif xy == 'y':
        axis = ax.yaxis
    else:
        raise ValueError()

    if interval is not None:
        if major:
            axis.set_major_locator(MultipleLocator(interval))
        else:
            axis.set_minor_locator(MultipleLocator(interval))

    if format is None:
        format = '%g' if major else 'None'

    if format is not None:
        if format == 'None' and major:
            if xy == 'x':
                ax.set_xticklabels([])
            else:
                ax.set_yticklabels([])
        if major:
            axis.set_major_formatter(FormatStrFormatter(format))
        else:
            axis.set_minor_formatter(FormatStrFormatter(format))

    if length is not None:
        kwargs = {**kwargs, 'length': length}
    if len(kwargs) > 0:
        axis.tick_params(which='major' if major else 'minor', **kwargs)


def hide_ticklabels(xy='xy', ax=None):
    if ax is None:
        ax = plt.gca()
    if 'x' in xy:
        plt.setp(ax.get_xticklabels(), visible=False)
    if 'y' in xy:
        plt.setp(ax.get_yticklabels(), visible=False)


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


def ____Colormaps____():
    pass


CMapType = Callable[[int], Iterable[float]]


def cool2(n_lev: int) -> CMapType:
    def cmap1(lev):
        return np.linspace([0.4, 0., 1.], [1., 0., 0.], n_lev)[lev]
    return cmap1


def cool2_rev(n_lev: int) -> CMapType:
    def cmap1(lev):
        return np.linspace([1., 0., 0.], [0.4, 0., 1.], n_lev)[lev]
    return cmap1


def winter2(n_lev: int) -> CMapType:
    def cmap1(lev):
        return np.linspace([0., 0.4, 1.], [0., 0.8, 0.25], n_lev)[lev]
    return cmap1


def winter2_rev(n_lev: int) -> CMapType:
    def cmap1(lev):
        return np.linspace([0., 0.8, 0.25], [0., 0.4, 1.], n_lev)[lev]
    return cmap1


def ____Heatmaps____():
    pass


# # 'BuRd': use plt.get_cmap('RdBu_rev')
# def cmap(name, **kw):
#     import matplotlib as mpl
#     from matplotlib.colors import ListedColormap
#
#     if name == 'BuRd':
#         cmap = ListedColormap(np.flip(mpl.cm.RdBu(range(256)), axis=0))
#     else:
#         cmap = plt.cmap(name, **kw)
#
#     return cmap


def cmap_alpha(cmap: Union[mpl.colors.Colormap, str, Iterable[float]],
               n: int = None,
               alpha_max=1.,
               alpha_min=0.,
               ) -> ListedColormap:
    """
    Add linear alphas to a colormap

    based on https://stackoverflow.com/a/37334212/2565317

    :param cmap: cmap with alpha of 1 / cmap.N, or a color (RGB/A or str)
    :param n:
    :return: cmap
    """

    if isinstance(cmap, mpl.colors.Colormap):
        if n is None:
            n = cmap.N
        cmap0 = cmap(np.arange(n))
    else:
        if n is None:
            n = 256
        cmap0 = np.repeat(
            np.array(mpl.colors.to_rgba(cmap))[None, :],
            repeats=n, axis=0
        )
    cmap0[:, -1] = np.linspace(alpha_min, alpha_max, cmap0.shape[0])
    cmap1 = ListedColormap(cmap0)
    return cmap1


def colormap2arr(arr,cmap):
    """
    https://stackoverflow.com/a/3722674/2565317

    EXAMPLE:
    arr=plt.imread('mri_demo.png')
    values=colormap2arr(arr,cm.jet)
    # Proof that it works:
    plt.imshow(values,interpolation='bilinear', cmap=cm.jet,
               origin='lower', extent=[-3,3,-3,3])
    plt.show()

    :param arr:
    :param cmap:
    :return:
    """

    import scipy.cluster.vq as scv

    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    gradient = cmap(np.linspace(0.0,1.0,100))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2 = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code, dist = scv.vq(arr2,gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values = code.astype('float')/gradient.shape[0]

    # Reshape values back to (240,240)
    values = values.reshape(arr.shape[0],arr.shape[1])
    values = values[::-1]
    return values


def imshow_discrete(x, shade=None,
                    colors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                    color_shade=(1, 1, 1),
                    **kw):
    """
    Given index x[row, col], show color[x[row, col]]
    :param x:
    :param shade: Weight given to the foreground color.
    :param colors: colors[i]: (R,G,B)
    :param color_shade: Background color.
    :param kw: Keyword arguments for imshow
    :return:
    """
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


def imshow_weights(
        w, colors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        color_bkg=(1, 1, 1),
        **kwargs
):
    """
    color[row, column] = sum_i(w[row, column, i] * (colors[i] - color_bkg))
    :param w: [row, column, i]: weight given to i-th color
    :type w: np.ndarray
    :param colors: [i] = [R, G, B]
    :param color_bkg:
    :return: h_imshow
    """
    assert isinstance(w, np.ndarray)
    assert w.ndim == 3
    colors = np.array(colors)
    color_bkg = np.array(color_bkg)
    dcolors = np.stack([
        c - color_bkg for c in colors
    ])[None, None, :, :]  # [1, 1, w, color]
    w = w / np.amax(np.sum(w, -1))
    color = np.sum(
        np.expand_dims(w, -1) * dcolors,
        -2
    )
    color = color + color_bkg[None, None, :]
    # color = np.concatenate([
    #     color, np.sum(w, -1, keepdims=True)
    # ], -1)
    # color = np.clip(color, 0, 1)
    # plt.gca().set_facecolor(color_bkg)
    return plt.imshow(color, **kwargs)


def plot_pcolor(x, y, c=None, norm=None, cmap=None, **kwargs):
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
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    lc = LineCollection(segments, norm=norm, cmap=cmap, **kwargs)
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


def colorbar(
        ax: plt.Axes = None,
        mappable: mpl.cm.ScalarMappable = None,
        loc='right',
        width='5%', height='100%',
        borderpad=-1,
        kw_inset=(),
        kw_cbar=(),
) -> mpl.colorbar.Colorbar:
    """
    Add a colorbar aligned to the mappable (e.g., image)

    :param ax:
    :param mappable: defaults to image in the axes
    :param loc: as for legend
    :param width: relative to the axes
    :param height: relative to the axes
    :param borderpad: relative to the fontsize of the axes.
    :param kw_inset:
    :param kw_cbar:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    if mappable is None:
        mappable = ax.findobj(mpl.image.AxesImage)[0]
    fig = ax.figure

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(
        ax, width=width, height=height, loc=loc,
        bbox_to_anchor=(0., 0., 1., 1.),
        bbox_transform=ax.transAxes,
        borderpad=borderpad,
        **dict(kw_inset)
    )
    cb = fig.colorbar(
        mappable, cax=axins,
        **dict(kw_cbar)
    )
    return cb


def ____Errorbar____():
    pass


def patch_wave(y_wave0, x_lim,
               wave_margin=0.05,
               wave_amplitude=0.05,
               width_wave=0.82,
               color='w',
               axis_wave='x',
               ax: plt.Axes = None) -> patches.Polygon:
    """
    Add a wavy occluding polygon to a bar graph to indicate out-of-limit values
    :param y_wave0:
    :param x_lim:
    :param wave_margin: relative to x_lim
    :param wave_amplitude: relative to x_lim
    :param width_wave: data unit, along y_wave0
    :param color:
    :param axis_wave: 'x' for barh; 'y' for bar (vertical)
    :param ax:
    :return:
    """

    wave_margin = x_lim * wave_margin
    wave_amplitude = -np.abs(x_lim) * wave_amplitude
    nxy_wave = 50
    x_wave = np.concatenate([
        np.array([x_lim]),
        x_lim - wave_margin - wave_amplitude
        + wave_amplitude * np.sin(np.linspace(0, 2 * np.pi, nxy_wave)),
        np.array([x_lim])
    ])
    y_wave = np.concatenate([
        np.array([y_wave0 - width_wave / 2]),
        y_wave0 + np.linspace(-1, 1, nxy_wave) * width_wave / 2,
        np.array([y_wave0 + width_wave / 2])
    ])
    xy_wave = np.stack([x_wave, y_wave], -1)
    if axis_wave == 'y':
        xy_wave = np.flip(xy_wave, -1)

    patch = patches.Polygon(
        xy_wave,
        edgecolor='None', facecolor=color)

    if ax is not None:
        ax.add_patch(patch)

    return patch


def patch_chance_level(
        level=None, signs=(-1, 1), ax: plt.Axes = None,
        xy='y', color=(0.7, 0.7, 0.7)
):
    if level is None:
        level = np.log(10.)
    if ax is None:
        ax = plt.gca()

    hs = []
    for sign in signs:
        if xy == 'y':
            if ax.yaxis.get_scale() == 'log':
                vmin = 1.
                level1 = level * 10 ** sign
            else:
                vmin = 0.
                level1 = level * sign

            lim = ax.get_xlim()
            rect = mpl.patches.Rectangle(
                [lim[0], vmin], lim[1] - lim[0], level1,
                linewidth=0,
                fc=color,
                zorder=-1
            )
        elif xy == 'x':
            if ax.xaxis.get_scale() == 'log':
                vmin = 1.
                level1 = level * 10 ** sign
            else:
                vmin = 0.
                level1 = level * sign

            lim = ax.get_ylim()
            rect = mpl.patches.Rectangle(
                [vmin, lim[0]], level1, lim[1] - lim[0],
                linewidth=0,
                fc=color,
                zorder=-1
            )
        ax.add_patch(rect)
        hs.append(rect)
    return hs


def bar_group(y: np.ndarray, yerr: np.ndarray = None,
              width=0.8, width_indiv=1.,
              cmap: Union[
                  mpl.colors.Colormap,
                  Iterable[Union[str, Iterable[float]]]
              ] = None,
              kw_color=('color',),
              **kwargs) -> (List[mpl.container.BarContainer], np.ndarray):
    """

    :param y: [x, series]
    :param yerr: [x, series]
    :param width: distance between centers of the 1st & last bars in a group
    :param width_indiv: individual bar's width in proportion to the distance
    between the left edge of neighboring bars within a group
    :param gap: proportion of the gap between bars within a group
    :param cmap: cmap or list of colors
    :param kw_color: tuple of keyword(s) to use the series color
    :param kwargs: fed to bar()
    :return: hs[series] = BarContainer, xs[x, series]
    """
    n = y.shape[0]
    m = y.shape[1]
    x = np.arange(n)
    xs = []

    if yerr is None:
        yerr = np.zeros_like(y) + np.nan

    if cmap is None:
        cmap = plt.get_cmap('tab10')
    elif not isinstance(cmap, mpl.colors.Colormap):
        cmap = mpl.colors.ListedColormap(cmap)

    width1 = width * width_indiv / m
    width0 = width * (m - 1) / m

    hs = []
    for i, (y1, yerr1) in enumerate(zip(y.T, yerr.T)):
        if isinstance(cmap, mpl.colors.ListedColormap):
            color = cmap(i)
        else:
            color = cmap(i / (m - 1))
        dx = (i / (m - 1) - 0.5) * width0
        kw = {k: color for k in kw_color}
        h = plt.bar(
            x + dx, height=y1, yerr=yerr1, width=width1,
            **{**kwargs, **kw})
        hs.append(h)
        xs.append(x + dx)

    xs = np.stack(xs, -1)
    return hs, xs


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

def ____ANIMATION____():
    pass


def fig2array(fig, dpi=None):
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

    if dpi is None:
        dpi = fig.dpi

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

    import os
    pth, _ = os.path.splitext(src_file)

    dst_file = pth + ext_new
    from send2trash import send2trash
    if os.path.exists(dst_file):
        send2trash(dst_file)

    import moviepy.editor as mp
    clip = mp.VideoFileClip(src_file)
    clip.write_videofile(dst_file)  # , write_logfile=True)

    # import ffmpy
    # ff = ffmpy.FFmpeg(
    #     inputs={src_file: None},
    #     outputs={dst_file: None}
    # )
    # ff.run()


class Animator:
    def __init__(self):
        self.frames = []

    def append_fig(self, fig: plt.Figure):
        self.frames.append(fig2array(fig))

    def export(self, file, ext=('.gif', '.mp4'), duration=100, loop=0,
               kw_gif=()) -> Iterable[str]:
        if type(ext) is str:
            ext = [ext]

        import os
        file_wo_ext = os.path.splitext(file)[0]
        file_gif = file_wo_ext + '.gif'

        files = [file_gif]

        arrays2gif(self.frames, file_gif,
                   duration=duration, loop=loop,
                   **dict(kw_gif))

        for ext1 in ext:
            if ext1 != '.gif':
                convert_movie(file_gif, ext1)
                files.append(file_wo_ext + ext1)

        if '.gif' not in ext:
            os.remove(file_gif)
            files = files[1:]

        return files
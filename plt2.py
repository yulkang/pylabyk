#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:42:06 2018

@author: yulkang
"""
#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.
import os
from typing import List, Callable, Sequence, Mapping, Tuple, Dict, Any, Type
import pickle

import PIL
import numpy as np
from PIL import Image
from numpy import typing as nptyp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches, gridspec, pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Union, Iterable
from copy import copy
import pylatex as ltx
import numpy_groupies as npg
from pdf2image import convert_from_path
from scipy.stats import gaussian_kde

from . import np2, plt_network as pltn
from .cacheutil import mkdir4file, Cache
from .argsutil import fname2title


def ____Settings____():
    pass


def rc_dpi(dpi=300, rc: Callable = None):
    """
    Set default DPI for pyplot
    :param dpi:
    :param rc: (optional) pass the function, pyplot.rc
    """
    if rc is None:
        rc = plt.rc
    rc('figure', dpi=dpi)


def rc_sanslatex(rc: Callable = None):
    """
    Use sans-serif font in Latex
    :param rc: (optional) pass the function, pyplot.rc
    """
    if rc is None:
        rc = plt.rc

    rc('text', usetex=True)
    rc('font', family='sans-serif')
    rc(
        'text.latex', preamble='\n'.join(
            [
                r'\usepackage{amsmath}',
                # r'\usepackage{siunitx}',
                # # i need upright \micro symbols, but you need...
                # r'\sisetup{detect-all}',
                # ...this to force siunitx to actually use your fonts
                r'\usepackage{helvet}',  # set the normal font here
                r'\usepackage{sansmath}',
                # load up the sansmath so that math -> helvet
                r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
            ]
        )
    )


def ____Subplots____():
    pass


AxesArray = Mapping[Tuple[Union[int, slice], ...], plt.Axes]


def supxy(axs: AxesArray, xprop=0.5, yprop=0.5) -> Tuple[float, float]:
    rect_sw = axs[-1, 0].get_position().bounds
    rect_ne = axs[0, -1].get_position().bounds

    x0 = rect_sw[0]
    y0 = rect_sw[1]
    x1 = rect_ne[0] + rect_ne[2]
    y1 = rect_ne[1] + rect_ne[3]

    return (x1 - x0) * xprop + x0, (y1 - y0) * yprop + y0


def get_size_inches(ax: Union[plt.Figure, plt.Axes, 'GridAxes']) -> Tuple[float, float]:
    if isinstance(ax, plt.Axes):
        # Get the bounding box of the Axes in figure-relative coordinates
        bbox = ax.get_position()

        # Get the figure size in inches
        fig_size = ax.get_figure().get_size_inches()

        # Compute the Axes size in inches
        axes_width = bbox.width * fig_size[0]
        axes_height = bbox.height * fig_size[1]
    else:
        axes_width, axes_height = ax.get_size_inches()

    return axes_width, axes_height


AxesSlice = Union[plt.Axes, Sequence[plt.Axes], np.ndarray, AxesArray]


class GridAxes:
    def __init__(
        self,
        nrows=1, ncols=1,
        left=0.5, right=0.1,
        bottom=0.5, top=0.5,
        wspace: Union[float, Sequence[float]] = 0.25,
        hspace: Union[float, Sequence[float]] = 0.25,
        widths: Union[float, Sequence[float]] = 1.,
        heights: Union[float, Sequence[float]] = 0.75,
        ha='center',
        va='center',
        kw_fig=(),
        kw_subplot: Union[None, Sequence[Sequence[Dict[str, Any]]]] = None,
        close_on_del=True,
        parent: Union[plt.Axes, plt.Figure, 'GridAxes'] = None,
        # close_on_del=False,  # CHECKED
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
        :param ha: horizontal alignment of the grid within the parent.
        :param va: vertical alignment of the grid within the parent.
        :param kw_fig:
        :param kw_subplot: if not None, will be converted & broadcasted
            to an array of size [row, col] of kwargs for each subplot.
            e.g.: kw_subplot=[[{}, {}, {'projection': 'polar'}]]
        :param parent: if given, all units are relative to the size
            of the parent, rather than in inches.
        :return: axs[row, col] = plt.Axes
        """
        if kw_subplot is None:
            kw_subplot = [[{}]]
        kw_subplot = np.broadcast_to(np.array(kw_subplot), [nrows, ncols])

        # repeat and truncate if too long for convenience
        wspace, hspace, widths, heights = [
            np.array(v)[np.arange(l) % len(v)]
            if np2.is_sequence(v) and len(v) != l
            else v
            for v, l in [
                (wspace, ncols - 1),
                (hspace, nrows - 1),
                (widths, ncols),
                (heights, nrows)
            ]
        ]

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

        if parent is None:
            parent = plt.figure(
                **{
                    **dict(kw_fig),
                    'figsize': [w.sum(), h.sum()]
                }
            )

        kw_gridspec = dict(
            nrows=nrows * 2 + 1, ncols=ncols * 2 + 1,
            wspace=0, hspace=0,
            width_ratios=w, height_ratios=h,
        )

        self.ha = ha
        self.va = va

        w_inch, h_inch = get_size_inches(parent)
        aspect_parent = w_inch / h_inch
        w_sum_self = np.sum(w)
        h_sum_self = np.sum(h)
        aspect_self = w_sum_self / h_sum_self
        if aspect_parent > aspect_self:
            w_without_margin = h_inch * aspect_self
            extra_prop = (w_inch - w_without_margin) / w_without_margin
            total_w_prop_with_extra = 1. + extra_prop
            total_h_prop_with_extra = 1.
            extra_top = 0.
            extra_bottom = 0.
            if self.ha == 'left':
                extra_left = 0.
                extra_right = extra_prop
            elif self.ha == 'center':
                extra_left = extra_prop / 2
                extra_right = extra_prop / 2
            elif self.ha == 'right':
                extra_left = extra_prop
                extra_right = 0.
            else:
                raise ValueError(f'Unknown horizontal alignment: {self.ha}')
        elif aspect_parent < aspect_self:
            h_without_margin = w_inch / aspect_self
            extra_prop = (h_inch - h_without_margin) / h_without_margin
            total_w_prop_with_extra = 1.
            total_h_prop_with_extra = 1. + extra_prop
            extra_left = 0.
            extra_right = 0.
            if self.va == 'top':
                extra_top = 0.
                extra_bottom = extra_prop
            elif self.va == 'center':
                extra_top = extra_prop / 2
                extra_bottom = extra_prop / 2
            elif self.va == 'bottom':
                extra_top = extra_prop
                extra_bottom = 0.
            else:
                raise ValueError(f'Unknown vertical alignment: {self.va}')
        else:
            extra_left = 0.
            extra_right = 0.
            extra_top = 0.
            extra_bottom = 0.
            total_w_prop_with_extra = 1.
            total_h_prop_with_extra = 1.

        # print(f'{extra_left=:1.3g}, {extra_right=:1.3g}, '
        #       f'{extra_top=:1.3g}, {extra_bottom=:1.3g}')  # CHECKED

        if isinstance(parent, plt.Figure):
            # print('parent=figure')  # CHECKED
            kw_gridspec['width_ratios'][0] += extra_left
            kw_gridspec['width_ratios'][-1] += extra_right
            kw_gridspec['height_ratios'][0] += extra_top
            kw_gridspec['height_ratios'][-1] += extra_bottom

            gs = plt.GridSpec(
                **kw_gridspec,
                figure=parent,
                left=0, right=1, bottom=0, top=1,
            )
        else:
            if isinstance(parent, type(self)):
                # print('parent=GridAxes')  # CHECKED
                w_parent = parent.w.sum()
                h_parent = parent.h.sum()

                w_mid_parent = parent.w[1:-1].sum()
                h_mid_parent = parent.h[1:-1].sum()

                gs0 = plt.GridSpec(
                    nrows=1, ncols=1,
                    left=parent.w[0] / w_parent + extra_left / total_w_prop_with_extra,
                    right=parent.w[0] + w_mid_parent * (1 + extra_left) / total_w_prop_with_extra,
                    bottom=parent.h[-1] / h_parent + extra_bottom / total_h_prop_with_extra,
                    top=parent.h[-1] + h_mid_parent * (1 + extra_bottom) / total_h_prop_with_extra,
                    figure=parent.figure,
                )
            elif isinstance(parent, plt.Axes):
                # print('parent=Axes')  # CHECKED
                bbox = parent.get_position()  # position within figure
                x0, y0, w_bbox, h_bbox = bbox.bounds
                d = dict(
                    left=x0 + extra_left / total_w_prop_with_extra * w_bbox,
                    right=x0 + (extra_left + 1) / total_w_prop_with_extra * w_bbox,
                    bottom=y0 + extra_bottom / total_h_prop_with_extra * h_bbox,
                    top=y0 + (extra_bottom + 1) / total_h_prop_with_extra * h_bbox,
                )
                # if ncols == 3:
                #     print(f'{extra_left=:1.3g}, {total_w_prop_with_extra=:1.3g}')
                #     print(f'{extra_bottom=:1.3g}, {total_h_prop_with_extra=:1.3g}')
                #     from pprint import pprint
                #     pprint(d)  # CHECKED
                #     print('--')
                gs0 = plt.GridSpec(
                    nrows=1, ncols=1,
                    **d,
                    figure=parent.figure,
                )
            else:
                raise ValueError()
            gs = gridspec.GridSpecFromSubplotSpec(
                subplot_spec=gs0[0],
                **kw_gridspec,
            )

        self.gs = gs  # for backward compatibility

        axs = np.empty([nrows, ncols], dtype=object)
        figure = parent if isinstance(parent, plt.Figure) else parent.figure

        for row in range(nrows):
            for col in range(ncols):
                # DEBUGGED: this ensures that the axes from GridSpecFromSubplotSpec()
                #   is added to the parent's figure
                axs[row, col] = figure.add_subplot(
                    gs[row * 2 + 1, col * 2 + 1],
                    **kw_subplot[row, col]
                )
                assert axs[row, col].figure == figure

        self.axs = axs
        self.axs_array = np.empty([nrows, ncols], dtype=object)  # can be used to store children when panels are used as parent

    def get_size_inches(self) -> Tuple[float, float]:
        return np.sum(self.w), np.sum(self.h)

    def get_bounds_inner_in_proportion_figure(self) -> Tuple[float, float, float, float]:
        """

        :return: (left, bottom, width, height) in proportion to the figure
        """
        return (
            self.w[0] / np.sum(self.w),
            self.h[-1] / np.sum(self.h),
            np.sum(self.w[1:-1]) / np.sum(self.w),
            np.sum(self.h[1:-1]) / np.sum(self.h),
        )

    @property
    def w(self) -> np.array:
        """left, width[0], wspace[0], width[1], ..., right (inches)"""
        w = [0.]
        for ax in self.axs[0, :]:
            bounds = ax.get_position().bounds
            w += [bounds[0], bounds[0] + bounds[2]]
        w.append(1.)
        return np.diff(w) * self.figure.get_size_inches()[0]

    @property
    def h(self) -> np.array:
        """top, height[0], hspace[0], height[1], ..., bottom (inches)"""
        h = [0.]
        for ax in np.flip(self.axs[:, 0]):
            bounds = ax.get_position().bounds
            h += [bounds[1], bounds[1] + bounds[3]]
        h.append(1.)
        # coord from the top
        return np.flip(np.diff(h)) * self.figure.get_size_inches()[1]

    def copy(self):
        gridaxes = copy(self)
        gridaxes._close_on_del = self._close_on_del
        return gridaxes

    @property
    def top(self):
        return self.h[0]

    @property
    def bottom(self):
        return self.h[-1]

    @property
    def left(self):
        return self.w[0]

    @property
    def right(self):
        return self.w[-1]

    @property
    def hspace(self):
        return self.h[2:-2:2]

    @property
    def wspace(self):
        return self.w[2:-2:2]

    @property
    def widths(self):
        return self.w[1::2]

    @property
    def heights(self):
        return self.h[1::2]

    @property
    def nrows(self):
        return self.axs.shape[0]

    @property
    def ncols(self):
        return self.axs.shape[1]

    def __getitem__(self, key) -> Union[AxesArray, AxesSlice, 'GridAxes']:
        axs = self.axs[key]

        if isinstance(axs, np.ndarray) and axs.ndim == 2:
            gridaxes = self.copy()
            gridaxes._close_on_del = False
            # to avoid closing the figure of the parent
            gridaxes.axs = axs
            gridaxes.axs_array = gridaxes.axs_array[key]
            return gridaxes

        if isinstance(axs, GridAxes):
            axs._close_on_del = False

        return axs

    def __setitem__(self, key, data: AxesSlice):
        self.axs[key] = data

    # def __len__(self) -> int:  # DEBUGGED: Don't use. Breaks setting .axs_array
    #     return self.nrows

    def flatten(self) -> Sequence[plt.Axes]:
        return self.axs.flatten()

    @property
    def figure(self) -> plt.Figure:
        return self.axs[0, 0].figure

    def close(
        self,
        # force=True
    ):
        fig = self.axs[0, 0].figure
        # import sys
        # if sys.getrefcount(fig) == 0 or force:
        plt.close(fig)
        # print('Closed figure %d!' % id(fig))  # CHECKED

    def __del__(self):
        """Close figure to prevent memory leak"""
        if self._close_on_del:
            self.close(
                # force=False
            )

    def supxy(self, xprop=0.5, yprop=0.5):
        return supxy(self.axs[:], xprop=xprop, yprop=yprop)

    @property
    def supheight(self):
        return self.supxy(yprop=1)[1] - self.supxy(yprop=0)[1]

    @property
    def supwidth(self):
        return self.supxy(xprop=1)[0] - self.supxy(xprop=0)[0]

    def suptitle(
        self, txt: str,
        preset='top',
        pad=0.5, fontsize=12,
        xprop=None, yprop=None,
        va=None, ha=None,
        rotation=None,
        **kwargs
    ):
        """

        :param txt:
        :param preset: 'top'|'left'
        :param pad: inches
        :param xprop:
        :param yprop:
        :param va:
        :param ha:
        :param fontsize:
        :param kwargs: fed to figtext()
        :return:
        """
        if preset == 'top':
            if xprop is None:
                xprop = 0.5
            if yprop is None:
                height_axes = np.sum(self.h[1:-1])
                yprop = 1. + pad / height_axes
            if va is None:
                va = 'bottom'
            if ha is None:
                ha = 'center'
            if rotation is None:
                rotation = 0
        elif preset == 'left':
            if xprop is None:
                xprop = -pad / np.sum(self.w[1:-1])
            if yprop is None:
                yprop = 0.5
            if va is None:
                va = 'center'
            if ha is None:
                ha = 'right'
            if rotation is None:
                rotation = 90
        else:
            raise ValueError()

        return self.figure.text(
            *self.supxy(xprop=xprop, yprop=yprop), txt,
            ha=ha, va=va, fontsize=fontsize, rotation=rotation,
            figure=self.figure,
            **kwargs
        )
        # return plt.figtext(
        #     *self.supxy(xprop=xprop, yprop=yprop), txt,
        #     ha=ha, va=va, fontsize=fontsize,
        #     figure=self.figure,
        #     **kwargs)

    def rowtitle(self, row_titles: Sequence[str], **kwargs):
        return rowtitle(row_titles, self.axs, **kwargs)

    def coltitle(self, col_titles: Sequence[str], **kwargs):
        return coltitle(col_titles, self.axs, **kwargs)

    @property
    def shape(self):
        return self.axs.shape


def subplotRC(nrow, ncol, row, col, **kwargs):
    iplot = (row - 1) * ncol + col
    ax = plt.subplot(nrow, ncol, iplot, **kwargs)
    return ax


def subplotRCs(nrow, ncol, **kwargs):
    ax = np.empty([nrow, ncol], dtype=object)
    for row in range(1, nrow + 1):
        for col in range(1, ncol + 1):
            ax[row - 1, col - 1] = subplotRC(nrow, ncol, row, col, **kwargs)
    return ax


def coltitle(
    col_titles: Sequence[str],
    axes: Union[GridAxes, AxesArray, Sequence[Sequence[plt.Axes]]],
    **kwargs
):
    """
    :param col_titles: list of string row title
    :type col_titles: Iterable[str]
    :param axes: 2-D array of axes, as from subplotRCs()
    :type axes: Iterable[Iterable[plt.Axes]]
    :return: array of title handles
    """
    h = []
    for ax, col in zip(axes[0, :], col_titles):
        h.append(ax.set_title(col, **kwargs))
    return np.array(h)


def rowtitle(
    row_titles: Sequence[str],
    axes: Union[GridAxes, AxesArray, Iterable[Iterable[plt.Axes]]],
    pad=5,
    ha='right',
    **kwargs
):
    """
    :param row_titles: list of string row title
    :param axes: 2-D array of axes, as from subplotRCs()
    :param pad: in points.
    :param ha: horizontal alignment
    :param kwargs: fed to Axes.annotate()
    :return: n_rows array of row title handles
    adapted from: https://stackoverflow.com/a/25814386/2565317
    """

    labels = []
    for ax, row in zip(axes[:, 0], row_titles):
        label = ax.annotate(
            row,
            **{**dict(
                xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                fontsize='large', ha=ha, va='center',
            ), **kwargs}
        )
        labels.append(label)

    # fig = axes[0,0].get_figure()
    # fig.tight_layout()

    # # tight_layout doesn't take these labels into account. We'll need
    # # to make some room. These numbers are are manually tweaked.
    # # You could automatically calculate them, but it's a pain.
    # fig.subplots_adjust(left=0.15, top=0.95)

    return np.array(labels)


def ____Saving____():
    pass


def savefig_w_data(
    fname: str,
    fun: Callable[..., Union[
        plt.Figure, GridAxes,
        Tuple[Union[plt.Figure, GridAxes], ...]
    ]] = None,
    kw_fun: Dict[str, Any] = None,
    fun_calc: Callable[..., Dict[str, Any]] = None,
    kw_fun_calc: Dict[str, Any] = None,
    kw_fun_nocache: Dict[str, Any] = None,
    kw_savefig: Dict[str, Any] = None,
    to_load_kw_fun_only=False,
    to_overwrite_cache=False,
    to_plot=True,
    to_savefig=True,
    to_return_kw_fun=False,
    to_save_cache=True,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """

    :param fname: file name for figures.
        Omit extension to avoid overlong cache file name.
    :param fun: can be called with fun(**kw_fun),
        and returns plt.Figure or plt2.GridAxes as the only or the first output
    :param kw_fun: if None, loaded from fname.zpkl
    :param fun_calc: can be called with fun_calc(**kw_fun_calc),
        and returns kw_fun as the only output
    :param kw_fun_calc: if None, loaded from fname.zpkl
    :param kw_fun_nocache: if not None,
        passed to fun(**{**kw_fun, **kw_fun_nocache})
    :param to_load_kw_fun_only: if True, load kw_fun only
    :param to_overwrite_cache: if True, overwrite cache even if it exists
    :param to_plot: if False, just save the data without plotting
    :param to_savefig: if True (default), save the figure; otherwise close it.
    :param to_return_kw_fun: if True, return kw_fun
    :return: output of fun (, kw_fun if return_kw_fun)
    """
    if kw_savefig is None:
        kw_savefig = {}

    if to_save_cache:
        with Cache(fname + '.zpkl', ignore_key=True) as cache:
            if to_load_kw_fun_only:
                if cache.exists():
                    kw_fun = cache.get()
                else:
                    raise FileNotFoundError('Cache file does not exist.')
                return kw_fun

            if kw_fun is None:
                if cache.exists() and not to_overwrite_cache:
                    kw_fun = cache.get()
                else:
                    if kw_fun_calc is None:
                        kw_fun_calc = {}
                    kw_fun = fun_calc(**kw_fun_calc)
                    cache.set(kw_fun)
            elif to_overwrite_cache or not cache.exists():
                cache.set(kw_fun)
    # from pprint import pprint
    # pprint(kw_fun)  # CHECKED
    if to_plot:
        assert fun is not None
        if kw_fun_nocache is None:
            kw_fun_nocache = {}
        out = fun(**{**kw_fun, **kw_fun_nocache})
        if np2.is_iter(out):
            fig = out[0]
        else:
            fig = out
        if hasattr(fig, 'figure'):
            fig = fig.figure
        if to_savefig and fig is not None:
            savefig(fname, fig=fig, **kw_savefig)
            plt.close(fig.figure)
    else:
        out = None

    if to_return_kw_fun:
        return out, kw_fun
    else:
        return out


from sys import platform

ext_savefig = ('.png', '.pdf', '.svg',) if platform == 'darwin' else ('.png',)
# ext_savefig = ('.png',)  # CHECK
"""
Default extensions that can be changed using plt2.ext_savefig = default
Use .pdf only on Mac (darwin)
"""


def savefig(
    fname: str, *args,
    fig: Union[mpl.figure.Figure, GridAxes] = None,
    ext: Union[str, Iterable[str]] = None,
    to_pickle=True,
    verbose=True,
    skip_pdf_on_error=True,
    **kwargs
):
    """

    :param fname:
        if ext is not None, use it after removing recognized extensions
            from fname
        otherwise, use the extension recognized from fname
    :param args:
    :param fig:
    :param ext:
    :param to_pickle:
    :param verbose:
    :param kwargs:
    :return:
    """
    if ext is None:
        ext = ext_savefig
    if fig is None:
        fig = plt.gcf()
        fig0 = None
    else:
        if hasattr(fig, 'figure'):
            fig = fig.figure
        fig0 = plt.gcf()
        plt.figure(fig.number)

    fname11, ext11 = os.path.splitext(fname)
    recognized_ext = ext11 in ['.pdf', '.png']
    if ext is None:
        fname1 = fname11
        if recognized_ext:
            ext1 = ext11
        else:
            raise ValueError(f'Unrecognized extension: {ext11}')
    else:
        ext1 = ext
        # remove recognized extension
        if recognized_ext:
            fname1 = fname11
        else:
            fname1 = fname

    if isinstance(ext1, str):
        ext1 = (ext1,)
    else:
        assert np.all([isinstance(v, str) for v in ext1])

    mkdir4file(fname1)
    for ext11 in ext1:
        try:
            if verbose:
                print(f'Saving image to {fname1 + ext11}', end='')
            plt.savefig(fname1 + ext11, *args, **kwargs)
            if verbose:
                print(f' ..saved.')
        except RuntimeError:
            if skip_pdf_on_error and ext11 == '.pdf':
                pass
            else:
                raise
    if to_pickle:
        # manager0 = fig.canvas.manager
        # fig.canvas.manager = None  # DEBUGGED: using this line seemed to help but it now runs without it
        with open(fname1 + '.mpl', 'wb') as file:
            pickle.dump(
                fig,
                file
            )
        # fig.canvas.manager = manager0

        # # saving matplotlib version perhaps not necessary - matplotlib checks it by itself
        # with open(fname1 + '.mpl', 'wb') as file:
        #     pickle.dump(
        #         {
        #             'matplotlib.__version__': mpl.__version__,
        #             'figure': fig,
        #         },
        #         file
        #     )
        # zpkl.save({
        #     'matplotlib.__version__': mpl.__version__,
        #     'figure': fig,
        # }, fname1 + '.mpl')
        if verbose:
            print(f'Pickled figure to {fname1}.mpl')

    plt.close(fig)  # avoid memory leakage

    if fig0 is not None:
        plt.figure(fig0.number)


def loadfig(fname: str) -> mpl.figure.Figure:
    with open(fname, 'rb') as file:
        fig = pickle.load(file)

    # fig_dummy = plt.figure()  # UNUSED: using these lines seemed to help but they seem not needed any more.
    # new_manager = fig_dummy.canvas.manager
    # new_manager.canvas.figure = fig
    # fig.set_canvas(new_manager.canvas)
    return fig

    # with open(fname, 'rb') as file:
    #     v = pickle.load(file)
    # # v = zpkl.load(fname, use_torch=False)
    # if v['matplotlib.__version__'] != mpl.__version__:
    #     import warnings
    #     warnings.warn(f'Current matplotlib version ({mpl.__version__}) '
    #                   f'!= version that pickled the figure '
    #                   f'({v["matplotlib.__version__"]}) loaded from '
    #                   f'{fname}')
    # assert isinstance(v['figure'], mpl.figure.Figure)
    # return v['figure']


def ____Axes_Limits____():
    pass


def lim_margin(
    v: np.ndarray, xy='y', margin=0.05, ax: plt.Axes = None,
    err: np.ndarray = None
):
    """

    :param v: [...]
    :param xy:
    :param margin:
    :param ax:
    :param err: [...] or [..., (le, re)] where le = lb - y and re = rb - y
    :return: amin, amax
    """
    try:
        _ = margin[1]
    except TypeError:
        margin = [margin, margin]
    try:
        _ = margin[1]
    except IndexError:
        margin = list(margin) * 2
    if err is not None:
        err = np.nan_to_num(err)
        if err.ndim == 1:
            v1 = np.r_[v - err, v + err]
        else:
            assert err.ndim == 2
            v1 = np.r_[v + err[..., 0], v + err[..., -1]]
    else:
        v1 = v

    vmax = np.nanmax(v1)
    vmin = np.nanmin(v1)
    v_range = vmax - vmin
    amin = vmin - v_range * margin[0]
    amax = vmax + v_range * margin[0]
    if ax is None:
        ax = plt.gca()
    if xy == 'x':
        ax.set_xlim(amin, amax)
    elif xy == 'y':
        ax.set_ylim(amin, amax)
    else:
        raise ValueError()
    return amin, amax


def break_axis(
    amin, amax=None, xy='x', ax: plt.Axes = None,
    fun_draw: Callable[[plt.Axes], None] = None,
    margin=0.05,
    prop=0.5,
) -> (plt.Axes, plt.Axes):
    """
    :param amin: data coordinate to start breaking from
    :param amax: data coordinate to end breaking at
    :param xy: 'x' or 'y'
    :param fun_draw: if not None, fun_draw(ax1) and fun_draw(ax2) will
    be run to recreate ax. Use the same function as that was called for
    with ax. Use, e.g., fun_draw=lambda ax: ax.plot(x, y)
    :param prop: None for auto; 0.5 makes the two resulting axis to be
        of equal widths or heights
    :return: axs: a list of axes created
    """

    if amax is None:
        amax = amin

    if ax is None:
        ax = plt.gca()

    if xy == 'x':
        rect = ax.get_position().bounds
        lim = ax.get_xlim()
        if prop is None:
            prop = (amin - lim[0]) / (amin - lim[0] + lim[1] - amax)
            # prop_min = (amin - lim[0]) / (lim[1] - lim[0])
            # prop_max = (amax - lim[0]) / (lim[1] - lim[0])

        rect1 = np.array(
            [
                rect[0],
                rect[1],
                rect[2] * (prop - margin / 2),
                # rect[2] * (prop_min - margin / 2),
                rect[3]
            ]
        )
        rect2 = np.array(
            [
                rect[0] + rect[2] * (prop + margin / 2),
                # rect[0] + rect[2] * (prop_max + margin / 2),
                rect[1],
                rect[2] * (1. - prop - margin / 2),
                # rect[2] * (1 - prop_max),
                rect[3]
            ]
        )

        fig = ax.figure  # type: plt.Figure
        ax1 = fig.add_axes(plt.Axes(fig=fig, rect=rect1))
        ax1.update_from(ax)
        if fun_draw is not None:
            fun_draw(ax1)
        ax1.set_xticks(ax.get_xticks())
        # ax1.set_xlim(right=amin)
        ax1.set_xlim(lim[0], amin)
        ax1.spines['right'].set_visible(False)

        ax2 = fig.add_axes(plt.Axes(fig=fig, rect=rect2))
        ax2.update_from(ax)
        if fun_draw is not None:
            fun_draw(ax2)
        ax2.set_xticks(ax.get_xticks())
        # ax2.set_xlim(left=amax)
        ax2.set_xlim(amax, lim[1])
        ax2.spines['left'].set_visible(False)
        ax2.set_yticks([])

        xlim0 = ax.get_xlim()
        xtick0 = ax.get_xticks()
        xticklabels0 = ax.get_xticklabels()

        ylim0 = ax.get_ylim()
        ytick0 = ax.get_yticks()
        yticklabels0 = ax.get_yticklabels()

        ax.cla()
        # ax.set_xlim(xlim0)
        # ax.set_xticks(xtick0)
        # ax.set_xticklabels(xticklabels0, color='w')
        #
        # ax.set_ylim(ylim0)
        # ax.set_yticks(ytick0)
        # ax.set_yticklabels(yticklabels0, color='w')

        box_off(
            'all', ax=ax,
            # remove_ticklabels=False, remove_ticks=False,
        )
        # ax.tick_params(axis='x', color='w')
        # ax.tick_params(axis='y', color='w')
        # plt.findobj(ax)
        # ax.set_visible(False)
        # plt.show()  # CHECKED
        axs = [ax1, ax2]

    elif xy == 'y':
        rect = ax.get_position().bounds
        lim = ax.get_ylim()
        prop_all = ((amin - lim[0]) + (lim[1] - amax)) / (1 - margin)
        prop_min = (amin - lim[0]) / prop_all
        prop_max = (lim[1] - amax) / prop_all
        rect1 = np.array(
            [
                rect[0],
                rect[1],
                rect[2],
                rect[3] * prop_min
            ]
        )
        rect2 = [
            rect[0],
            rect[1] + rect[3] * (1 - prop_max),
            rect[2],
            rect[3] * (1 - prop_max)
        ]

        fig = ax.figure  # type: plt.Figure
        ax1 = fig.add_axes(plt.Axes(fig=fig, rect=rect1))
        ax1.update_from(ax)
        if fun_draw is not None:
            fun_draw(ax1)
        ax1.set_yticks(ax.get_yticks())
        ax1.set_ylim(lim[0], amin)
        ax1.spines['top'].set_visible(False)

        ax2 = fig.add_axes(plt.Axes(fig=fig, rect=rect2))
        ax2.update_from(ax)
        if fun_draw is not None:
            fun_draw(ax2)
        ax2.set_yticks(ax.get_yticks())
        ax2.set_ylim(amax, lim[1])
        ax2.spines['bottom'].set_visible(False)
        ax2.set_xticks([])

        ax.set_visible(False)
        # plt.show()  # CHECKED
        axs = [ax1, ax2]

    else:
        raise ValueError()

    return axs


def sameaxes(
    ax: Union[AxesArray, GridAxes],
    ax0: plt.Axes = None, xy='xy',
    lim: Sequence[Tuple[float, float]] = None
) -> Sequence[Tuple[float, float]]:
    """
    Match the chosen limits of axes in ax to ax0's (if given) or the max range.
    Also consider: ax1.get_shared_x_axes().join(ax1, ax2)
    Optionally followed by ax1.set_xticklabels([]); ax2.autoscale()
    See: https://stackoverflow.com/a/42974975/2565317
    :param ax: np.ndarray (as from subplotRCs) or list of axes.
    :param ax0: a scalar axes to match limits to. if None (default),
    match the maximum range among axes in ax.
    :param xy: 'x'|'y'|'xy'(default)
    :param lim: [[min, max]] of limits. If xy='xy', contains two pairs.
    :return: [[min, max]] of limits. If xy='xy', contains two pairs.
    """
    if type(ax) is np.ndarray or type(ax) is GridAxes:
        ax = ax.flatten()

    def cat_lims(lims):
        return np.concatenate([np.array(v1).reshape(1, 2) for v1 in lims])

    if lim is not None:
        if np2.is_iter(lim[0]):
            lims_res = lim
        else:
            lims_res = [lim, lim]
    else:
        lims_res = []
        for xy1 in xy:
            if lim is None:
                if ax0 is None:
                    if xy1 == 'x':
                        lims = cat_lims([ax1.get_xlim() for ax1 in ax])
                        lim0 = ax[0].get_xlim()
                        try:
                            is_inverted = ax[0].get_xaxis().get_inverted()
                        except AttributeError:
                            is_inverted = ax[0].xaxis_inverted()
                    else:
                        lims = cat_lims([ax1.get_ylim() for ax1 in ax])
                        try:
                            is_inverted = ax[0].get_yaxis().get_inverted()
                        except AttributeError:
                            is_inverted = ax[0].yaxis_inverted()
                    if is_inverted:
                        lims0 = [np.max(lims[:, 0]), np.min(lims[:, 1])]
                    else:
                        lims0 = [np.min(lims[:, 0]), np.max(lims[:, 1])]
                else:
                    if xy1 == 'x':
                        lims0 = ax0.get_xlim()
                    elif xy1 == 'y':
                        lims0 = ax0.get_ylim()
                    else:
                        raise ValueError()
            else:
                lims0 = lim
            lims_res.append(lims0)

    for xy1, lims0 in zip(xy, lims_res):
        if xy1 == 'x':
            for ax1 in ax:
                ax1.set_xlim(lims0)
        elif xy1 == 'y':
            for ax1 in ax:
                ax1.set_ylim(lims0)
        else:
            raise ValueError()

    return lims_res


def same_clim(
    images: Union[mpl.image.AxesImage, Iterable[plt.Axes]],
    img0: Union[mpl.image.AxesImage, plt.Axes] = None,
    clim=None, symmetric=False
) -> Sequence[float]:
    """
    Unify color axis
    :param images: images whose clim will be unified
    :param img0: image(s) whose maximum clim will be used to unify clims.
        If None, images is used.
    :param clim: If provided, img0 is ignored
    :param symmetric: If True, clim=[-max(abs(clim)), +max(abs(clim))] is used
    :return: clim
    """
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

    if len(images) == 0:
        return

    if clim is None:
        if img0 is None:
            img0 = images
            try:
                if isinstance(img0, np.ndarray):
                    img0.flatten()
            except:
                img0 = [img0]
        if isinstance(img0[0], plt.Axes):
            img0 = [img1.findobj(mpl.image.AxesImage)[0] for img1 in img0]

        # # DEBUGGED: just using array min and max ignores existing
        # #  non-None clims
        # arrays = np.concatenate([
        #     im.get_array().flatten() for im in images], 0)
        # clim = [np.amin(arrays), np.amax(arrays)]

        # # DEBUGGED: np.amax(clims) doesn't work when either clim is None.
        clims = np.array([im.get_clim() for im in img0], dtype=object)

        def fun_or_val(fun, v, im):
            if v is not None:
                return v
            else:
                a = im.get_array()
                if a.size > 0:
                    return fun(a)
                else:
                    return np.nan

        clims[:, 0] = [fun_or_val(np.nanmin, v, im)
            for v, im in zip(clims[:, 0], images)]
        clims[:, 1] = [fun_or_val(np.nanmax, v, im)
            for v, im in zip(clims[:, 1], images)]
        clims = clims.astype(float)
        clim = [np.nanmin(clims[:, 0]), np.nanmax(clims[:, 1])]

    if symmetric:
        cmax = np.amax(np.abs(clim))
        clim = [-cmax, +cmax]
    for img in images:
        img.set_clim(clim)
    return clim


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


def beautify_psychometric(
    ax=None,
    ylim=[0, 1],
    y_margin=0.05,
    axvline=False,
    axhline=False
):
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
        plt.axvline(
            x=0,
            color=[.9, .9, .9],
            linestyle='-', zorder=-1,
            linewidth=0.5
        )
    if axhline:
        plt.axhline(
            y=0.5,
            color=[.9, .9, .9],
            linestyle='-', zorder=-1,
            linewidth=0.5
        )


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


def box_prop(
    linewidth=3,
    color='r',
    spines: Union[str, Iterable[str]] = 'all',
    ax: plt.Axes = None,
    visible=True,
    zorder=None,
):
    if isinstance(spines, str) and spines == 'all':
        spines = ('left', 'right', 'top', 'bottom')
    if ax is None:
        ax = plt.gca()
    for spine in spines:
        s = ax.spines[spine]
        s.set_edgecolor(color)
        s.set_linewidth(linewidth)
        s.set_visible(visible)
        if zorder is not None:
            s.set_zorder(zorder)


def box_off(
    remove_spines: Union[str, Iterable[str]] = ('right', 'top'),
    remove_ticklabels=True,
    remove_ticks=True,
    ax=None
):
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
        # ax.set_xticks([])
        # ax.set_yticks([])

    if 'left' in remove_spines:
        if remove_ticks:
            ax.tick_params(axis='y', length=0)
        if remove_ticklabels:
            ax.set_yticklabels([])
    if 'bottom' in remove_spines:
        if remove_ticks:
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


def ticks(
    ax=None, xy='y',
    major=True,
    interval=None, format=None, length=None, **kwargs
):

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

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


def hide_ticklabels(xy='xy', ax=None, to_hide=True):
    if ax is None:
        ax = plt.gca()
    if 'x' in xy:
        plt.setp(ax.get_xticklabels(), visible=not to_hide)
    if 'y' in xy:
        plt.setp(ax.get_yticklabels(), visible=not to_hide)


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


def cmap2rgba(
    cmap: mpl.colors.Colormap, n: int = None, prop: Sequence[float] = None
) -> np.ndarray:
    """
    Following https://stackoverflow.com/a/26109298/2565317
    :param cmap:
    :param prop: [i] = between 0 and 1
    :return: rgb[i, (r,g,b)]
    """
    if n is not None:
        prop = np.linspace(0., 1., n)
    else:
        assert prop is not None

    from matplotlib import cm
    norm = mpl.colors.Normalize(vmin=0, vmax=1.)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    return scalar_map.to_rgba(prop)


def cmap_alpha(
    cmap: Union[mpl.colors.Colormap, str, Iterable[float]],
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


def cmap_gamma(
    cmap: Union[mpl.colors.Colormap, str, Iterable[float]] = None,
    n=256,
    piecelin=(0.2, 0.8),
    f: Callable[[np.ndarray, ], np.ndarray] = None,
    name='gamma',
) -> ListedColormap:
    """
    Add linear alphas to a colormap.
    Give directly as cmap=cmap_gamma() to imshow(),
    rather than setting with set_cmap().
    The latter will result in an error on savefig().

    based on https://stackoverflow.com/a/37334212/2565317
    and https://stackoverflow.com/questions/49367144/modify-matplotlib-colormap

    :param cmap: cmap with alpha of 1 / cmap.N, or a color (RGB/A or str)
    :param n:
    :return: cmap
    """
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    if piecelin is not None:
        x_control, y_control = piecelin
        f = np.vectorize(
            lambda v: v * y_control / x_control if v < x_control
            else (v - x_control) * (1 - y_control) / (
                1 - x_control) + y_control
        )
    elif f is None:
        f = lambda v: v

    assert isinstance(cmap, mpl.colors.Colormap)
    v = np.arange(n)
    v = f(v / n)
    cmap0 = cmap(v)
    cmap1 = ListedColormap(
        cmap0, name=name,
        N=cmap0.shape[0],
    )
    return cmap1


def colormap2arr(arr, cmap):
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
    gradient = cmap(np.linspace(0.0, 1.0, 100))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2 = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code, dist = scv.vq(arr2, gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values = code.astype('float') / gradient.shape[0]

    # Reshape values back to (240,240)
    values = values.reshape(arr.shape[0], arr.shape[1])
    values = values[::-1]
    return values


def ____Heatmaps____():
    pass


def kde2d(
    xy, xs=None, ys=None, nx=100, ny=100, ax=None,
    to_plot=True,
    **kwargs
):
    """

    :param xy: [(x, y), ...]
    :param xs:
    :param ys:
    :param kwargs:
    :return:
    """
    assert xy.shape[0] == 2
    kde = gaussian_kde(xy)
    if xs is None:
        xs = np.linspace(xy[0].min(), xy[0].max(), nx)
    if ys is None:
        ys = np.linspace(xy[1].min(), xy[1].max(), ny)

    x_grid, y_grid = np.meshgrid(xs, ys, indexing='ij')
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = np.reshape(kde(positions), x_grid.shape)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[1]

    if to_plot:
        if ax is None:
            ax = plt.gca()
        h = ax.imshow(
            z.T, origin='lower',
            extent=[
                xs[0] - dx / 2, xs[-1] + dx / 2,
                ys[0] - dy / 2, ys[-1] + dy / 2,
            ],
            **kwargs
        )
    else:
        h = None

    return h, z


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


def imshow_discrete(
    x, shade=None,
    colors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    color_shade=(1, 1, 1),
    **kw
):
    """
    Given index x[row, col], show color[x[row, col]]
    :param x:
    :param shade: Weight given to the foreground color.
    :param colors: colors[i]: (R,G,B), (R,G,B,A), or color name
    :param color_shade: Background color.
    :param kw: Keyword arguments for imshow
    :return:
    """
    from matplotlib.colors import to_rgba
    colors = [to_rgba(color) for color in colors]

    if shade is None:
        shade = np.ones(list(x.shape[:-1]) + [1])
    else:
        shade = shade[:, :, np.newaxis]

    n_color = len(colors)
    c = np.zeros(list(x.shape) + [len(colors[0])])
    for i_color in range(n_color):
        incl = np.float32(x == i_color)[:, :, np.newaxis]
        c += incl * shade * np.float32(np2.vec_on(colors[i_color], 2, 3)) \
             + incl * (1. - shade) * np.float32(np2.vec_on(color_shade, 2, 3))

    plt.imshow(c, **kw)


def imshow_weights(
    w, colors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    color_bkg=(1, 1, 1),
    to_plot=True,
    **kwargs
):
    """
    color[row, column] = sum_i(w[row, column, i] * (colors[i] - color_bkg))
    :param w: [row, column, i]: weight given to i-th color
    :type w: np.ndarray
    :param colors: [i] = [R, G, B], [R, G, B, A], or color name
    :param color_bkg:
    :return: h_imshow if to_plot, else color[row, column, RGBA]
    """
    assert isinstance(w, np.ndarray)
    assert w.ndim == 3

    from matplotlib.colors import to_rgba, to_rgba_array
    colors = to_rgba_array(colors)
    color_bkg = np.array(to_rgba(color_bkg))
    dcolors = np.stack(
        [
            c - color_bkg for c in colors
        ]
    )[None, None, :, :]  # [1, 1, w, color]
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

    if to_plot:
        return plt.imshow(color, **kwargs)
    else:
        return color


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
    # borderpad=-1,
    offset_inch=0.1,
    label: str = '',
    orientation: str = 'vertical',
    # kw_inset=(),
    kw_cbar=(),
    to_mark_range=False,
    range_lim=None,
    kw_mark_range=(('color', 'k'), ('linewidth', 2.)),
    levels_to_mark=(),
    kw_mark_levels=(('color', 'k'), ('linewidth', .5)),
) -> mpl.colorbar.Colorbar:
    """
    Add a colorbar aligned to the mappable (e.g., image)

    :param ax:
    :param mappable: defaults to image in the axes
    :param loc: as for legend. 'right', 'lower center', ...
    :param width: relative to the axes.
        Ends with '%' (of the axis width) or 'in' (inch)
    :param height: relative to the axes
        Ends with '%' (of the axis height) or 'in' (inch)
    :param borderpad: relative to the fontsize of the axes.
        When loc='right',
            0 aligns the right edges of the colorbar and the parent axis.
            Negative value pushes the colorbar further to the right.
    :param kw_inset:
    :param kw_cbar:
    :param to_mark_range:
    :param range_lim:
    :param kw_mark_range:
    :param levels_to_mark:
    :param kw_mark_levels:
    :return:
    """
    assert isinstance(width, str)
    assert isinstance(height, str)

    if ax is None:
        ax = plt.gca()
    if mappable is None:
        try:
            mappable = ax.findobj(mpl.image.AxesImage)[0]
        except IndexError:
            mappable = ax.findobj(mpl.collections.QuadMesh)[0]
    # if borderpad is not None:
    #     raise DeprecationWarning(
    #         'borderpad is used with inset_axes, '
    #         'which makes the figure unpicklable '
    #         '- use offset_inch instead!')

    if orientation is None:
        if loc in ['right']:
            orientation = 'vertical'
        elif loc in ['lower center']:
            orientation = 'horizontal'
        else:
            raise ValueError()
    if orientation == 'horizontal':
        width, height = height, width
    axins = add_inset(
        width=width, height=height, ax=ax, loc=loc, offset_inch=offset_inch
    )

    fig = ax.figure
    cb = fig.colorbar(
        mappable, cax=axins,
        label=label, orientation=orientation, **dict(kw_cbar)
    )

    xlim = plt.xlim()
    ylim = plt.ylim()

    if to_mark_range:
        plt.sca(cb.ax)

        if range_lim is None:
            range_lim = (
                np.nanmin(mappable.get_array()),
                np.nanmax(mappable.get_array()))

        if orientation == 'vertical':
            plt.plot(
                np.mean(cb.ax.xaxis.get_data_interval()) + np.zeros(2),
                range_lim,
                **{
                    'marker': '_',
                    **dict(kw_mark_range)
                }
            )
        else:
            plt.plot(
                range_lim,
                np.mean(cb.ax.yaxis.get_data_interval()) + np.zeros(2), **{
                    'marker': '|',
                    **dict(kw_mark_range)
                }
            )

    for level in levels_to_mark:
        if orientation == 'vertical':
            f = plt.axhline
        else:
            f = plt.axvline
        f(level, **dict(kw_mark_levels))

    if to_mark_range or len(levels_to_mark) > 0:
        plt.xlim(xlim)
        plt.ylim(ylim)

    return cb


def add_inset(
    width: str, height: str,
    ax: plt.Axes = None,
    loc='right',
    offset_inch=0.1,
) -> plt.Axes:
    """

    :param width: '100%' (relative to the axes) or '0.1in', etc.
    :param height: '100%' (relative to the axes) or '0.1in', etc.
    :param ax:
    :param loc: as for legend. 'right', 'lower center', ...
    :param offset_inch:
    :return:
    """

    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    bbox = ax.get_position()  # type: mpl.transforms.Bbox
    xmin_ax_prop, ymin_ax_prop, width_ax_prop, height_ax_prop = bbox.bounds
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    def parse_unit(s: str, ax_inch: float, fig_inch: float) -> float:
        """

        :param s: size number followed by '%' or 'in'
        :param ax_inch: axis width or height in inch
        :param fig_inch: figure width or height in inch
        :return: v_prop: size in proportion of fig_in
        """
        if s.endswith('%'):
            v_prop = float(s[:-1]) / 100 * ax_inch / fig_inch
        elif s.endswith('in'):
            v_prop = float(s[:-2]) / fig_inch
        else:
            raise ValueError()
        return v_prop

    width_ax_inch = width_ax_prop * fig_width_inch
    height_ax_inch = height_ax_prop * fig_height_inch

    width_bar_prop, height_bar_prop = [
        parse_unit(s, ax_inch, fig_inch) for s, ax_inch, fig_inch in [
            (width, width_ax_inch, fig_width_inch),
            (height, height_ax_inch, fig_height_inch)
        ]
    ]
    if loc == 'right':
        bounds = [
            xmin_ax_prop + width_ax_prop + offset_inch / fig_width_inch,
            ymin_ax_prop + (height_ax_prop - height_bar_prop) / 2,
            width_bar_prop,
            height_bar_prop
        ]
    elif loc == 'lower center':
        bounds = [
            xmin_ax_prop + (width_ax_prop - width_bar_prop) / 2,
            ymin_ax_prop - offset_inch / fig_height_inch - height_bar_prop,
            width_bar_prop,
            height_bar_prop
        ]
    else:
        raise NotImplementedError()
    axins = fig.add_axes(rect=bounds)
    return axins


def multiheatmap(
    a: np.ndarray,
    kw_imshow: dict = None,
    inches_per_cell: float = 0.2,
    wspace: Union[float, Sequence[float]] = 0.5,
    hspace: Union[float, Sequence[float]] = 0.5,
    parent: Union[plt.Axes, plt.Figure, GridAxes] = None,
    kw_gridaxes=None,
) -> (GridAxes, (GridAxes, nptyp.NDArray[GridAxes])):
    """

    :param a: multidimensional array.
        Last two dimensions are used as the (row, column) of the unit heatmap
        The preceding two dimensions are used as the (row, column)
        of an array of the unit heatmaps, and so on.
    :param parent: axes, figure, or GridAxes
    :param kw_imshow: kwargs for imshow
    :param inches_per_cell: width and height of each cell in inches
    :param wspace: [level] = width of the space between cells in inches
        on that level.
        level=0 is the space between the top level cells, and
        level=-1 is the space between the unit imshows.
    :param hspace: [level] = width of the space between cells in inches
        on that level.
        level=0 is the space between the top level cells, and
        level=-1 is the space between the unit imshows.
    :param kw_gridaxes: kwargs for GridAxes
    :return: hs, (axs, array_subaxs)
        hs[...] = handle to the unit axes
    """
    if kw_imshow is None:
        kw_imshow = {}
    if kw_gridaxes is None:
        kw_gridaxes = {}

    shape = list(a.shape)
    if len(shape) % 2 == 1:
        shape = [1] + shape
        a = np.array([a])
    if len(shape) == 2:
        shape = [1, 1] + shape
        a = np.array([[a]])

    if np2.is_iter(wspace):
        assert len(wspace) == len(shape) // 2 - 1
    else:
        wspace = np.r_[np.zeros(len(shape) // 2 - 1) + wspace]
    if np2.is_iter(hspace):
        assert len(hspace) == len(shape) // 2 - 1
    else:
        hspace = np.r_[np.zeros(len(shape) // 2 - 1) + hspace]

    def compute_cell_size(
        shape1: Sequence[int],
        hspace1: Sequence[float], wspace1: Sequence[float],
    ) -> (float, float):
        """

        :param shape1:
        :param wspace1: in inches
        :param hspace1: in inches
        :return: height_inch, width_inch
        """
        assert len(shape1) % 2 == 0
        assert len(shape1) >= 4
        if len(shape1) == 4:
            # ignore wspace1 and hspace1
            h, w = tuple(
                list(
                    np.array([inches_per_cell, inches_per_cell])
                    * np.array(shape1[-2:])
                )
            )
            # print (h, w)
            return h, w
        else:
            h, w = tuple(
                list(
                    np.array(
                        compute_cell_size(
                            shape1[2:], hspace1[1:], wspace1[1:],
                        )
                    )
                    * np.array(shape1[2:4])
                    + np.array([hspace1[0], wspace1[0]])
                    * (np.array(shape1[:2]) - 1)
                )
            )
            # print (h, w)
            return h, w

    cell_height, cell_width = compute_cell_size(
        shape, hspace[1:], wspace[1:],
    )
    axs = GridAxes(
        nrows=shape[0], ncols=shape[1],
        heights=cell_height,
        widths=cell_width,
        hspace=hspace[0],
        wspace=wspace[0],
        parent=parent,
        **(kw_gridaxes if parent is None else dict(
            left=0, right=0, top=0, bottom=0
        )),
    )

    if len(shape) == 4:
        # plot the unit heatmap
        hs = np.empty(shape[:2], dtype=object)
        for row in range(axs.nrows):
            for col in range(axs.ncols):
                plt.sca(axs[row, col])
                hs[row, col] = plt.imshow(a[row, col], **kw_imshow)
        array_subaxs = hs
    else:
        hs, array_subaxs = np.vectorize(
            multiheatmap,
            otypes=(object, object)
        )(
            a=np2.arrayobj(a, ndim_objarray=2),
            parent=axs.axs,
            kw_imshow=kw_imshow,
            inches_per_cell=inches_per_cell,
            wspace=np2.arrayobj1d([wspace[1:]]),
            hspace=np2.arrayobj1d([hspace[1:]]),
        )
    hs = np2.cell2mat(hs, dtype=object)
    return hs, (axs, array_subaxs)


def ____Errorbar____():
    pass


def patch_wave(
    y_wave0, x_lim,
    wave_margin=0.05,
    wave_amplitude=0.05,
    width_wave=0.82,
    color='w',
    axis_wave='x',
    ax: plt.Axes = None
) -> patches.Polygon:
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
    x_wave = np.concatenate(
        [
            np.array([x_lim]),
            x_lim - wave_margin - wave_amplitude
            + wave_amplitude * np.sin(np.linspace(0, 2 * np.pi, nxy_wave)),
            np.array([x_lim])
        ]
    )
    y_wave = np.concatenate(
        [
            np.array([y_wave0 - width_wave / 2]),
            y_wave0 + np.linspace(-1, 1, nxy_wave) * width_wave / 2,
            np.array([y_wave0 + width_wave / 2])
        ]
    )
    xy_wave = np.stack([x_wave, y_wave], -1)
    if axis_wave == 'y':
        xy_wave = np.flip(xy_wave, -1)

    patch = patches.Polygon(
        xy_wave,
        edgecolor='None', facecolor=color
    )

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


def bar_group(
    y: np.ndarray, yerr: np.ndarray = None,
    width=0.8, width_indiv=1.,
    cmap: Union[
        mpl.colors.Colormap,
        Iterable[Union[str, Iterable[float]]]
    ] = None,
    kw_color=('color',),
    **kwargs
) -> (List[mpl.container.BarContainer], np.ndarray):
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
            **{**kwargs, **kw}
        )
        hs.append(h)
        xs.append(x + dx)

    xs = np.stack(xs, -1)
    return hs, xs


def errorbar_shade(x, y, yerr=None, **kw):
    """

    :param x:
    :param y:
    :param yerr: as in plt.errorbar.
        If 2D,  yerr[0,:] = err_low, yerr[1,:] = err_high
    :param kw:
    :return:
    """
    if yerr is None:
        y1 = y[0, :]
        y2 = y[1, :]
    else:
        if yerr.ndim == 1:
            yerr = np.concatenate(
                [-yerr[np.newaxis, :],
                    yerr[np.newaxis, :]], axis=0
            )
        elif yerr.ndim == 2:
            pass

        else:
            raise ValueError()

        y1 = y + yerr[0, :]
        y2 = y + yerr[1, :]

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


def ecdf(
    x0, *args, w=None, flip_y=False,
    ax: plt.Axes = None,
    **kwargs
) -> List[plt.Line2D]:
    """
    See also np2.ecdf()
    :param x0:
    :param args: fed to step()
    :param w: weight
    :param kwargs: fed to step()
    :return: list of lines
    """
    p, x = np2.ecdf(x0, w=w)
    return step_ecdf(p, x, *args, flip_y=flip_y, ax=ax, **kwargs)


def step_ecdf(
    p: np.ndarray, x: np.ndarray, *args,
    p_left=0., p_right=1., flip_y=False,
    ax: plt.Axes = None,
    **kwargs
) -> List[plt.Line2D]:
    p = np.r_[p_left, p, p_right]
    if flip_y:
        p = 1 - p
    if ax is None:
        ax = plt.gca()
    return ax.step(
        np.r_[x[0], x, x[-1]],
        # np.concatenate([x[:1], x, x[-1:]], 0),
        p,
        # np.concatenate(
        #     [np.array([0.]), p, np.array([1.])
        #      ], 0),
        *args, **kwargs
    )


def significance(
    x: Union[Sequence[float], np.ndarray],
    y: Union[Sequence[float], np.ndarray],
    text='*', kw_line=(), kw_text=(),
    x_text=None,
    y_text=None,
    margin_axis='y',
    margin_line=None,
    margin_text=0.05,
    baseline=0.,
    ax: plt.Axes = None,
) -> (plt.Line2D, plt.Text):
    """
    Plot a line spanning bars to mark the significance of the comparison
    :param x: x coordinates of the points to straddle
        (usually the middle of the tip of the bars)
    :param y: y coordinates of the points to straddle
        (usually the middle of the tip of the bars)
    :param text: usually from significance_marker()
    :param kw_line: keyword arguments for the line
    :param kw_text: keyword arguments for the text
    :param x_text: x position of the text
    :param y_text: y position of the text
    :param margin_axis: The axis (or direction) of the margin.
        'y' or 'upper': the margin is on the upper side of the bars.
        'lower': the margin is on the lower side of the bars.
        'x' or 'right': the margin is on the right side of the bars.
        'left': the margin is on the left side of the bars.
    :param margin_text: margin between the line and the text relative to the
        range of the axis
    :return: h_line, h_text
    """
    if ax is None:
        ax = plt.gca()

    if margin_line is None:
        margin_line = margin_text

    if margin_axis in ['y', 'upper']:
        range_axis = np.diff(plt.ylim())
        margin_sign = 1
        v_across_bars = x
        v_along_bar = y
        ha = 'center'
        va = 'bottom'
    elif margin_axis in ['lower']:
        range_axis = np.diff(plt.ylim())
        margin_sign = -1
        v_across_bars = x
        v_along_bar = y
        ha = 'center'
        va = 'top'
    elif margin_axis in ['x', 'right']:
        range_axis = np.diff(plt.xlim())
        margin_sign = 1
        v_across_bars = y
        v_along_bar = x
        ha = 'left'
        va = 'center'
    elif margin_axis in ['left']:
        range_axis = np.diff(plt.xlim())
        margin_sign = -1
        v_across_bars = y
        v_along_bar = x
        ha = 'right'
        va = 'center'
    else:
        raise ValueError(f'Unknown margin_axis: {margin_axis}')

    def add_margin(margin):
        return margin_sign * (
            max(
                [
                    margin_sign * baseline,
                    np.amax(margin_sign * v_along_bar)
                ]
            ) + margin * range_axis)

    v_text_along_bars = add_margin(margin_text)
    v_line_along_bars = add_margin(margin_line) + np.zeros(2)
    v_text_across_bars = np.mean(v_across_bars)

    if margin_axis in ['y', 'upper', 'lower']:
        x_text = v_text_across_bars if x_text is None else x_text
        y_text = v_text_along_bars if y_text is None else y_text
        x_line = v_across_bars
        y_line = v_line_along_bars

    elif margin_axis in ['x', 'right', 'left']:
        x_text = v_text_along_bars if x_text is None else x_text
        y_text = v_text_across_bars if y_text is None else y_text
        x_line = v_line_along_bars
        y_line = v_across_bars

    else:
        raise ValueError()

    kw_line = {
        'color': 'k', 'linewidth': 0.5, 'linestyle': '-',
        **dict(kw_line)
    }
    kw_text = {'ha': ha, 'va': va, **dict(kw_text)}
    h_line = ax.plot(x_line, y_line, **kw_line)
    h_text = ax.text(x_text, y_text, text, **kw_text)
    return h_line, h_text


def significance_marker(
    p: Union[float, np.ndarray],
    thres=(0.1, 0.05, 0.01, 0.001),
    markers=('n.s.', '+', '*', '**', '***')
) -> Union[str, np.ndarray]:
    """

    :param p:
    :param thres:
    :param markers:
    :return:
    """
    if p is None:
        return ''

    markers = np.array(markers)
    p = np.array(p)
    lessthan = np.stack([p < thres1 for thres1 in thres]).sum(0).astype(int)
    return markers[lessthan]


def ____Gaussian____():
    pass


def plot_centroid(
    mu=np.zeros(2), cov=np.eye(2),
    add_axis=True, *args, **kwargs
):
    th = np.linspace(0, 2 * np.pi, 100)[np.newaxis, :]
    u, s, _ = np.linalg.svd(cov)
    x = np.concatenate((np.cos(th), np.sin(th)), axis=0)
    us = u @ np.diag(np.sqrt(s))
    x = us @ x + mu[:, np.newaxis]
    h = plt.plot(x[0, :], x[1, :], *args, **kwargs)
    res = {
        'u': u,
        's': s,
        'us': us,
        'x': x,
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
    st = siz1 * np.array([c - 1, r - 1])
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


def arrays2gif(
    arrays: Sequence[np.ndarray],
    file='ani.gif', duration=100, loop=0,
    **kwargs
):
    """
    Convert numpy arrays, as from fig2array(), into gif.
    :param arrays: arrays[i]: an [height,width,color]-array, e.g.,
    from fig2array()
    :type arrays: Union[Iterable, np.ndarray]
    :param duration: of each frame, in ms
    :param loop: 0 to loop forever, None not to loop
    :rtype: Iterable[PIL.Image]
    """
    from PIL import Image

    height, width, n_channel = arrays[0].shape
    images = []
    for arr in arrays:
        images.append(Image.fromarray(arr))

    kwargs.update(
        {
            'duration': duration,
            'loop': loop
        }
    )
    if kwargs['loop'] is None:
        kwargs.pop('loop')

    images[0].save(
        file,
        save_all=True,
        append_images=images[1:],
        **kwargs
    )
    return images


def convert_movie(src_file: str, ext_new='.mp4', fps=10) -> str:
    """
    Adapted from: https://stackoverflow.com/a/40726572/2565317
    :param src_file: path to the source file, including extension.
        Currently only '.gif' is supported.
    :param ext_new: new extension. Currently only '.mp4' is supported
    :return: path to the new file
    """

    assert ext_new == '.mp4'
    assert fps >= 9, 'fps < 9 doesn''t work with QuickTime'

    import os
    pth_wo_ext, ext_old = os.path.splitext(src_file)
    assert ext_old == '.gif'

    dst_file = pth_wo_ext + ext_new
    from send2trash import send2trash
    if os.path.exists(dst_file):
        send2trash(dst_file)

    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    import imageio

    # def gif_to_mp4(gif_file, output_file):
    # Read the GIF file using imageio
    gif = imageio.mimread(src_file)

    # # Write the frames to a temporary directory as PNG files
    # with imageio.imiter(dst_file, format='png') as writer:
    #     for frame in gif:
    #         writer.append_data(frame)

    # Create a moviepy clip from the PNG files and write it to an MP4 file
    clip = ImageSequenceClip(gif, fps=fps)
    clip.write_videofile(dst_file, codec='mpeg4')

    # import moviepy.editor as mp
    # clip = mp.VideoFileClip(src_file)
    #
    # mkdir4file(dst_file)
    # clip.write_videofile(dst_file)  # , write_logfile=True)

    # import ffmpy
    # ff = ffmpy.FFmpeg(
    #     inputs={src_file: None},
    #     outputs={dst_file: None}
    # )
    # ff.run()

    return dst_file


class Animator:
    def __init__(self):
        self.frames = []

    def __len__(self) -> int:
        return len(self.frames)

    def append_fig(self, fig: plt.Figure):
        self.frames.append(fig2array(fig))

    def export(
        self, file, ext=('.gif', '.mp4'), duration=250, loop=0,
        kw_gif=()
    ) -> Iterable[str]:
        """

        :param file:
        :param ext: extension of the output file, e.g., '.gif', '.mp4'
        :param duration:
        :param loop:
        :param kw_gif:
        :return:
        """
        if type(ext) is str:
            ext = [ext]

        import os
        file_wo_ext = os.path.splitext(file)[0]
        file_gif = file_wo_ext + '.gif'
        mkdir4file(file_gif)

        files = [file_gif]

        arrays2gif(
            self.frames, file_gif,
            duration=duration, loop=loop,
            **dict(kw_gif)
        )

        for ext1 in ext:
            if ext1 != '.gif':
                convert_movie(file_gif, ext1)
                files.append(file_wo_ext + ext1)

        if '.gif' not in ext:
            os.remove(file_gif)
            files = files[1:]

        return files


def ____Latex____():
    pass


def use_sans_serif():
    """
    Use sans-serif fonts in matplotlib.
    from https://stackoverflow.com/a/20709149/2565317
    :return:
    """
    mpl.rcParams['text.latex.preamble'] = [
        r'\usepackage{siunitx}',
        # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',
        # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ]


def ____COMPOSITE_FIGURES____():
    pass


class SimpleFilename:
    """
    copy files to simple names as they are appended; clean them up on del.
    """

    def __init__(self, file_out: str, to_save_csv=True):
        self.file_out = file_out
        self.file_in = []
        self.file_temp_rel = []
        self.file_temp_abs = []
        self.temp_dir_rel = '_temp_SimpleFilename'
        self.temp_dir_abs = os.path.join(
            os.path.dirname(self.file_out),
            self.temp_dir_rel
        )
        self.to_save_csv = to_save_csv
        self._closed = False

    def append(self, file_in) -> str:
        self.file_in.append(file_in)

        import shutil, hashlib

        mkdir4file(os.path.join(self.temp_dir_abs, 'temp'))

        file_name1 = (
            hashlib.md5(file_in.encode('utf-8')).hexdigest()
            + os.path.splitext(file_in)[1])
        file_temp_rel = os.path.join(self.temp_dir_rel, file_name1)
        file_temp_abs = os.path.join(self.temp_dir_abs, file_name1)
        shutil.copy(file_in, file_temp_abs)
        self.file_temp_rel.append(file_temp_rel)
        self.file_temp_abs.append(file_temp_abs)

        return file_temp_rel

    def close(self):
        if not self._closed:
            from send2trash import send2trash
            for file_abs in self.file_temp_abs:
                if os.path.exists(file_abs):
                    send2trash(file_abs)
            if os.path.exists(self.temp_dir_abs):
                send2trash(self.temp_dir_abs)

            if self.to_save_csv:
                import pandas as pd
                df = pd.DataFrame(
                    data={
                        'file_in': self.file_in,
                        'file_temp_rel': self.file_temp_rel
                    }
                )
                file_csv = os.path.splitext(self.file_out)[0] + '.csv'
                df.to_csv(file_csv)
                print('Saved original paths to %s' % file_csv)
            self._closed = True

    def __del__(self):
        self.close()

    def __exit__(self, *args, **kwargs):
        self.close()


class SimpleFilenameArray:
    def __init__(
        self, file_out, files,
        to_save_csv=True,
        to_skip_absent=False,
    ):
        """
        rename files to simple names
        :param file_out:
        :param files:
        """
        self.to_save_csv = to_save_csv
        self.to_skip_absent = to_skip_absent

        if not isinstance(files, np.ndarray):
            files = np.array([files])
        if files.ndim == 1:
            files = files[None]
        elif files.ndim == 3:
            files = files[0]
        assert files.ndim == 2
        self.files = files
        self.file_out = file_out
        self.files_abs = None
        self.temp_dir_abs = None

    def __enter__(self):
        """

        :return: files_rel
        """
        files = self.files
        file_out = self.file_out

        files_all = file_out + '\n'.join(
            [
                ('' if v is None else v)
                for v in files.flatten()
            ]
        )

        import shutil, hashlib
        temp_name = hashlib.md5(files_all.encode('utf-8')).hexdigest()
        temp_dir_rel = '_temp_subfig' + temp_name
        temp_dir_abs = os.path.join(os.path.dirname(file_out), temp_dir_rel)
        mkdir4file(os.path.join(temp_dir_abs, 'temp'))
        files_rel = np.empty_like(files)
        files_abs = np.empty_like(files)
        for row in range(files.shape[0]):
            for col in range(files.shape[1]):
                file0 = files[row, col]
                if file0 is None or file0 == '':
                    files_rel[row, col] = None
                    files_abs[row, col] = None
                    continue
                # NOTE: consider using os.path.relpath()
                file_name1 = 'row%dcol%d%s' % (
                    row, col, os.path.splitext(file0)[1])
                files_rel[row, col] = os.path.join(temp_dir_rel, file_name1)
                files_abs[row, col] = os.path.join(temp_dir_abs, file_name1)
                try:
                    shutil.copy(file0, files_abs[row, col])
                except FileNotFoundError:
                    if not self.to_skip_absent:
                        raise

        if self.to_save_csv:
            import pandas as pd
            df = pd.DataFrame(files)
            file_csv = os.path.splitext(file_out)[0] + '.csv'
            df.to_csv(file_csv)
            print('Saved original paths to %s' % file_csv)

        self.files_abs = files_abs
        self.files_rel = files_rel
        self.temp_dir_abs = temp_dir_abs
        return self

    def __exit__(self, exc_type=None, exc_value=None, exc_traceback=None):
        from send2trash import send2trash

        files = self.files
        files_abs = self.files_abs
        temp_dir_abs = self.temp_dir_abs

        for file0, file1 in zip(files.flatten(), files_abs.flatten()):
            if (
                file1 is not None and file0 != file1 and
                os.path.exists(file1)
            ):
                send2trash(file1)
        if os.path.exists(temp_dir_abs):
            os.rmdir(temp_dir_abs)

        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            raise RuntimeError('An exception occurred!')


def get_pdf_size(file) -> (float, float):
    """
    :param file:
    :return: width_point, height_point
    """
    from PyPDF2 import PdfFileReader
    input1 = PdfFileReader(open(file, 'rb'))
    rect = input1.getPage(0).mediaBox
    width_document = rect[2]
    height_document = rect[3]
    return width_document, height_document


def get_image_size(file) -> (int, int):
    """
    :param file:
    :return: width_pixel, height_pixel
    """
    from PIL import Image
    return Image.open(file).size


class LatexDoc(ltx.Document, np2.ContextManager):
    def __init__(
        self,
        *args,
        file_out=None,
        title='',
        to_save_csv=True,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.file_out = file_out
        self.simple_filename = SimpleFilename(
            self.file_out,
            to_save_csv=to_save_csv
        )

        self.n_row = 0

        doc = self
        doc.packages.append(ltx.Package('subcaption'))
        doc.packages.append(
            ltx.Package(
                'caption', [
                    'labelformat=parens',
                    'labelsep=quad',
                    'justification=centering',
                    'font=scriptsize',
                    'labelfont=sf',
                    'textfont=sf',
                ]
            )
        )
        doc.packages.append(ltx.Package('graphicx'))
        doc.append(
            ltx.Command(
                'setlength', [
                    ltx.Command('abovecaptionskip'), '0pt']
            )
        )
        doc.append(
            ltx.Command(
                'setlength', [
                    ltx.Command('belowcaptionskip'), '0pt']
            )
        )

        if len(title) > 0:
            doc.preamble.append(ltx.Command('title', title))
            doc.append(ltx.Command(r'\maketitle'))

    def simplify_path(self, fullpath: str) -> str:
        return self.simple_filename.append(fullpath)

    def close(self, **kwargs):
        file_out = self.file_out
        mkdir4file(file_out)
        if file_out.lower().endswith('.pdf'):
            file_out = file_out[:-4]
        self.generate_pdf(
            file_out,
            **{
                **dict(
                    clean=True,
                    clean_tex=True,
                ),
                **kwargs
            }
        )

        # should close simple_filename after generate_pdf()
        # so that temporary files can be used before being deleted.
        self.simple_filename.close()

    def __exit__(self, *args, **kwargs):
        self.close()
        super().__exit__(*args)


class Frame(ltx.base_classes.Environment):
    """
    Usage:
    from pylatex import Command

    doc = plt2.LatexDoc(
        file_out='demo_beamer.pdf',
        documentclass=['beamer'],
    )

    doc.append(Command('title', 'Sample title'))
    doc.append(Command('frame', Command('titlepage')))

    for i in range(3):
        with doc.create(plt2.Frame()):
            doc.append(f'page {i}')

    doc.close()

    See also: https://www.overleaf.com/learn/latex/Beamer
    """
    pass


class Adjustbox(ltx.base_classes.Environment):
    def __init__(
        self, *args,
        arguments=ltx.NoEscape(
            r'max width=\textwidth, '
            r'max totalheight=\textheight-2\baselineskip,'
            r'keepaspectratio'
        ), **kwargs
    ):
        super().__init__(*args, arguments=arguments, **kwargs)


def latex_table(
    doc: LatexDoc, dicts: Sequence[Dict[str, str]]
):
    """
    Use in, e.g., "with doc.create(ltx.Table()) as table:" block
    :param doc:
    :param dicts: [row: int][column: str] = element
    :return: None
    """
    # with doc.create(ltx.Center()):
    with doc.create(
        ltx.Tabular(
            '|' +
            '|'.join(['c'] * len(dicts[0]))
            + '|'
        )
    ) as tabular:
        tabular.add_hline()
        tabular.add_row(dicts[0].keys())
        for row in dicts:
            tabular.add_hline()
            tabular.add_row(list(row.values()))
        tabular.add_hline()


class Figure(ltx.Figure):
    # TODO: add new SubFigure that accepts height:
    #  see subfig_rows() below and
    #  https://tex.stackexchange.com/questions/47245/set-a-maximum-width-and-height-for-an-image
    def add_image(
        self, filename, *, width=ltx.NoEscape(r'0.8\textwidth'),
        placement=ltx.NoEscape(r'\centering'), height=None
    ):
        """Add an image to the figure.

        Args
        ----
        filename: str
            Filename of the image.
        width: str
            The width of the image
        placement: str
            Placement of the figure, `None` is also accepted.

        """

        if width is None:
            width = ''
        else:
            if self.escape:
                width = ltx.escape_latex(width)

            width = 'width=' + str(width)

        if height is None:
            height = ''
        else:
            if self.escape:
                height = ltx.escape_latex(height)

            height = 'height=' + str(height)

        if placement is not None:
            self.append(placement)

        self.append(
            ltx.StandAloneGraphic(
                image_options=(
                    None if (width == '' and height == '') else
                    ', '.join([width, height, 'keepaspectratio'])
                ),
                filename=ltx.utils.fix_filename(filename)
            )
        )


class LatexDocStandalone(LatexDoc):
    def __init__(self, *args, width_document_cm=21, **kwargs):
        super().__init__(
            *args, **{
                **dict(
                    documentclass=('standalone',),
                    document_options=[
                        'varwidth=%fcm' % width_document_cm,
                        'border=0pt'
                    ],
                ),
                **kwargs
            }
        )
        self.width_document_cm = width_document_cm

    def append_subfig_row(
        self, files_rel,
        caption=None,
        subcaptions=None,
        width_column: Union[float, Sequence[float]] = None,
        # height_row: float = None,  # TODO: use with the new Figure above
        hspace_cm=0.,
        caption_on_top=False,
        subcaption_on_top=False,
        ncol=None,
        to_skip_absent=False,
        to_add_newpage=False,
    ):
        """

        :param files_rel:
        :param caption:
        :param subcaptions:
        :param width_column:
        :param hspace_cm:
        :param caption_on_top:
        :param subcaption_on_top:
        :return:
        """
        files_rel = np.array(files_rel)

        if ncol is None:
            if files_rel.ndim == 1:
                ncol = int(np.floor(np.sqrt(files_rel.size)))
            else:
                assert files_rel.ndim == 2
                ncol = files_rel.shape[1]
        files_rel = reshape_ragged(files_rel, ncol)

        if subcaptions is not None:
            subcaptions = np.array(subcaptions)
            if subcaptions.ndim == 1:
                reshape_ragged(subcaptions, files_rel.shape[1])
            else:
                assert files_rel.shape == subcaptions.shape

        if width_column is None:
            width_column = (
                               self.width_document_cm - hspace_cm * (ncol - 1)
                           ) / ncol
        if np.isscalar(width_column):
            width_column = [width_column]
        if len(width_column) > ncol:
            width_column = width_column[:ncol]
        elif len(width_column) < ncol:
            width_column = (
                list(width_column)
                * int(np.ceil(ncol / len(width_column)))
            )[:ncol]
        width_column = np.array(width_column)
        assert width_column.ndim == 1

        if subcaption_on_top is None:
            subcaption_on_top = caption_on_top

        doc = self
        with doc.create(ltx.Figure()) as fig:
            doc.append(ltx.Command('centering'))
            if caption_on_top and caption is not None:
                fig.add_caption(caption)
            for row in range(files_rel.shape[0]):
                for col in range(files_rel.shape[1]):
                    file = files_rel[row, col]
                    if file is None or file == '':
                        continue
                    with doc.create(
                        ltx.SubFigure('%f cm' % width_column[col])
                        # TODO: add and use a new SubFigure
                        #  subclassing the new Figure class above
                        #  that accepts height
                    ) as subfig:  # type: ltx.SubFigure
                        doc.append(ltx.Command('centering'))
                        if subcaption_on_top and subcaptions is not None:
                            subfig.add_caption(subcaptions[row, col])
                        if not to_skip_absent or os.path.exists(
                            os.path.join(
                                os.path.dirname(self.file_out),
                                file
                            )
                        ):
                            subfig.add_image(
                                file, width='%f cm' % width_column[col],
                                # **{
                                #     ('height'):
                                # }
                            )
                        if (not subcaption_on_top) and subcaptions is not None:
                            subfig.add_caption(subcaptions[row, col])
                        doc.append(ltx.VerticalSpace('%f cm' % hspace_cm))
                any_added_in_row = np.any(
                    [
                        (f is not None and f != '') for f in files_rel[row, :]
                    ]
                )
                if not to_add_newpage and any_added_in_row:
                    doc.append(ltx.NewLine())
            if (not caption_on_top) and caption is not None:
                fig.add_caption(caption)
        if to_add_newpage and np.any(
            np.vectorize(
                lambda f: f is not None and f != ''
            )(files_rel)
        ):
            doc.append(ltx.NewPage())


def convert_unit(src, src_unit, dst_unit):
    if src_unit == 'pt':
        inch = src / 72.
    elif src_unit == 'in':
        inch = src
    elif src_unit == 'cm':
        inch = src * 2.54
    else:
        raise NotImplementedError()
    if dst_unit == 'cm':
        dst = inch / 2.54
    elif dst_unit == 'in':
        dst = inch
    elif dst_unit == 'pt':
        dst = inch * 72
    else:
        raise NotImplementedError()
    return dst


def subfigs(
    files: Union[Sequence, np.ndarray],
    file_out: str,
    width_document=None,
    width_column=(2,),
    hspace_cm=0.,
    ncol: int = None,
    caption=None,
    subcaptions: Union[Sequence, np.ndarray] = None,
    caption_on_top=False,
    subcaption_on_top=None,
    suptitle='',
    to_save_csv=True,
    to_skip_absent=False,
):
    """

    :param files: 1D or 2D array of strings.
        if 2D, [row, col] = file path relative to file_out's folder,
        or absolute path as obtained from os.path.abspath()
        if '', skipped
    :param file_out:
    :param width_document: in cm
    :param width_column: in cm if numerical, otherwise str
    :param hspace: space between rows
    :param ncol: defaults to files.shape[1] if it is an array;
        makes the array close to square otherwise
    :param clean_tex: delete intermediate files
    :param caption: caption for the whole array of subfigures
    :param subcaptions: caption under each subfig
    :param caption_on_top:
    :param subcaption_on_top: defaults to caption_on_top
    :return:
    """
    if width_document is None:
        files = np.array(files)
        width_document = np.sum(
            np.array(width_column) + np.zeros(files.shape[1])
        )

    with SimpleFilenameArray(
        file_out, files,
        to_save_csv=to_save_csv,
        to_skip_absent=to_skip_absent,
    ) as simplenames:
        with LatexDocStandalone(
            title=suptitle,
            width_document_cm=width_document,
            file_out=file_out,
            to_save_csv=False,  # duplicate with SimpleFilenameArray
        ) as doc:
            doc.append_subfig_row(
                files_rel=simplenames.files_rel,
                caption=caption,
                subcaptions=subcaptions,
                width_column=width_column,
                hspace_cm=hspace_cm,
                caption_on_top=caption_on_top,
                subcaption_on_top=subcaption_on_top,
                ncol=ncol,
                to_skip_absent=to_skip_absent,
            )


pdfs2subfigs = subfigs  # alias for backward compatibility


def reshape_ragged(v, ncol=None):
    if ncol is None:
        ncol = int(np.ceil(np.sqrt(v)))
    return np.r_[
        v.flatten(),
        [None] * (int(np.ceil(v.size / ncol)) * ncol - v.size)
    ].reshape([-1, ncol])


def subfigs_from_template(
    file_out: str, template: str,
    srcs: Iterable[str],
    dstss: Iterable[Union[
        Mapping[Tuple[int, int], str],
        np.ndarray
    ]],
    caption='',
    caption_on_top=True,
    subcaptions: Union[None, str, np.ndarray] = 'auto',
    **kwargs
) -> None:
    """

    :param file_out: output file
    :param template: path relative to the output file
    :param srcs: [pair] = src_pair
    :param dstss: [pair][row, col] = dst_pair
    :param caption: title on top
    :param caption_on_top:
    :param subcaptions: defaults to combination of destination strings. In the
        example, these are 'row0; col0', etc.
        Give subcaptions[row, col] = 'subcaption_row_col' to set manually.
        Give None to omit.

    EXAMPLE:
    subfigs_from_template(
        'out.pdf',
        'input_row0_col1.png',
        ['row0', 'col1'],
        [
            np.array([
                ['row0'],
                ['row1']
            ]),
            np.array([
                ['col0', 'col1']
            ])
        ]
    )
    """
    files = np.vectorize(
        lambda *dsts:
        np2.replace(
            template, [
                (src, dst) for src, dst in zip(srcs, dsts)
            ]
        )
    )(*dstss)

    if isinstance(subcaptions, str) and subcaptions == 'auto':
        subcaptions = np.vectorize(
            lambda *args: '; '.join(args)
        )(*dstss)

    subfigs(
        files, file_out,
        caption=caption,
        caption_on_top=caption_on_top,
        subcaptions=subcaptions,
        **kwargs
    )


def subfig_rows(
    file_fig: str, rows_out: Iterable[dict],
    to_save_csv=True,
    page_per_row=False,
    kw_row=None,
):
    """

    :param file_fig: combined figure name
    :param rows_out: [row][('caption', 'files')]
        rows_out[row]['files'][column] = subfigure file name
    :param to_save_csv:
    :param page_per_row: if True, add a new page per row;
        otherwise add a new line
    :return: None
    """
    assert all(['files' in row.keys() for row in rows_out])
    assert all(['caption' in row.keys() for row in rows_out])

    if kw_row is None:
        kw_row = {}

    import contextlib
    file_fig_name, file_fig_ext = os.path.splitext(file_fig)
    with contextlib.ExitStack() as stack:
        # noinspection PyTypeChecker
        simplefiles = [
            stack.enter_context(
                SimpleFilenameArray(
                    file_fig_name
                    + '+row=' + row['caption'] + file_fig_ext,
                    row.pop('files'),
                    to_save_csv=to_save_csv
                )
            )
            for row in rows_out
        ]
        with LatexDocStandalone(
            file_out=file_fig,
            to_save_csv=to_save_csv,
            documentclass=('article',) if page_per_row else ('standalone',),
        ) as doc:
            for row, simplefile in zip(rows_out, simplefiles):
                doc.append_subfig_row(
                    simplefile.files_rel,
                    to_add_newpage=page_per_row,
                    **row,
                    **kw_row,
                )


def plot_collage(
    fnames: nptyp.NDArray[str],
    crop: (slice, slice) = (slice(None), slice(None)),
    dpi: float = 300,
    kw_gridaxes: dict = (),
    ignore_missing_file=False,
    rowtitles: Sequence[str] = None,
    coltitles: Sequence[str] = None,
    suptitle: str = None,
) -> GridAxes:
    """

    :param fnames: [row, col] = str
    :param crop: (slice(y_top, y_bottom), slice(x_left, x_right))
    :param dpi:
    :return:
    """
    nrow, ncol = fnames.shape
    kw_gridaxes = {
        'hspace': 0.1,
        'wspace': 0.1,
        'left': 1,
        'top': 1,
        **dict(kw_gridaxes)
    }

    if crop[1].start is not None:
        kw_gridaxes = {
            **kw_gridaxes,
            'widths': (crop[1].stop - crop[1].start) / dpi,
        }
    if crop[0].start is not None:
        kw_gridaxes = {
            **kw_gridaxes,
            'heights': (crop[0].stop - crop[0].start) / dpi,
        }

    axs = GridAxes(nrow, ncol, **kw_gridaxes)
    for row in range(nrow):
        for col in range(ncol):
            ax = axs[row, col]
            # noinspection PyTypeChecker
            fname = fnames[row, col]  # type: str
            try:
                if fname == '':
                    raise FileNotFoundError
                else:
                    im = np.array(Image.open(fname))
            except (
                FileNotFoundError, IsADirectoryError, PIL.UnidentifiedImageError
            ):
                if ignore_missing_file:
                    plt.sca(ax)
                    box_off('all')
                    continue
                else:
                    raise
            im = im[crop[0], crop[1]]

            plt.sca(ax)
            plt.imshow(im)
            box_off('all')

    if rowtitles is not None:
        axs.rowtitle(rowtitles)
    if coltitles is not None:
        axs.coltitle(coltitles)
    if suptitle is not None:
        axs.suptitle(suptitle)
    return axs


def pdf_to_png(
    pdf_path, output_folder=None,
    skip_page_number=True,
):
    # Convert PDF to images
    images = convert_from_path(pdf_path)

    if output_folder is None:
        output_folder = os.path.dirname(pdf_path)

    output_fname = os.path.splitext(os.path.basename(pdf_path))[0]

    if skip_page_number:
        assert len(images) == 1, (
            'Skipping page number, but there are multiple pages')
        image = images[0]

        output_path = os.path.join(output_folder, f'{output_fname}.png')
        image.save(output_path, 'PNG')
        print(f"Saved {output_path}")
    else:
        # Save each page as a PNG
        for i, image in enumerate(images):
            output_path = f"{output_folder}/{output_fname}_page_{i + 1}.png"
            image.save(output_path, 'PNG')
            print(f"Saved {output_path}")


def ____MODEL_COMPARISON_PLOTS____():
    pass


def plot_bipartite_recovery(mean_losses, model_labels=None, ax=None):
    """

    :param mean_losses: [subj, model_sim, model_fit]
    :param model_labels: [model]
    :return: axs
    """
    n_model = mean_losses.shape[1]
    if model_labels is None:
        model_labels = [('model %d' % i) for i in range(n_model)]

    best_model_recovered = np.argmin(mean_losses, -1)
    adj = np.zeros([n_model, n_model])
    for src in range(n_model):
        for dst in range(n_model):
            adj[src, dst] = np.sum(best_model_recovered[:, src] == dst)

    # === Recovery confusion plot
    if ax is None:
        axs = GridAxes(1, 1, widths=2, heights=2, left=2)
        ax = axs[0, 0]

    plt.sca(ax)
    G, pos = pltn.draw_bipartite(adj)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=True,
    )
    yticks = np.linspace(-1, 1, n_model)
    node_ys = [pos[k][1] for k in range(n_model)]
    labels = [model_labels[node_ys.index(y)] for y in yticks]
    plt.yticks([-1, 1], labels)
    return ax


def imshow_costs_by_subj_model(
    costs_by_subj_model: np.ndarray,
    model_names: Sequence[str] = None,
    subjs: Union[Sequence[int], Sequence[str]] = None,
    label_colorbar: str = None,
    thres_colorbar: float = None,
    axs: GridAxes = None,
    size_per_cell: float = 0.35,
    subtract_min_in_row=True,
    offset=(0., 0.),
    to_add_colorbar=True,
    mask_below_thres=True,
    # offset=(0.025, -0.004),
) -> (GridAxes, mpl.colorbar.Colorbar):
    """

    :param costs_by_subj_model: [subj, model] = cost
    :param model_names:
    :param subjs:
    :param label_colorbar:
    :param thres_colorbar:
    :param size_per_cell: in inches
    :param subtract_min_in_row:
    :return: (axs, colorbar)
    """
    if subtract_min_in_row:
        costs_by_subj_model = (
            costs_by_subj_model
            - np.nanmin(costs_by_subj_model, -1, keepdims=True))
    n_subj1, n_model1 = costs_by_subj_model.shape
    if axs is None:
        axs = GridAxes(
            1, 1,
            widths=[size_per_cell * n_model1],
            heights=[size_per_cell * n_subj1],
            right=1.5, top=1.5, left=0.75,
            bottom=0.1,
        )
    ax = axs[0, 0]
    plt.sca(ax)
    cost_plot = costs_by_subj_model.copy()
    if mask_below_thres and thres_colorbar is not None:
        cost_plot[cost_plot < thres_colorbar] = np.nan
    im = plt.imshow(
        cost_plot, zorder=0, vmin=0.
    )
    if subjs is not None:
        plt.yticks(np.arange(n_subj1), subjs)
    if model_names is not None:
        xticklabel_top(model_names, ax)
    for row, loss_subj in enumerate(costs_by_subj_model):
        best_model = np.nanargmin(loss_subj)
        plt.text(
            best_model + offset[0], row + offset[1],
            '*',
            color='k' if mask_below_thres else 'w',
            zorder=2, fontsize=16,
            ha='center', va='center'
        )
    if to_add_colorbar:
        cb = colorbar(
            ax, im, height=f'{int(3 / n_subj1 * 100)}%',
        )
        if label_colorbar is not None:
            cb.set_label(label_colorbar)
        if thres_colorbar is not None:
            import matplotlib.patches as patches
            ax_cbar = cb.ax  # type: plt.Axes
            x_lim = ax_cbar.get_xlim()
            ax_cbar.add_patch(
                patches.Rectangle(
                    (x_lim[0], 0), x_lim[1] - x_lim[0],
                    thres_colorbar, ls='None', fc='w',
                    zorder=2
                )
            )
            # cb.ax.axhline(thres_colorbar, color='w', lw=0.5)
    else:
        cb = None
    plt.sca(ax)
    return axs, cb


def xticklabel_top(xtick_labels: Sequence[str], ax: plt.Axes = None):
    """

    :param ax:
    :param xtick_labels:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    ax.xaxis.tick_top()
    _, ticklabels = plt.xticks(np.arange(len(xtick_labels)), xtick_labels)
    for ticklabel in ticklabels:
        ticklabel.set_rotation(30)
        ticklabel.set_ha('left')


def imshow_confusion(
    best_model_sim: np.ndarray = None,
    model_labels: Sequence[str] = None,
    axs: GridAxes = None,
    kind: str = 'p_best_given_true',
    n_best_fit_by_sim: np.ndarray = None,
) -> (GridAxes, np.ndarray):
    """
    Plot a confusion matrix. Optionally pool across models by group
    :param best_model_sim: [batch, model_sim] = i_model_best
    :param model_labels: [i_model_group]
    :param group: [i_model] = i_model_group
    :return: axs, p_plot_fit_sim[group_fit, group_sim] =
        P(best_fit_model_group | model_group_sim)
    """
    assert kind in ['p_best_given_true', 'p_true_given_best']

    if n_best_fit_by_sim is None:
        assert best_model_sim is not None
        assert best_model_sim.ndim == 2

        # n_best_fit_by_sim[fit, sim] = n_subj
        n_best_fit_by_sim = count_best_model(best_model_sim)

    n_model = n_best_fit_by_sim.shape[1]
    if model_labels is None:
        model_labels = [''] * n_model

    if axs is None:
        axs = GridAxes(
            1, 1, widths=0.2 * n_model, heights=0.2 * n_model,
            top=1.5, left=2, right=1.5
        )

    if kind == 'p_best_given_true':
        p_plot_fit_sim = np2.sumto1(n_best_fit_by_sim, 0)
    elif kind == 'p_true_given_best':
        p_plot_fit_sim = np2.sumto1(n_best_fit_by_sim, 1)
    else:
        raise ValueError()

    plt.imshow(p_plot_fit_sim, origin='upper', vmin=0, vmax=1)

    for i_model_sim in range(n_model):
        for i_model_fit in range(n_model):
            plt.text(
                i_model_fit, i_model_sim,
                f'{p_plot_fit_sim[i_model_sim, i_model_fit]:.2f}'
                .lstrip('0'),
                color='w', va='center', ha='center', fontsize=5
            )

    xticklabel_top(model_labels)
    plt.yticks(np.arange(n_model), model_labels)
    plt.xlabel('simulated with')
    plt.ylabel('best fit with')
    colorbar(axs[0, 0])

    if kind == 'p_best_given_true':
        plt.ylabel(
            r'$\mathrm{P}(\mathrm{best\ fit} \mid \mathrm{sim})$'
        )
    elif kind == 'p_true_given_best':
        plt.ylabel(
            r'$\mathrm{P}(\mathrm{sim} \mid \mathrm{best\ fit})$'
        )
    else:
        raise ValueError()

    return axs, p_plot_fit_sim


def count_best_model(best_model_sim):
    """

    :param best_model_sim: [subj, i_model_sim] = i_model_fit
    :return: n_best_fit_by_sim[i_model_fit, i_model_sim]
    """
    n_model = best_model_sim.shape[-1]
    n_best_fit_by_sim = npg.aggregate(
        np.reshape(
            np.broadcast_arrays(
                np2.permute2en(best_model_sim)[:, None],
                # [sim, fit, subj]
                np.arange(n_model)[:, None, None]  # [sim, fit, subj]
            ), [2, -1]
        ), 1, 'sum', size=[n_model, n_model]
    )
    return n_best_fit_by_sim


def consolidate_count_matrix(
    mat: np.ndarray, group: Sequence[int]
) -> np.ndarray:
    """

    :param mat: [row, column] = count
    :param group: [row_or_column] = i_group
    :return: mat1[group_row, group_column]
        = sum(mat[rows in group_row, :][columns in group_col, :])
    """
    assert mat.ndim == 2
    assert len(group) == mat.shape[0]
    assert len(group) == mat.shape[1]

    group = np.array(group)
    i_groups = np.unique(group)
    n_group = np.amax(i_groups) + 1

    mat1 = np.zeros([n_group, n_group])
    for group_row in i_groups:
        for group_col in i_groups:
            rows = group == group_row
            columns = group == group_col
            mat1[group_row, group_col] = np.sum(mat[rows, :][:, columns])

    assert np.sum(mat1) == np.sum(mat)
    return mat1

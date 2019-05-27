#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:42:06 2018

@author: yulkang
"""

import matplotlib.pyplot as plt
import numpy as np

import numpy_groupies as npg

from . import np2

#%% Subplots
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

#%% Axes & limits
def sameaxes(ax, ax0=None, xy='xy'):
    """
    Match the chosen limits of axes in ax to ax0's (if given) or the max range.
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

def beautify_psychometric(ax=None, 
                          ylim=[0, 1],
                          axvline=False,
                          axhline=False):
    if ax is None:
        ax = plt.gca()
        
    dylim = ylim[1] - ylim[0]
    ylim_actual = [ylim[0] - dylim / 20., ylim[1] + dylim / 20.]
    plt.ylim(ylim_actual)
    detach_yaxis(ymin=ylim[0], ymax=ylim[1])
    if ylim[0] == 0.5 and ylim[1] == 1:
        plt.yticks([0.5, 0.75, 1])
    
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

def detach_axis(xy, amin=0, amax=None, ax=None, spine=None):
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

def box_off(ax=None,
            remove_spines=['right', 'top']):
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

#%% Heatmaps
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

#%% Errorbar
def errorbar_shade(x, y, yerr=None, **kw):
    if yerr is None:
        y1 = y[0,:]
        y2 = y[1,:]
    else:
        if yerr.ndim == 1:
            yerr = np.concatenate([-yerr[np.newaxis,:],
                                   yerr[np.newaxis,:]], axis=0)
        
        y1 = y + yerr[0,:]
        y2 = y + yerr[1,:]
        
        # print([x, y, y1, y2])
    
    if not ('alpha' in kw.keys()):
        kw['alpha'] = 0.2
    
    h = plt.fill_between(x, y1, y2, **kw)
    return h

#%% Psychophysics
def plot_binned_ch(x0, ch, n_bin=9, **kw):
    ix, x = np2.quantilize(x0, n_quantile=n_bin, return_summary=True)
    p = npg.aggregate(ix, ch, func='mean')
    se = npg.aggregate(ix, ch, func=np2.sem)
    
    h = plt.errorbar(x, p, yerr=se, **kw)
    
    return h, x, p, se

#%% Stats/probability
def ecdf(x0, *args, **kw):
    p, x = np2.ecdf(x0)
    return plt.step(np.concatenate([x[:1], x], 0),
                    np.concatenate([np.array([0.]), p], 0),
                    *args, **kw)

#%% Gaussian
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

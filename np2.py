#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:28:15 2018

@author: yulkang
"""
import numpy as np
import torch
from scipy import stats
import numpy_groupies as npg
import pandas as pd
from . import numpytorch

npt = numpytorch.npt_torch # choose between torch and np

#%% Shape
def vec_on(arr, dim, n_dim=None):
    arr = np.array(arr)
    if n_dim is None:
        n_dim = np.amax([arr.ndim, dim + 1])
    
#    if dim >= n_dim:
#        raise ValueError('dim=%d must be less than n_dim=%d' % (dim, n_dim))
    
    sh = [1] * n_dim
    sh[dim] = -1
    return np.reshape(arr, sh)

def cell2mat2(l, max_len=None):
    """
    INPUT: a list containing vectors
    OUTPUT: a matrix with NaN filled to the longest
    """
    if max_len is None:
        max_len = np.amax([len(l1) for l1 in l])
        
    n = len(l)
    m = np.zeros([n, max_len]) + np.nan
    
    for ii in range(n):
        l1 = l[ii]
        if len(l1) > max_len:
            m[ii,:] = l1[:max_len]
        elif len(l1) < max_len:
            m[ii,:len(l1)] = l1
        else:
            m[ii,:] = l1

    return m     

def dict_shapes(d):
    sh = {}
    for k in d.keys():
        v = d[k]
        if type(v) is list:
            sh1 = len(v)
            if sh1 == 0:
                compo = None
            else:
                compo = type(v[0])
        elif type(v) is np.ndarray:
            sh1 = v.shape
            compo = v.dtype.type
        elif torch.is_tensor(v):
            sh1 = tuple(v.shape)
            compo = v.dtype
        elif v is None:
            sh1 = 0
            compo = None
        else:
            sh1 = 1
            compo = None
        sh[k] = (type(v), compo, sh1)
    return sh

def DataFrame(dat):
    """
    Converts dict with 1- or 2-D np.ndarrays into DataFrame
    with 2-level MultiIndex of (name, column index)
    where all values are 2-D.
    :param dat: a dict()
    :return: pd.DataFrame
    """
    keys = dat.keys()
    l = []
    for key in keys:
        v = dat[key]
        assert type(v) is np.ndarray and v.ndim <= 2 and v.ndim >= 1, \
            '%s must be np.ndarray with 1 <= ndim <= 2 !' % key

        if v.ndim == 1:
            ix = pd.MultiIndex.from_product([[key]] + [[0]])
            l.append(pd.DataFrame(v[:,np.newaxis], columns=ix))
        else:
            ix = pd.MultiIndex.from_product([[key]] + [
                np.arange(s) for s in v.shape[1:]
            ])
            l.append(pd.DataFrame(v, columns=ix))
    return pd.concat(l, axis=1)

#%% Type
def is_None(v):
    return v is None or (type(v) is np.ndarray and v.ndim == 0)

#%% Stat
def sem(v, axis=0):
    v = np.array(v)
    if v.ndim == 1:
        return np.std(v) / np.sqrt(v.size)
    else:
        return np.std(v, axis=axis) / np.sqrt(v.shape[axis])
    
def quantilize(v, n_quantile=5, return_summary=False, fallback_to_unique=True):
    """Quantile starting from 0. Array is flattened first."""

    v = np.array(v)

    if fallback_to_unique:
        x, ix = uniquetol(v, return_inverse=True)
    
    if (not fallback_to_unique) or len(x) > n_quantile:
        n = v.size
        ix = np.int32(np.ceil((stats.rankdata(v, method='ordinal') + 0.) \
                              / n * n_quantile) - 1)
    
    if return_summary:
        x = npg.aggregate(ix, v, func='mean')
        return ix, x
    else:   
        return ix
    
def discretize(v, cutoff):
    """
    Discretize given cutoff.
    
    ix[i] = k if cutoff[k - 1] <= v[i] < cutoff[k]
    if v[i] < cutoff[0], ix[i] = 0
    if v[i] >= cutoff[-1], v[i] = len(cutoff) - 1
    """
    v = np.array(v)
    ix = np.zeros(v.shape)
    
    cutoff = list(cutoff)
    cutoff.append(np.inf)
    n = len(cutoff)
    
    for ii in range(1,n):
        ix[(v >= cutoff[ii - 1]) & (v < cutoff[ii])] = ii
    
    ix[v >= cutoff[-1]] = n - 1    
    return ix
    
def uniquetol(v, tol=1e-6, return_inverse=False, **kwargs):
    return np.unique(np.round(np.array(v) / tol) * tol, 
                     return_inverse=return_inverse, **kwargs)

def ecdf(x0):
    """
    Empirical distribution.
    INPUT:
    x0: a vector or a list
    OUTPUT: 
    p[i] = Pr(x0 <= x[i])
    x: sorted x0
    """
    
    n = len(x0)
    p = np.arange(1.,n+1.) / n
    x = np.sort(x0)
    return p, x

def argmax_margin(v, margin=0.1, margin_from='second', 
                  fillvalue=-1, axis=None, out=None):
    """
    argmax with margin; If within margin, use fillvalue instead.
    margin_from: 'second' or 'last'.
    """
    
    assert out is None, "'out' input argument is not supported yet!"

    if type(v) is not np.ndarray:
        v = np.array(v)
        
    if axis is None:
        r = np.reshape(v, v.size)
    else:
        dims = [axis] + list(set(range(v.ndim)) - set([axis]))
        r = np.transpose(v, dims)
        
    a = np.argmax(r, axis=0)
    s = np.sort(r, axis=0)
    
    if margin_from == 'second':
        m = s[-1, ...] - s[-2, ...]
    else:
        m = s[-1, ...] - s[0, ...]
        
    not_enough_margin = m < margin
    a[not_enough_margin] = fillvalue
    return a

def argmin_margin(v, **kw):
    """argmin with margin. See argmax_margin for details."""
    return argmax_margin(-v, **kw)

def sumto1(v, axis=None):
    if type(v) is np.ndarray:
        return v / v.sum(axis=axis, keepdims=True)
    else: # v is torch.Tensor
        return v / v.sum(axis, keepdim=True)

#%% Distribution
def pdf_trapezoid(x, center, width_top, width_bottom):
    height = 1. / ((width_top + width_bottom) / 2.)
    proportion_between = ((width_bottom - width_top) / width_bottom)
    width2height = height / proportion_between

    p = (1. - npt.abs(x - center) / (width_bottom / 2.)) * width2height
    p[p > height] = height
    p[p < 0] = 0
    return p

#%% Transform
def logit(v):
    """logit function"""
    return np.log(v) - np.log(1 - v)

def logistic(v):
    """inverse logit function"""
    return 1 / (np.exp(-v) + 1)

def softmax(dv):
    if type(dv) is torch.Tensor:
        edv = torch.exp(dv)
        p = edv / torch.sum(edv)
    else:
        edv = np.exp(dv)
        p = edv / np.sum(edv)
        
    return p

def softargmax(dv):
    p = softmax(dv)
    a = np.nonzero(np.random.multinomial(1, p))[0][0]
    return a

#%% Binary operations
def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
        
        from https://stackoverflow.com/a/38034801/2565317
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))
    
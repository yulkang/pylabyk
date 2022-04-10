#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:28:15 2018

@author: yulkang
"""
#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
import torch
from scipy import interpolate
from scipy import stats
import numpy_groupies as npg
import pandas as pd
from copy import deepcopy
from pylabyk import numpytorch
from typing import Union, Sequence, Iterable, Type, Callable, Tuple, List
from multiprocessing.pool import Pool as Pool0
# from multiprocessing import Pool

from .numpytorch import npy, npys

npt = numpytorch.npt_torch # choose between torch and np

#%% Shape
def ____SHAPE____():
    pass


def cat(arrays0, dim=0, add_dim=True):
    """
    @type arrays0: list
    """
    arrays = []
    if add_dim:
        for ii in range(len(arrays0)):
            arrays.append(np.expand_dims(arrays0[ii], dim))
    else:
        arrays = arrays0
    return np.concatenate(arrays, dim)


def vec_on(arr, dim, n_dim=None):
    arr = np.array(arr)
    if n_dim is None:
        n_dim = np.amax([arr.ndim, dim + 1])
    
#    if dim >= n_dim:
#        raise ValueError('dim=%d must be less than n_dim=%d' % (dim, n_dim))
    
    sh = [1] * n_dim
    sh[dim] = -1
    return np.reshape(arr, sh)


def cell2mat(c: np.ndarray, dtype=np.float) -> np.ndarray:
    """
    convert from object to numeric
    :param c:
    :param dtype:
    :return:
    """
    shape0 = c.shape
    vs = np.stack([
        v.astype(dtype) if isinstance(v, np.ndarray)
        else np.array(v).astype(dtype)
        for v in c.flatten()
    ])
    return np.reshape(vs, shape0 + vs[0].shape)


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


def mat2cell(m: np.ndarray, remove_nan=True) -> List[np.ndarray]:
    """
    Make a list of array
    :param remove_nan: if True (default), remove trailing NaNs from each row
    """
    if remove_nan:
        return [v[~np.isnan(v)] for v in m]
    else:
        return [v for v in m]


def mat2list(m: np.ndarray) -> List[np.ndarray]:
    """
    Make a list of array
    """
    return [v for v in m]


def dict_shapes(d, verbose=True):
    if not isinstance(d, dict):
        assert is_iter(d)
        d = {k: v for k, v in enumerate(d)}
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
            if isinstance(v, np.object) and v.size > 0:
                compo = type(v.flatten()[0])
            else:
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
        sh[k] = (sh1, type(v), compo)

        if verbose:
            if compo is None:
                str_compo = ''
            else:
                try:
                    str_compo = '[' + compo.__name__ + ']'
                except AttributeError:
                    str_compo = '[' + compo.__str__() + ']'
            print('%15s: %s %s%s' % (k, sh1, type(v).__name__, str_compo))

    return sh


def filt_dict(d: dict, incl: np.ndarray,
              copy=True, ignore_diff_len=True) -> dict:
    """
    Copy d[k][incl] if d[k] is np.ndarray and d[k].shape[1] == incl.shape[0];
    otherwise copy the whole value.
    @type d: dict
    @type incl: np.ndarray
    @rtype: dict
    """
    if copy:
        if ignore_diff_len:
            return {
                k: (deepcopy(v[incl]) if (isinstance(v, np.ndarray)
                                          and v.shape[0] == incl.shape[0])
                    else deepcopy(v))
                for k, v in d.items()
            }
        else:
            return {k: deepcopy(v[incl]) for k, v in d.items()}
    else:
        if ignore_diff_len:
            return {
                k: (v[incl] if (isinstance(v, np.ndarray)
                                          and v.shape[0] == incl.shape[0])
                    else v)
                for k, v in d.items()
            }
        else:
            return {k: v[incl] for k, v in d.items()}


def listdict2dictlist(listdict: list, to_array=False) -> dict:
    """
    @type listdict: list
    @param listdict: list of dicts with the same keys
    @return: dictlist: dict of lists of the same lengths
    @rtype: dict
    """
    d = {k: [d[k] for d in listdict] for k in listdict[0].keys()}
    if to_array:
        for k in d.keys():
            v = d[k]
            if torch.is_tensor(v[0]):
                v = np.array([npy(v1) for v1 in v])
            else:
                v = np.array(v)
            d[k] = v
    return d


def dictlist2listdict(dictlist: dict) -> list:
    keys = list(dictlist.keys())
    return [{k: dictlist[k][i] for k in keys}
            for i in range(len(dictlist[keys[0]]))]


def dictkeys(d, keys):
    return [d[k] for k in keys]


def rmkeys(d: dict, keys: Union[str, Iterable[str]]):
    if type(keys) is str:
        keys = [keys]
    return {k: v for k, v in d.items() if k not in keys}


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


def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of tensor v to the first
    @type v: np.ndarray
    @type ndim_en: int
    @rtype: np.ndarray
    """
    nd = v.ndim
    return v.transpose([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])
p2st = permute2st


def permute2en(v, ndim_st=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: np.ndarray
    :type ndim_st: int
    :rtype: np.ndarray
    """
    nd = v.ndim
    return v.transpose([*range(ndim_st, nd)] + [*range(ndim_st)])
p2en = permute2en


def ____COPY____():
    pass


def copy_via_pickle(obj):
    import pickle
    import io
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    return pickle.load(buf)


def ____BATCH____():
    pass


def static_vars(**kwargs):
    """
    Add kwargs as static variables.
    Example:
        @static_vars(counter=0)
        def foo():
            foo.counter += 1
            print "Counter is %d" % foo.counter

    From https://stackoverflow.com/a/279586/2565317
    :param kwargs: static variables to add to the decorated function
    :return: decorated function
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def arrayfun(fun, *args: np.ndarray):
    """

    :param fun:
    :param args: arrays
    :return: res[...] = fun(arrays[0][...], arrays[1][...], ...)
    """
    shape = args[0].shape
    args_flatten = [v.flatten() for v in args]
    res = [fun(*v) for v in zip(*args_flatten)]
    return np.array(res, dtype=np.object).reshape(shape)


def meshfun(fun, list_args, n_out=1, dtype=None, outshape_first=False):
    """
    EXAMPLE:
    out[i,j] = fun(list_args[0][i], list_args[1][j])
    @type fun: function
    @type list_args: Iterable
    @type n_out: int
    @type dtype: Iterable
    @param dtype: Iterable of dtype for each output; give None for
    default
    @param outshape_first:  If False (default), each output's shape is
    all args' shapes and the individual output's shape, concatenated
    in order. If True, the individual output shape comes first.
    @rtype: np.ndarray
    @return: tuple of outputs, each an np.ndarray.
    shape first.
    """
    shape_all = ()
    list_args1 = []
    if dtype is None:
        dtype = [None] * n_out
    for arg in list_args:
        try:
            shape_all += arg.shape
            list_args1 += [arg.flatten()]
        except:
            shape_all += (len(arg),)
            list_args1 += [arg]
    # shape_all += (n_out,)
    list_args1 = np.meshgrid(*list_args1, indexing='ij')
    for i in range(len(list_args1)):
        list_args1[i] = list_args1[i].flatten()
    res = []
    for args in zip(*list_args1):
        out1 = fun(*args)
        if n_out == 1:
            res.append([out1])
        else:
            res.append(list(out1))
    try:
        shape_each = res[0].shape[1:]
    except:
        shape_each = ()
    out = []
    for i_out in range(n_out):
        res1 = res[0][i_out]
        try:
            shape_each = res1.shape
        except:
            if isinstance(res1, dict):
                shape_each = ()
            else:
                try:
                    shape_each = (len(res1),)
                except:
                    shape_each = ()
        out.append(
            np.array(
                [res1[i_out] for res1 in res],
                dtype=dtype[i_out]
            ).reshape(shape_all + shape_each, order='C')
        )
        if outshape_first:
            out[-1] = p2st(out[-1], len(shape_each))
    return tuple(out)

    # out = np.transpose(
    #     np.array(res).reshape(shape_all + shape_each, order='C'),
    #     (
    #             [len(shape_all) - 1]
    #             + list(np.arange(len(shape_all) - 1))
    #             + list(len(shape_all) + np.arange(len(shape_each)))
    #     )
    # )
    # return out

def demo_meshfun():
    out = meshfun(
        lambda a, b: a + b * 10,
        [(1,2), (10, 20, 30)]
    )
    # out[i,j] = fun(arg0[i], arg1[j])
    print(out)

    out1, out2 = meshfun(
        lambda a, b: (a + b * 10, a + b),
        [(1,2), (10, 20, 30)],
        n_out=2
    )
    # out1[i,j], out2[i,j] = fun(arg0[i], arg1[j])
    print(out1)
    print(out2)

    out1, out2 = meshfun(
        lambda a, b: ([a, a + b], [a, a + b]),
        [(1,2), (10, 20, 30)],
        n_out=2,
        dtype=[None, np.object]
    )
    # out1[i,j], out2[i,j] = fun(arg0[i], arg1[j])
    print((out1, out1.shape, out1.dtype))
    print((out2, out2.shape, out2.dtype))

    return out, out1, out2


def ____TYPE____():
    pass


def is_None(v):
    return v is None or (type(v) is np.ndarray and v.ndim == 0)


def is_iter(v):
    return hasattr(v, '__iter__')


def is_sequence(v):
    return hasattr(v, '__len__')


def ____NAN____():
    pass


def nan2v(v0, v=0):
    v0[np.isnan(v0)] = v
    return v0


def ____STAT____():
    pass


def mean_distrib(p, v, axis=None):
    if axis is None:
        kw = {}
    else:
        kw = {'axis': axis}
    return (p * v).sum(**kw) / p.sum(**kw)


def var_distrib(p, v, axis=None):
    return np.clip(
        mean_distrib(p, v ** 2, axis=axis)
        - mean_distrib(p, v, axis=axis) ** 2,
        0, np.inf
    )

def std_distrib(p, v, axis=None):
    return np.sqrt(var_distrib(p, v, axis=axis))


def sem(v, axis=0):
    v = np.array(v)
    if v.ndim == 1:
        return np.std(v) / np.sqrt(v.size)
    else:
        return np.std(v, axis=axis) / np.sqrt(v.shape[axis])


def wmean(values: np.ndarray, weights: np.ndarray,
          axis=None) -> np.ndarray:
    return (values * weights).sum(axis=axis) / weights.sum(axis)


def wstd(values: np.ndarray, weights: np.ndarray,
         axis=None, keepdim=False) -> np.ndarray:
    """
    Return the weighted standard deviation.

    from: https://stackoverflow.com/a/2415343/2565317

    values, weights -- Numpy ndarrays with the same shape.
    """
    sum_wt = weights.sum(axis=axis, keepdims=True)
    avg = (values * weights).sum(axis=axis, keepdims=True) / sum_wt
    var = ((values - avg) ** 2 * weights).sum(axis=axis, keepdims=True) / sum_wt
    if not keepdim:
        var = np.squeeze(var, axis=axis)
    return np.sqrt(var)


def wsem(values: np.ndarray, weights: np.ndarray,
         axis=None, keepdim=False) -> np.ndarray:
    """
    Weighted standard error of mean.
    :param values:
    :param weights:
    :param axis:
    :param keepdim:
    :return:
    """
    return (
        wstd(values, weights, axis, keepdim)
        / np.sqrt(np.sum(weights, axis=axis, keepdims=keepdim))
    )


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
    
    ix[i] = 0 if v[i] < cutoff[0]
    ix[i] = k if cutoff[k - 1] <= v[i] < cutoff[k]
    v[i] = len(cutoff) if v[i] >= cutoff[-1]
    """
    v = np.array(v)
    ix = np.zeros(v.shape, dtype=np.long)
    
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
    :param x0: a vector or a list of samples
    :return: p[i] = Pr(x0 <= x[i]), x: sorted x0
    """
    
    n = len(x0)
    p = np.linspace(0, 1, n)
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


def argmedian(v, axis=None):
    median = np.median(v, axis=axis, keepdims=True)
    return np.argmin(np.abs(v - median), axis=axis)


def sumto1(v, axis=None, ignore_nan=True):
    if ignore_nan:
        if type(v) is np.ndarray:
            return v / np.nansum(v, axis=axis, keepdims=True)
        else:  # v is torch.Tensor
            return v / v.nansum(axis, keepdim=True)
    else:
        if type(v) is np.ndarray:
            return v / v.sum(axis=axis, keepdims=True)
        else:  # v is torch.Tensor
            return v / v.sum(axis, keepdim=True)


def maxto1(v, axis=None, ignore_nan=True):
    if ignore_nan:
        if type(v) is np.ndarray:
            return v / np.nanmax(v, axis=axis, keepdims=True)
        else:  # v is torch.Tensor
            return v / v.nanmax(axis, keepdim=True)
    else:
        if type(v) is np.ndarray:
            return v / np.amax(v, axis=axis, keepdims=True)
        else:  # v is torch.Tensor
            return v / v.max(axis, keepdim=True)


def nansem(v, axis=None, **kwargs):
    s = np.nanstd(v, axis=axis, **kwargs)
    n = np.sum(~np.isnan(v), axis=axis, **kwargs)
    return s / np.sqrt(n)


def wpercentile(w: np.ndarray, prct, axis=None):
    """
    """
    if axis is not None:
        raise NotImplementedError()
    cw = np.concatenate([np.zeros(1), np.cumsum(w)])
    cw /= cw[-1]
    f = interpolate.interp1d(cw, np.arange(len(cw)) - .5)
    return f(prct / 100.)

    # if axis is None:
    #     axis = 0
    #     v = v.flatten()
    #     w = w.flatten()
    # z = vec_on(np.zeros(v.shape[axis]), axis, v.ndim)
    # cv = np.cumsum(v, axis)
    # cv = np.concatenate([z, cv], axis)
    # cw = np.cumsum(w, axis)
    # cw = np.concatenate
    # f = stats.interpolate.interp1d(w, cv)


def wmedian(w, axis=None):
    return wpercentile(w, prct=50, axis=axis)


def pearsonr_ci(x,y,alpha=0.05):
    """
    calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals

    from https://gist.github.com/zhiyzuo/d38159a7c48b575af3e3de7501462e04
    """
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def info_criterion(nll, n_trial, n_param, kind='BIC'):
    """

    :param nll: negative log likelihood of the data given parameters
    :param n_trial:
    :param n_param:
    :param kind: 'BIC'|'NLL'
    :return: the chosen information criterion
    """
    if kind == 'AIC':
        return 2 * n_param + 2 * nll
    elif kind == 'nAIC':
        return n_param + nll
    elif kind == 'BIC':
        return n_param * np.log(n_trial) + 2 * nll
    elif kind == 'nBIC':
        # nBIC: following Bishop's convention except for the sign,
        #   since signs are used to choose the best model,
        #   and hence may introduce a bug downstream.
        return (n_param * np.log(n_trial) + 2 * nll) / 2
    elif kind == 'NLL':
        return nll
    else:
        raise ValueError()


def dkl(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
    """
    DKL[a || b]
    :param a:
    :param b:
    :param axis:
    :return: DKL[a || b] = sum(a * (log(a) - log(b)), axis)
    """
    return np.sum(a * (np.log(a) - np.log(b)), axis=axis)


def wsum_rvs(mu: np.ndarray, sigma: np.ndarray, w: np.ndarray
             ) -> (np.ndarray, np.ndarray):
    """
    Mean and covariance of weighted sum of random variables
    :param mu: [..., RV]
    :param sigma: [..., RV, RV]
    :param w: [RV]
    :return: mu_sum[...], variance_sum[...]
    """
    mu1 = mu * w  # type: np.ndarray
    ndim = mu1.ndim
    # not using axis=-1, to make it work with DataFrame and Series
    mu1 = mu1.sum(axis=ndim - 1)
    sigma1 = (sigma *  (w[..., None] * w[..., None, :])
              ).sum(axis=ndim).sum(axis=ndim - 1)
    return mu1, sigma1


#%% Distribution
def ____DISTRIBUTION____():
    pass

def pdf_trapezoid(x, center, width_top, width_bottom):
    height = 1. / ((width_top + width_bottom) / 2.)
    proportion_between = ((width_bottom - width_top) / width_bottom)
    width2height = height / proportion_between

    p = (1. - npt.abs(x - center) / (width_bottom / 2.)) * width2height
    p[p > height] = height
    p[p < 0] = 0
    return p


def ____CIRCSTAT____():
    pass


def circmean_distrib(p: np.ndarray, dim=-1, keepdim=False) -> np.ndarray:
    """
    Circular mean in radian.
    p is assumed to sum to 1 along dim, and is assumed to correspond to
    [0, 1/n, 2/n, ..., (n-1)/n] * 2 * pi radian, where n = p.shape[dim]
    :param p:
    :param dim: defaults to -1.
    :return: circmean
    """
    n = p.shape[dim]
    th = vec_on(np.linspace(0., 1. - 1. / n, n), dim, p.ndim)
    c = np.cos(th * 2. * np.pi)
    s = np.sin(th * 2. * np.pi)
    c1 = np.sum(p * c, dim, keepdims=keepdim)
    s1 = np.sum(p * s, dim, keepdims=keepdim)
    return np.arctan2(s1, c1)

def circvar_distrib(p: np.ndarray, dim=-1, keepdim=False) -> np.ndarray:
    """
    Circular variance = 1 - length of the resultant vector.
    p is assumed to sum to 1 along dim, and is assumed to correspond to
    [0, 1/n, 2/n, ..., (n-1)/n] * 2 * pi radian, where n = p.shape[dim]
    :param p:
    :param dim:
    :return: circvar
    """
    n = p.shape[dim]
    th = vec_on(np.linspace(0., 1. - 1. / n, n), dim, p.ndim)
    c = np.cos(th * 2. * np.pi)
    s = np.sin(th * 2. * np.pi)
    c1 = np.sum(p * c, dim, keepdims=keepdim)
    s1 = np.sum(p * s, dim, keepdims=keepdim)
    return 1. - np.sqrt(c1 ** 2 + s1 ** 2)

def rad2deg(rad):
    return rad / np.pi * 180.


def deg2rad(deg):
    return deg / 180. * np.pi


def circdiff(angle1, angle2, maxangle=None):
    """
    :param angle1: angle scaled to be between 0 and maxangle
    :param angle2: angle scaled to be between 0 and maxangle
    :param maxangle: max angle. defaults to 2 * pi.
    :return: angle1 - angle2, shifted to be between -.5 and +.5 * maxangle
    """
    if maxangle is None:
        maxangle = np.pi * 2
    return (((angle1 / maxangle)
             - (angle2 / maxangle) + .5) % 1. - .5) * maxangle


def pconc2conc(pconc: np.ndarray) -> np.ndarray:
    # pconc = np.clip(pconc, a_min=1e-6, a_max=1-1e-6)
    # pconc = np.clip(pconc, 0., 1.)
    return 1. / (1. - pconc) - 1.


def conc2pconc(conc: np.ndarray) -> np.ndarray:
    return 1. - 1. / (conc + 1.)


def rotation_matrix(rad, dim=(-2, -1)):
    if np.ndim(rad) < 2:
        if not isinstance(rad, np.ndarray):
            rad = np.array(rad)
        rad = np.expand_dims(np.array(rad),
                             list(-(np.arange(2 - np.ndim(rad)) + 1)))
    cat = np.concatenate
    return cat((
        cat((np.cos(rad), -np.sin(rad)), dim[1]),
        cat((np.sin(rad), np.cos(rad)), dim[1])), dim[0])


def rotate(v, rad: np.ndarray) -> np.ndarray:
    """

    :param v: [batch_dims, (x0, y0)]
    :param rad: [batch_dims]
    :return: [batch_dims, (x, y)]
    """
    rotmat = rotation_matrix(np.expand_dims(rad, (-1, -2)))
    return np.squeeze(rotmat @ np.expand_dims(v, -1), -1)


def ellipse2cov(th, long_axis, short_axis) -> np.array:
    """

    :param th: radian
    :param long_axis:
    :param short_axis:
    :return: covariance matrix
    """
    rot = rotation_matrix(th)
    cov = rot @ np.diag([long_axis, short_axis]) ** 2 @ rot.T
    return cov


def ____GEOMETRY____():
    pass


def lineseg_dists(p, a, b):
    """
    Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892
    From: https://stackoverflow.com/a/58781995/2565317

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


def distance_point_line(
        point: np.ndarray,
        line_st: np.ndarray,
        line_en: np.ndarray) -> np.ndarray:
    """
    Adapted from https://stackoverflow.com/a/48137604/2565317
    :param point: [index, (x, y)]
    :param line_st: [index, (x, y)]
    :param line_en: [index, (x, y)]
    :return: distance[index]
    """
    d = np.cross(
        line_en - line_st, point - line_st
    ) / np.linalg.norm(line_en - line_st)
    return d


def ____TRANSFORM____():
    pass


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


def project(a, b, axis=None, scalar_proj=False):
    """
    Project vector a onto b (vector dimensions are along axis).
    :type a: np.array
    :type b: np.array
    :type axis: None, int
    :rtype: np.array
    """
    proj = np.sum(a * b, axis) / np.sum(b**2, axis)
    if scalar_proj:
        return proj
    else:
        return proj * b


def inverse_transform(xy0: np.ndarray, xy1: np.ndarray) -> np.ndarray:
    """

    :param xy0: [(x, y), ix, iy]: original grid
    :param xy1: [(x, y), ix, iy]: transformed grid
    :return: xy2: [(x, y), ix, iy]: inverse-transformed original grid
    """
    from scipy.interpolate import griddata
    if xy0.ndim == 3:
        xy2 = np.stack([
            np.stack([
                inverse_transform(xy00, xy11)
                for xy00, xy11 in zip(xy0[0].T, xy1[0].T)
            ]).T,
            np.stack([
                inverse_transform(xy00, xy11)
                for xy00, xy11 in zip(xy0[1], xy1[1])
            ])
        ])
    elif xy0.ndim == 1:
        xy2 = griddata(xy1, xy0, xy0, method='linear')
    else:
        raise ValueError()
    return xy2


def ____BINARY_OPS____():
    pass


def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
        
        from https://stackoverflow.com/a/38034801/2565317
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))


def ____COMPARISON____():
    pass


def startswith(a: Sequence, b: Sequence) -> bool:
    """
    a and b should be the same type: tuple, list, np.ndarray, or torch.tensor

    EXAMPLE:
        startswith(np.array([1, 2, 3]), np.array([1, 2]))
            True
        startswith(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
            False
        startswith((1, 2), (1, 2, 3))
            False
        startswith((1, 2), (1,))
            True

    :param a: tuple, list, np.ndarray, or torch.tensor
    :param b: same type as a
    :return: True if a starts with b
    """
    v = len(a) >= len(b) and a[:len(b)] == b
    try:
        return v.all()
    except AttributeError:
        return v


def ____IMAGE____():
    pass


def nancrosscorr(
        fr1: np.ndarray,
        fr2: np.ndarray = None,
        thres_n=2,
        fillvalue=np.nan,
        processes=1,
) -> np.ndarray:
    """
    Normalized cross-correlation ignoring NaNs.
    As in Barry et al. 2007
    :param fr1: [x, y, batch]
    :param fr2: [x, y, batch]
    :param fillvalue:
    :param thres_n: Minimum number of non-NaN entries to compute crosscorr with.
    :param processes: >0 to run in parallel
    :return: cc[i_dx, i_dy, batch]
    """
    if fr2 is None:
        fr2 = fr1

    is_fr1_ndim2 = fr1.ndim == 2
    if is_fr1_ndim2:
        fr1 = fr1[..., None]

    is_fr2_ndim2 = fr2.ndim == 2
    if is_fr2_ndim2:
        fr2 = fr2[..., None]

    assert fr1.ndim == 3
    assert fr2.ndim == 3
    assert thres_n >= 2, 'to compute correlation thres_n needs to be >= 2'

    fsh1 = np.array(fr1.shape[:2])
    fsh2 = np.array(fr2.shape[:2])
    # csh = fsh1 + fsh2

    # NOTE: pad smaller of the two to match max_shape + 2,
    #   + 2 to ensure both are padded on both sides to remove smoothing artifact
    max_sh0 = np.amax(np.stack([fsh1, fsh2], axis=0), axis=0)
    max_sh = max_sh0 + 2
    # max_sh = (max_sh // 2) * 2 + 1  # enforce odd numbers so it has a center
    pad1 = max_sh - fsh1
    # pad1 = np.stack([
    #     int(np.floor(pad1 / 2))])
    pad2 = max_sh - fsh2
    fr1 = np.pad(fr1, [
        (int(np.floor(pad1[0] / 2)),
         int(np.ceil(pad1[0] / 2))),
        (int(np.floor(pad1[1] / 2)),
         int(np.ceil(pad1[1] / 2))),
        (0, 0)
    ], constant_values=np.nan)
    fr2 = np.pad(fr2, [
        (int(np.floor(pad2[0] / 2)),
         int(np.ceil(pad2[0] / 2))),
        (int(np.floor(pad2[1] / 2)),
         int(np.ceil(pad2[1] / 2))),
        (0, 0)
    ], constant_values=np.nan)

    csh = max_sh0 * 2
    cc = np.zeros(tuple(csh) + fr1.shape[2:]) + fillvalue
    # fsh = np.amin(np.stack([fsh1, fsh2], axis=0), axis=0)
    # fsh = np.ceil(max_sh / 2).astype(int)
    fsh = max_sh0

    pool = Pool(processes=processes)
    # if processes > 0:
    #     pool = Pool(processes=processes)
    #     f_map = pool.map
    # else:
    #     def f_map(*args, **kwargs):
    #         return list(map(*args, **kwargs))

    # def ccorrs(dx: int):
    # cc0 = _ccorrs_given_dx(dx, csh, fillvalue, fr1, fr2, fsh, thres_n)

    dxs = np.arange(-fsh[0], fsh[0])
    cc[fsh[0] + dxs] = np.array(pool.map(
        _ccorrs_given_dx,
        ((dx, csh, fillvalue, fr1, fr2, fsh, thres_n) for dx in dxs)
    ))
    # cc[fsh[0] + dxs] = np.array(pool.map(ccorrs, dxs))

    # if processes > 0:
    #     pool.close()

    if is_fr1_ndim2 and is_fr2_ndim2:
        assert cc.shape[-1] == 1
        cc = cc[..., 0]

    return cc


def _ccorrs_given_dx(inp):
    """

    :param inp: dx, csh, fillvalue, fr1, fr2, fsh, thres_n
        dx: int
        csh: [(x, y)] shape of the results (cross correlation)
        fillvalue: what to fill when the number of bins < thres_n
        fr1: [x, y, batch]
        fr2: [x, y, batch]
        fsh: [(x, y)]
        thres_n: min number of bins required
    :return: cross_correlation[x, y]
    """
    dx, csh, fillvalue, fr1, fr2, fsh, thres_n = inp
    n_batch = fr1.shape[-1]
    cc0 = np.zeros([csh[1], n_batch]) + fillvalue
    if dx == 0:
        f1 = fr1
        f2 = fr2
    elif dx > 0:
        f1 = fr1[dx:]
        f2 = fr2[:-dx]
    else:
        f1 = fr1[:dx]
        f2 = fr2[-dx:]
    for dy in range(-fsh[1], fsh[1]):
        if dy == 0:
            g1 = f1
            g2 = f2
        elif dy > 0:
            g1 = f1[:, dy:]
            g2 = f2[:, :-dy]
        else:
            g1 = f1[:, :dy]
            g2 = f2[:, -dy:]

        # g1 = g1.flatten()
        # g2 = g2.flatten()
        g1 = g1.reshape([np.prod(g1.shape[:2]), -1])
        g2 = g2.reshape([np.prod(g2.shape[:2]), -1])

        incl = np.all(~np.isnan(g1), -1) & np.all(~np.isnan(g2), -1)
        if np.sum(incl) >= thres_n:
            # cc0[dy + fsh[1]] = stats.pearsonr(g1[incl], g2[incl])[0]
            cc0[dy + fsh[1]] = pearsonr(g1[incl].T, g2[incl].T)

        # return cc0
    return cc0


def pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Same as scipy.stats.pearsonr, but works along dim=-1, w/o checks or pvalue.
    :param a:
    :param b:
    :param dim:
    :return:
    """
    xmean = x.mean(axis=-1, keepdims=True)
    ymean = y.mean(axis=-1, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    from scipy import linalg
    # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
    # scipy.linalg.norm(xm) does not overflow if xm is, for example,
    # [-5e210, 5e210, 3e200, -3e200]
    normxm = linalg.norm(xm, axis=-1, keepdims=True)
    normym = linalg.norm(ym, axis=-1, keepdims=True)

    threshold = 1e-13
    import warnings
    from scipy.stats import PearsonRNearConstantInputWarning
    if (
        np.any(normxm < threshold * np.abs(xmean)) or
        np.any(normym < threshold * np.abs(ymean))
    ):
        # If all the values in x (likewise y) are very close to the mean,
        # the loss of precision that occurs in the subtraction xm = x - xmean
        # might result in large errors in r.
        warnings.warn(PearsonRNearConstantInputWarning())

    # YK: Assume dot product along the last dim;
    #   the preceding dims are considered batch
    r = ((xm / normxm) * (ym / normym)).sum(-1)
    # r = np.dot(xm / normxm, ym / normym)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = np.clip(r, -1., 1.)
    return r


def ____MULTIPROCESSING____():
    pass


class PoolParallel(Pool0):
    def __del__(self):
        self.close()
        # print('Closed!')  # CHECKED


class PoolSim:
    def map(self, fun, iter, chunksize=1, **kwargs):
        # ignore chunksize
        return list(map(fun, iter, **kwargs))

    def starmap(self, fun, iter, chunksize=1, **kwargs):
        # ignore chunksize
        from itertools import starmap
        return list(starmap(fun, iter, **kwargs))

    def close(self):
        pass


def Pool(
        processes=None, *args, **kwargs
):
    """

    :param processes: When < 1, use n_processors + processes.
        When 1, do not use multiprocessing (only simulate).
    :param args:
    :param kwargs:
    :return:
    """
    import multiprocessing
    n_processors = multiprocessing.cpu_count()

    if processes is None:
        processes = n_processors
    elif processes < 1:
        processes = n_processors + processes
    elif (0 < processes) and (processes < 1):
        processes = int(np.clip(n_processors * processes,
                                a_min=1, a_max=n_processors))

    if processes == 1:
        return PoolSim()
    else:
        return PoolParallel(processes=processes, *args, **kwargs)


def fun_deal(f, inp):
    return f(*inp)


def arrayobj1d(inp: Iterable, copy=False):
    """
    Return a 1D np.ndarray of dtype=np.object.
    Different from np.array(inp, dtype=np.object) because the latter may
    return a multidimensional array, which gets flattened when fed to
    np.meshgrid, unlike the output from this function.
    """
    return np.array([None] + list(inp), dtype=np.object, copy=copy)[1:]


def scalararray(inp) -> np.ndarray:
    """
    Return a scalar np.ndarray of dtype=np.object.
    :param inp:
    :return:
    """
    return np.array([None, inp], dtype=np.object)[[1]].reshape([])


def meshgridflat(*args, copy=False):
    """
    flatten outputs from meshgrid, for use with np.vectorize()
    :param args:
    :param copy: whether to copy during meshgrid
    :return:
    """
    outputs = np.meshgrid(*args, indexing='ij', copy=copy)  # type: Iterable[np.ndarray]
    outputs = [v.flatten() for v in outputs]
    return outputs


def vectorize_par(f: Callable, inputs: Iterable,
                  pool: Pool = None, processes=None, chunksize=1,
                  nout=None, otypes: Union[Sequence[Type], Type] = None,
                  use_starmap=True,
                  ) -> Sequence[np.ndarray]:
    """
    Run f in parallel with meshgrid of inputs along each input's first dimension
    and return the expanded outputs.
    See demo_vectorize_par() for examples.
    If you get an error like __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__()
    then you may have to force using the 'spawn' method in your main script:
        if __name__ == '__main__':
            import multiprocessing
            multiprocessing.set_start_method('spawn')
    See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

    :param f: function. This function should be on the top level of the
        module (e.g., cannot be a lambda or a nested function).
    :param inputs: Iterable of arguments. Each can be iterable or not.
        e.g., vectorize_par(np.ones, [np.arange(5), 2, np.arange(2], nout=1)
        will give an output of
    :param pool: a Pool object. If None (default), one will be created.
    :param processes: Number of parallel processes.
        If 1, use PoolSim, which does not use multiprocessing.
        If None (default), set to the number of CPU cores.
        If < 1, use n_processors + processes.
        Ignored if pool is given.
    :param chunksize: Giving an integer larger than 1 may boost efficiency.
    :param nout: If unspecified, set to the length of the first output from
        pool.starmap(). Give nout=1 to suppress this behavior when using
        functions that return iterables, e.g., np.ones
    :param otypes: output type(s) of f.
    :param use_starmap: if True (default), the inputs are given as multiple
        arguments to f.
        If False, an iterable containing all inputs is given as one argument
        to f.
        Ignored if processes=1 and multiprocessing is not used.
    :return: (iterable of) outputs from f.
    """
    inputs = [inp if (isinstance(inp, np.ndarray) and type(inp[0]) is np.object)
              else (arrayobj1d(inp) if is_iter(inp)
                    else arrayobj1d([inp]))
              for inp in inputs]
    lengths = [len(inp) for inp in inputs]
    mesh_inputs = np.meshgrid(*inputs, indexing='ij')  # type: Iterable[np.ndarray]
    mesh_inputs = [m.flatten() for m in mesh_inputs]
    m = zip(*mesh_inputs)
    m = [m1 for m1 in m]

    if pool is None:
        pool = Pool(processes=processes)  # type: PoolParallel

    # if processes == 0:
    #     use_starmap = False

    if chunksize is None:
        # NOTE: this doesn't seem to work well, unlike chunksize=1.
        #   Need further experiment.
        chunksize = np.max([
            int(np.floor(np.prod(lengths) / pool._processes)),
            1
        ])

    if use_starmap:
        outs = pool.starmap(f, m, chunksize=chunksize)
    else:
        outs = pool.map(f, m, chunksize=chunksize)

    if nout is None:
        try:
            nout = len(outs[0])
        except TypeError:
            nout = 1

    if otypes is None:
        otypes = [np.object] * nout
    elif not is_sequence(type(otypes)):
        otypes = [otypes] * nout

    # NOTE: deliberately keeping outs, outs1, and outs2 for debugging.
    #  After confirming everything works well, rename all to "outs"
    #  to save memory.
    # DEF: outs1[argout][i_input_flattened]
    if nout > 1:
        outs1 = zip(*outs)
    else:
        if use_starmap:
            outs1 = [outs]
        else:
            # Reverse the action of map() putting each output in a list
            outs1 = [[out1[0] for out1 in outs]]

    # --- outs2: reshape to inputs' dimensions
    # DEF: outs2[argout][i_input1, i_input2, ...]
    outs2 = [arrayobj1d(out).reshape(lengths) for out in outs1]

    # --- outs3: set to a correct otype
    # DEF: outs3[argout][i_input1, i_input2, ...]
    outs3 = [cell2mat(out, otype) if otype is not np.object
             else out
             for out, otype in zip(outs2, otypes)]
    return outs3


def demo_vectorize_par():
    import torch

    def f(a, b):
        return a + b, a * b

    out = vectorize_par(f, [(10, 20), (1, 2, 3)])
    print(out[0])
    print(out[1])
    """
    Output:
    [[11 12 13]
     [21 22 23]]
    [[10 20 30]
     [20 40 60]]
    """

    out = vectorize_par(torch.ones, [(2, 3), (4, 5, 6)], nout=1)
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            print(
                'out[%d,%d].shape: (%d,%d)' % (row, col, *out[row, col].shape))
    """
    Output:
    out[0,0].shape: (2,4)
    out[0,1].shape: (2,5)
    out[0,2].shape: (2,6)
    out[1,0].shape: (3,4)
    out[1,1].shape: (3,5)
    out[1,2].shape: (3,6)
    """


def nanautocorr(firing_rate: np.ndarray, thres_n=2) -> np.ndarray:
    """
    Normalized autocorrelation ignoring NaNs.
    As in Krupic et al. 2015, which corrected typos in Hafting et al. 2005.
    :param firing_rate: [x, y]
    :param thres_n: Minimum number of non-NaN entries to compute autocorr with.
    :return: ac[i_dx, i_dy]
    """
    f = firing_rate
    assert f.ndim == 2
    assert thres_n >= 2, 'to compute correlation thres_n needs to be >= 2'

    fsh = np.array(f.shape)
    ash = fsh * 2
    # DEBUGGED: ash = fsh * 2 - 1 is too small: index needs to be fsh * 2 - 1,
    #  which needs ash to be at least fsh * 2.

    ac = np.zeros(ash) + np.nan
    for i in range(-fsh[0], fsh[0]):
        if i == 0:
            f1 = f
            f2 = f
        elif i > 0:
            f1 = f[i:]
            f2 = f[:-i]
        else:
            f1 = f[:i]
            f2 = f[-i:]

        for j in range(-fsh[1], fsh[1]):
            if j == 0:
                g1 = f1
                g2 = f2
            elif j > 0:
                g1 = f1[:, j:]
                g2 = f2[:, :-j]
            else:
                g1 = f1[:, :j]
                g2 = f2[:, -j:]

            g1 = g1.flatten()
            g2 = g2.flatten()

            incl = ~np.isnan(g1) & ~np.isnan(g2)
            if np.sum(incl) >= thres_n:
                ac[i + fsh[0], j + fsh[1]] = stats.pearsonr(
                    g1[incl], g2[incl])[0]
    return ac


def nansmooth(u, sigma=1.):
    from scipy import ndimage

    isnan = np.isnan(u)

    v = u.copy()
    v[isnan] = 0.
    vv = ndimage.gaussian_filter(v, sigma=sigma)

    w = 1. - isnan
    ww = ndimage.gaussian_filter(w, sigma=sigma)

    r = vv / ww
    r[isnan] = np.nan

    return r


def convolve_time(src, kernel, dim_time=0, mode='same'):
    """
    @type src: np.ndarray
    @type kernel: np.ndarray
    @type dim_time: int
    @rtype: np.ndarray
    """
    if kernel.ndim == 1 and kernel.ndim < dim_time + 1:
        kernel = vec_on(kernel, dim_time, src.ndim)
    if kernel.ndim < src.ndim:
        kernel = np.expand_dims(
            kernel,
            np.arange(kernel.ndim, src.ndim)
        )
    if np.mod(kernel.shape[dim_time], 2) != 1:
        pad_width = np.zeros((kernel.ndim, 2), dtype=np.long)
        pad_width[dim_time, 1] = 1
        kernel = np.pad(kernel, pad_width, mode='constant')

    len_kernel_half = (kernel.shape[dim_time] - 1) // 2
    pad_width = np.zeros((src.ndim, 2), dtype=np.long)
    pad_width[dim_time, :] = len_kernel_half
    src = np.pad(src, pad_width, mode='constant')

    from scipy import ndimage
    dst = ndimage.convolve(src, kernel, mode='constant')
    dst = np.moveaxis(dst, dim_time, 0)

    if mode == 'same':
        dst = dst[:-(len_kernel_half * 2)]
    elif mode == 'full':
        pass
    else:
        raise ValueError('Unsupported mode=%s' % mode)
    dst = np.moveaxis(dst, 0, dim_time)
    return dst


def demo_convolve_time():
    # src = np.ones(3)
    src = np.array([1., 0., 0., 0., 0.])
    kernel = np.array([2., 3., 1., 0.])
    res = convolve_time(src, kernel, mode='same')

    from matplotlib import pyplot as plt
    plt.plot(kernel, 'b-')
    plt.plot(res, 'ro')
    plt.show()
    print(res)
    print((src.shape, kernel.shape, res.shape))

    src2 = vec_on(src, 2, 3)
    res = convolve_time(src2, kernel, dim_time=2)

    plt.plot(kernel, 'b-')
    plt.plot(res.flatten(), 'ro')
    plt.show()
    print(res)
    print((src2.shape, kernel.shape, res.shape))


def ____TIME____():
    pass


def timeit(fun, *args, repeat=1, return_out=False, **kwargs):
    """
    Accepts function, args, and kwargs; can return the output too.
    :type repeat: long
    :type fun: function
    :type return_out: bool
    :return: t_en - t_st, output from the function
    :rtype: (float, tuple)
    """
    import time

    assert repeat >= 1
    out = None
    t_st = time.time()
    for i in range(repeat):
        out = fun(*args, **kwargs)
    t_en = time.time()
    t_el = t_en - t_st
    if return_out:
        return t_el, out
    else:
        return t_el


def nowstr():
    import datetime
    return '{date:%Y%m%dT%H%M%S}'.format(date=datetime.datetime.now())


def ____STRING____():
    pass


def truncate_at(s: str, max_len: int) -> str:
    return s[:min([max_len, len(s)])]


def replace(s: str, src_dst: Iterable[Tuple[str, str]]) -> str:
    """

    :param s: string
    :param src_dst: [(src1, dst1), (src2, dst2), ...]
    :return: string with srcX replaced with dstX
    """
    for src, dst in src_dst:
        s = s.replace(src, dst)
    return s


def shorten_dict(d: dict, src_dst=()):
    return {k: shorten(v, src_dst) for k, v in d.items()}


def shorten(v, src_dst: Iterable[Tuple[str, str]] = ()) -> Union[str, None]:
    """

    :param v: string, Iterable[Number], or Number
    :param src_dst: [(src1, dst1), (src2, dst2), ...]
    :return: string with srcX replaced with dstX, or printed '%g,%g,...'
    """
    if isinstance(v, str):
        return replace(v, src_dst)
    elif isinstance(v, bool):
        return '%d' % int(v)
    elif is_iter(v):
        try:
            v = list(npy(v))
            if isinstance(v[0], str):
                return '%s' % (','.join([
                    ('%s' % shorten(v1, src_dst))
                    for v1 in v]))
            else:
                return '%s' % (','.join([
                    ('%s' % shorten(v1, src_dst))
                    for v1 in npy(v).flatten()]))
        except TypeError:
            return '%g' % v
    elif v is None:
        return None
    else:
        return '%g' % v


def filt_str(s, filt_preset='alphanumeric', replace_with='_'):
    import re

    if filt_preset == 'alphanumeric':
        f = r'\W+'
    else:
        raise ValueError()
    return re.sub(f, replace_with, s)


make_alphanumeric = filt_str
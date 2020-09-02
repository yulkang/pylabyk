#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:11:31 2018

@author: yulkang
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from scipy import stats
#%%

def ms2lognorm(m0, s0):
    '''mean and stdev into mu and sigma params of the lognormal distribution'''
    m = m0 + 0.
    s = s0 + 0.
    sig = np.sqrt(np.log((s / m) ** 2. + 1.))
    mu = np.log(m) - sig ** 2. / 2.
    return mu, sig

def lognorm2ms(mu0, sig0):
    mu = mu0 + 0.
    sig = sig0 + 0.
    m = np.exp(mu + sig ** 2. / 2.)
    s = np.sqrt((np.exp(sig ** 2.) - 1.) * np.exp(2. * mu + sig ** 2.))
    return m, s

def beta_mixture_of_betas(a0, b0, a, b):
    p = np.arange(0., 1., 0.01)
    n = p.size
    pp = np.zeros(n)
    
    raise ValueError('Not implemented yet!')
    
    # for [] = stats.beta.pdf(p, a, b)
    pass


def ____Regression____():
    pass


eng = None


def get_matlab_engine():
    import matlab.engine
    global eng
    if eng is None:
        eng = matlab.engine.start_matlab()
    return eng


def lsqcubic(X, Y, sX=None, sY=None, tl=1e-6, nargout=8):
    """
    Model-2 least squares fit from weighted data.
    Ported from MATLAB lsqcubic.m by Esward T Peltzer (rev Mar 17 2016)
    Requires lsqcubic.m and lsqfitma.m in the current working directory
    or matlab engine's search path.
    https://www.mbari.org/results-for-model-i-and-model-ii-regressions/

    @param X: x data (vector)
    @param Y: y data (vector)
    @param sX: uncertainty of x data (vector)
    @param sY: uncertainty of y data (vector)
    @param tl: test limit for difference between slope iterations

    @return: (m, b, r, sm, sb, xc, yc, ct)
    m: slope
    b: y-intercept
    r: weighted correlation coefficient
    sm: standard deviation of the slope
    sb: standard deviation of the y-intercept
    xc: weighted mean of x values
    yc: weighted mean of y values
    ct: count: number of iterations
    @rtype: (float, float, float, float, float, float, float, float)
    """
    import matlab
    from . import matlab2py as m2p
    eng = get_matlab_engine()

    if sX is None:
        sX = np.ones_like(X)
    if sY is None:
        sY = np.ones_like(Y)

    return eng.lsqcubic(*[m2p.array2matrix(v) for v in [
        X, Y, sX, sY, tl,
    ]],
        nargout=nargout
    )

model2regr = lsqcubic
type2regr = lsqcubic
regress2 = lsqcubic
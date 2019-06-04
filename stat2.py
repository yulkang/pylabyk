#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:11:31 2018

@author: yulkang
"""

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
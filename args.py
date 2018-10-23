#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:58:48 2018

@author: yulkang
"""

def varargin2props(obj, kw, skip_absent=True):
    keys0 = obj.__dict__.keys()
    for key in kw.keys():
        if (key in keys0) or not skip_absent:
            obj.__dict__[key] = kw[key]
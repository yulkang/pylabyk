#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:58:48 2018

@author: yulkang
"""

#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

from collections import OrderedDict as odict
import hashlib
import os


def varargin2props(obj, kw, skip_absent=True, error_absent=False):
    keys0 = obj.__dict__.keys()
    for key in kw.keys():
        if (key in keys0):
            obj.__dict__[key] = kw[key]
        elif error_absent:
            raise AttributeError('Attribute %s does not exist!' % key)
        elif not skip_absent:
            obj.__dict__[key] = kw[key]


def kwdefault(kw_given, **kw_default):
    """
    To ensure the order is preserved, use odict or [(key:value), ...]
    :param kw_given:
    :param kw_default:
    :return:
    """
    if kw_given is None:
        kw_given = {}
    kw_given = odict(kw_given)
    kw_default = odict(kw_default)
    for k in kw_given:
        kw_default[k] = kw_given[k]
    return kw_default


def kwdef(kw_given, kw_def=None,
          sort_merged=True,
          sort_def=False, sort_given=False,
          skip_None=False,
          def_bef_given=True):
    """
    To ensure the order is preserved, use odict or [(key:value), ...]
    :param kw_given:
    :param kw_def:
    :param sort_merged: if True, ignore sort_def, sort_given, def_bef_given
    :param sort_def:
    :param sort_given:
    :param def_bef_given:
    :rtype: odict
    """
    if kw_def is None:
        kw_def = {}

    kw_def = odict(kw_def)
    kw_given = odict(kw_given)

    if sort_merged:
        if def_bef_given:
            pass
            # raise Warning('def_bef_given=True is ignored when sort_merged=True')
        for k in kw_given:
            kw_def[k] = kw_given[k]
        kw_merged = odict(sorted(kw_def.items()))
    else:
        if sort_def:
            kw_def = odict(sorted(kw_def.items()))
        if sort_given:
            kw_given = odict(sorted(kw_given.items()))
        if def_bef_given:
            kw_merged = kw_def
            for k in kw_given:
                kw_merged[k] = kw_given[k]
        else:
            kw_merged = kw_given
            for k in kw_def:
                kw_merged[k] = kw_def[k]
    if skip_None:
        k_to_pop = [k for k, v in kw_merged.items() if v is None]
        for k in k_to_pop:
            kw_merged.pop(k)
    return kw_merged


def kwdefs(kws, **kwargs):
    if type(kws) in {dict, odict}:
        kws = [kws]
    res = kws[0].copy()
    for ii in range(1, len(kws)):
        res = kwdef(kws[ii], res, **kwargs)
    return res


def merge_subdict(d0, key):
    d = d0.copy()
    subdict = d.pop(key)
    return kwdef(subdict, d)


def merge_subdict_recur(d0):
    """
    :type d0: Union[dict, odict]
    :return:
    """
    d = d0.copy()
    keys = [k for k in d.keys()]
    for key in keys:
        if type(d[key]) is dict or type(d[key]) is odict:
            d[key] = merge_subdict_recur(d[key])
            d = merge_subdict(d, key)
    return d


def merge_fileargs(list_of_kws, **kwargs):
    return merge_subdict_recur(kwdefs(list_of_kws, **kwargs))


def dict2fname(d, skip_None=True):
    """
    :type d: Union[odict, dict]
    :param skip_None:
    :return:
    """
    def to_include(k):
        if skip_None:
            return d[k] is not None
        else:
            return True
    return '+'.join(['%s=%s' % (k, d[k]) for k in d if to_include(k)])


def rmkeys(d, keys):
    return {k:v for k, v in d.items() if k not in keys}


def filename2hash(filename):
    return hashlib.md5(filename.encode('utf-8')).hexdigest()


def fullpath2hash(fullpath):
    """
    :param fullpath: only file name is hashed, saving ext or ext.zip
    :type fullpath: string
    :rtype: string
    """
    pth, name_ext = os.path.split(fullpath)
    name, ext = os.path.splitext(name_ext)
    if ext == '.zip':
        name, ext2 = os.path.splitext(name)
    else:
        ext2 = ''

    name = filename2hash(name)
    name_ext = name + ext2 + ext
    return os.path.join(pth, name_ext)


def dict2hash(d, skip_None=True):
    return filename2hash(dict2fname(d, skip_None=skip_None))
#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
import pandas as pd
import h5py
import matlab, matlab.engine

#%%
def unpackarray(s, max_unpack=5, squeeze=True):
    if s is not np.ndarray:
        s = np.array(s)

    i_unpack = 0
    while i_unpack < max_unpack \
            and type(s[0]) is np.ndarray \
            and s.ndim == 1:
        dtype_ = s.dtype
        s = np.array([s1[0] if s1.size > 0 else np.array([], dtype=dtype_)
                      for s1 in s])
        i_unpack += 1

    if squeeze:
        s = np.squeeze(s)

    # if type(s[0]) is np.ndarray:
    #     s = np.array([list(s1) for s1 in s])
    return s


def array2matrix(v):
    """
    @type v: np.ndarray
    @return:
    """
    if not isinstance(v, np.ndarray):
        # assume it is a scalar
        return matlab.double([[v]])
    if v.ndim == 1:
        return matlab.double([
            [v1] for v1 in v
        ])
    elif v.ndim == 2:
        return matlab.double([
            list(v1) for v1 in v
        ])
    else:
        raise ValueError('ndim must be <=2')


a2m = array2matrix


def structlist2df(slist, obj2dict=False, unpack=0, return_df=False):
    slist = unpackarray(slist)
    def elem2dict(elem):
        for ii in range(unpack):
            elem = elem[0]
        if obj2dict:
            return elem.__dict__
        else:
            return elem

    d = {}
    slist = [elem2dict(s) for s in slist]
    for k in slist[0].keys():
        if k[0] != '_':
            d[k] = [s[k] for s in slist]
            d[k] = unpackarray(d[k])

    if return_df:
        d = pd.DataFrame(data = d)
    return d

def hdf2dict(file_in, to_str=None, to_list=None, verbose=True):
    def asc2str(v):
        return np.squeeze(np.array(v, dtype=np.uint8)).tostring().decode(
            'ascii')

    def hdf2v(h):
        try:
            v = np.squeeze(np.array([hdf2v(h1) for h1 in h]))
        except TypeError:
            if type(h) == h5py.h5r.Reference:
                v = f[h]
                v = hdf2v(v)
            else:
                v = h
        return v

    # %%
    f = h5py.File(file_in, 'r')
    if verbose:
        print('Loaded ' + file_in)

    # %%
    # print({k:f[k]})
    d = {k: np.squeeze(np.array(f[k])) for k in f.keys() if k[0] != '#'}

    for k in (set(d.keys()) - set(to_str) - set(to_list)):
        if type(d[k][0]) == h5py.h5r.Reference:
            if verbose:
                print('Converting to referent: ' + k)
            d[k] = [np.squeeze(np.array(f[v])) for v in d[k]]  # keep as lists

    if to_str is not None:
        for k in to_str:
            if verbose:
                print('Converting to str: ' + k)
            d[k] = np.array([asc2str(f[v]) for v in d[k]])

    if to_list is not None:
        for k in to_list:
            if verbose:
                print('Converting to list: ' + k)
            d[k] = [np.squeeze(np.array(f[v])) for v in d[k]]  # keep as lists

    if verbose:
        print('Conversion done!')

    # %% Saving to zpkl can take a very long time
    # pkl_file = pth + nam + '.zpkl'
    # zpkl.save(d, pkl_file)
    # print('Saved to ' + pkl_file)
    #
    # #%% Test
    # dat = zpkl.load(pkl_file)
    # dict_shapes(dat)

    return d, f

def get_dict_row(d, rows):
    return {k:d[k][rows] for k in d.keys()}

def set_dict_row(d, rows, values):
    for k in values.keys():
        d[k][rows] = values[k]
    return d

import numpy as np
import pandas as pd

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

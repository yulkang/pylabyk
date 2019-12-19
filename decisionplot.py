from scipy.io import loadmat
import numpy as np
from pprint import pprint
from importlib import reload
from matplotlib import pyplot as plt
import numpy_groupies as npg
import pandas as pd

from lib.pylabyk import plt2
from lib.pylabyk import np2
from lib.pylabyk import argsutil
from lib.pylabyk.numpytorch import npy, npys

#%%
def hist_ch_rt(ch, rt, n_ch=2, nt=None, dt=1/75, to_plot=True):
    """

    @param ch: ch[trial]
    @param rt: rt[trial]
    @param n_ch:
    @param nt:
    @return:
    """
    if nt is None:
        nt = np.amax(rt) + 1
    n = npg.aggregate(np2.cat([ch, rt]), 1., 'sum', [n_ch, nt])
    t = np.arange(nt) * dt

    if to_plot:
        h = plt.plot(t, n.T)
    else:
        h = None
    return n, t, h
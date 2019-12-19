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

class Decision(object):
    def __init__(
            self,
            max_t=5,
            dt=1/75,
            n_ch=2,
    ):
        self.max_t = max_t
        self.dt = dt
        self.n_ch = n_ch

    @property
    def nt(self):
        return int(self.max_t // self.dt)

    @property
    def t(self):
        return self.dt * np.arange(self.nt)

    def hist_ch_rt(self, ch, rt, to_plot=True):
        """

        @param ch: ch[trial]
        @param rt: rt[trial]
        @param n_ch:
        @param nt:
        @return:
        """
        n = npg.aggregate(np2.cat([ch, rt]), 1., 'sum',
                          [self.n_ch, self.nt])

        if to_plot:
            h = plt.plot(self.t, n.T)
        else:
            h = None
        return n, h

    def ev_for_ch(self, ev, ch, rt=None):
        pass
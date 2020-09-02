#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

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

    def hist_ch_rt(
            self, ch, rt,
            to_plot=True,
            normalize='density',
    ):
        """

        @param ch: ch[trial]
        @param rt: rt[trial]
        @param n_ch:
        @param nt:
        @return:
        """
        n = npg.aggregate(
            np2.cat([ch, np.round(rt / self.dt).astype('long')]),
            1., 'sum', [self.n_ch, self.nt])

        if normalize == 'density':
            n = n / np.sum(n) / self.dt
        elif normalize == 'None':
            pass
        else:
            raise ValueError('Unsupported normalize=%s' % normalize)

        if to_plot:
            h = plt.plot(self.t, n.T)
        else:
            h = None
        return n, h

    def ev_for_ch(
            self, ev, ch,
            rt=None,
            summary_across_trials='mean_residual',
            summary_within_trial='None',
            t_align=None,
            t_plot=None,
            to_plot=True
    ):
        """
        @param ev: [trial, it]
        @param ch: [trial]
        @param t_align: scalar or [trial]
        @param t_st: scalar or [trial]
        @param t_en: scalar or [trial]
        @param to_plot:
        @return: ev_for_ch[it], e_ev_for_ch[it]
        """

        n_tr = ev.shape[0]
        ev = ev.copy()
        if rt is not None:
            irt = (rt / self.dt).astype('long')
            for tr in range(n_tr):
                ev[tr, irt:] = np.nan

        ev = ev - np.nanmean(ev, 0, keepdims=True)
        if summary_within_trial == 'None':
            pass
        elif summary_within_trial == 'cumsum':
            ev = np.cumsum(ev, 1)
        else:
            raise ValueError('Unsupported summary_within_trial=%s' %
                             summary_within_trial)

        ev_for_ch = np.nanmean(ev * np.sign(ch[:, None] - 0.5), 0)
        e_ev_for_ch = np2.nansem(ev * np.sign(ch[:, None] - 0.5), 0)

        if t_plot is None:
            t_plot = self.t
        if t_align is not None:
            raise NotImplementedError()

        if to_plot:
            plt2.errorbar_shade(t_plot, ev_for_ch, e_ev_for_ch)
            plt.plot(t_plot, ev_for_ch)

        return ev_for_ch, e_ev_for_ch
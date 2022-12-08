#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.

from pylabyk import plt2
import pathlib
import os
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
from importlib import reload

def check_savefig() -> str:
    dir = pathlib.Path(__file__).parent.resolve()
    fname = os.path.join(dir, 'test_plt2/test_fig')

    fig = plt.figure()
    # axs = plt2.GridAxes(nrows=1, ncols=2)
    # plt.figure(axs.figure.number)
    # plt.sca(axs[0, 0])
    plt.plot([1, 3, 2])

    # plt.sca(axs[0, 1])
    # plt.imshow(np.tile(np.arange(5)[None], [3, 1]))

    # fig = axs.figure
    # fig = plt.gcf()
    plt2.savefig(fname, ext=('.png',), fig=fig)

    # check_loadfig(fname)
    return fname


def check_loadfig(fname: str = None):
    if fname is None:
        dir = pathlib.Path(__file__).parent.resolve()
        # fname = os.path.join(dir, 'test_plt2/test_fig')
        fname = os.path.join(dir, 'test_plt2/plt=traj_posterior+bi=0+bl=2+ct=.2+cw=.2+df=1.0+dp=1+dr=.75+dx=.04+eb=tknr+et=tknr+ig=2+iu=al+nb=1+nk=n+no=.99+ns=.15+nt=400+pc=goal+pn=from_prev_goal+pt=dense+rd=0.75+rf=2+s0=1948+sd=0+si=0+tp=.99+tr=.1+vf=40')

    fig_copy = plt2.loadfig(fname + '.mpl')
    fname_copy = fname + ' (2)'
    plt2.savefig(fname_copy + '.png', fig=fig_copy)

    img_orig = Image.open(fname + '.png')
    img_copy = Image.open(fname_copy + '.png')
    assert np.all(np.asarray(img_orig) == np.asarray(img_copy))
    print('--')


if __name__ == '__main__':
    # fname = check_savefig()
    # reload(mpl)
    check_loadfig()

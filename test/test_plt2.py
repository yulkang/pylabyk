#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.

from pylabyk import plt2
import pathlib
import os
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
# from importlib import reload


def test_savefig() -> str:
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


def test_loadfig(fname: str = None):
    if fname is None:
        dir = pathlib.Path(__file__).parent.resolve()
        fname = os.path.join(dir, 'test_plt2/test_fig')
        # fname = os.path.join(dir, 'test_plt2/plt=traj_posterior+bi=0+bl=2+ct=.2+cw=.2+df=1.0+dp=1+dr=.75+dx=.04+eb=tknr+et=tknr+ig=2+iu=al+nb=1+nk=n+no=.99+ns=.15+nt=400+pc=goal+pn=from_prev_goal+pt=dense+rd=0.75+rf=2+s0=1948+sd=0+si=0+tp=.99+tr=.1+vf=40')

    fig_copy = plt2.loadfig(fname + '.mpl')
    fname_copy = fname + ' (2)'
    plt2.savefig(fname_copy + '.png', fig=fig_copy)

    img_orig = Image.open(fname + '.png')
    img_copy = Image.open(fname_copy + '.png')
    assert np.all(np.asarray(img_orig) == np.asarray(img_copy))
    print('--')


def test_consolidate_count_matrix():
    mat0 = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
            [100, 200, 300]
        ]
    )
    for group, mat1 in [
        ((0, 1, 2), mat0),
        ((0, 0, 1), np.array([
            [33, 33],
            [300, 300]
         ])),
        ((0, 1, 1), np.array([
            [1, 5],
            [110, 550]
         ])),
        ((0, 0, 0), np.array([[np.sum(mat0)]]))
    ]:
        assert np.all(
            plt2.consolidate_count_matrix(mat0, group=group) == mat1
        )
    print('passed all test_consolidate_count_matrix()')


def test_GridAxes():
    axs = plt2.GridAxes(
        nrows=2, ncols=3,
        widths=1, heights=1,
        left=0.5, top=0.5, bottom=0.5, right=0.5,
        wspace=0.25, hspace=0.25
    )
    axs_inner = plt2.GridAxes(
        nrows=2, ncols=3, top=3, bottom=0, left=0, right=0,
        widths=1, heights=1,
        wspace=1, hspace=1,
        parent=axs[0, 1]
    )
    for row in range(axs_inner.nrows):
        for col in range(axs_inner.ncols):
            plt.sca(axs_inner[row, col])
            plt.title(f'({row}, {col})')
    plt.show()

    axs = plt2.GridAxes(
        nrows=2, ncols=3,
        widths=1, heights=1,
        left=0.5, top=0.5, bottom=0.5, right=0.5,
        wspace=0.25, hspace=0.25
    )
    axs_inner = plt2.GridAxes(
        nrows=2, ncols=3, top=3, bottom=0, left=0, right=0,
        widths=1, heights=1,
        wspace=1, hspace=1,
        parent=axs[1:, 1:]
    )
    for row in range(axs_inner.nrows):
        for col in range(axs_inner.ncols):
            plt.sca(axs_inner[row, col])
            plt.title(f'({row}, {col})')
    plt.show()
    print('test_GridAxes')

def gridspec_within_gridspec():
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    # gridspec inside gridspec
    fig = plt.figure()

    gs0 = gridspec.GridSpec(1, 2, figure=fig)

    gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

    ax1 = fig.add_subplot(gs00[:-1, :])
    ax2 = fig.add_subplot(gs00[-1, :-1])
    ax3 = fig.add_subplot(gs00[-1, -1])

    # the following syntax does the same as the GridSpecFromSubplotSpec call above:
    gs01 = gs0[1].subgridspec(3, 3)

    ax4 = fig.add_subplot(gs01[:, :-1])
    ax5 = fig.add_subplot(gs01[:-1, -1])
    ax6 = fig.add_subplot(gs01[-1, -1])

    plt.suptitle("GridSpec Inside GridSpec")
    format_axes(fig)

    plt.show()


if __name__ == '__main__':
    # reload(mpl)
    # test_savefig()  # NOTE: should run test_savefig() and test_loadfig() on different runs of python to really test the persistence
    # test_loadfig()
    # test_consolidate_count_matrix()
    test_GridAxes()
    # gridspec_within_gridspec()
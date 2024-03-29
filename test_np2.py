#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.


import pytest
import numpy as np
import numpy.typing as nptyp
from pylabyk import np2


def test_vectorize_par():
    # noinspection PyTypeChecker
    o = np2.vectorize_par(
        np.ones,
        dict(
            dtype=np2.arrayobj1d([
                int, float
            ])[:, None],
            shape=np2.arrayobj1d([
                (2, 3),
                (4, 1),
            ])[None],
        ),
        otypes=np.ndarray,
    )  # type: nptyp.NDArray[np.ndarray]

    assert o[0, 0].shape == (2, 3)
    assert o[0, 1].shape == (4, 1)
    assert o[1, 0].shape == (2, 3)
    assert o[1, 1].shape == (4, 1)

    assert o[0, 0].dtype == int
    assert o[0, 1].shape == int
    assert o[1, 0].shape == float
    assert o[1, 1].shape == float


def test_index_arg():
    a0 = np.array([[0, 3, 0], [0, 0, 2], [1, 0, 0]])
    for a, res_correct in [
        (
            a0,
            np.amax(a0, -1)
        ),
        (
            a0[None],
            np.amax(a0, -1)[None]
        ),
        (
            a0[..., None],
            a0
        )
    ]:
        res = np2.index_arg(a, np.argmax(a, -1))
        assert res.shape == res_correct.shape
        assert np.all(res == res_correct)


def test_pmf_delta_aliased():
    for (center, xs) in [
        (0, np.linspace(-5, 3, 40)),
        (-5, np.linspace(-5, 3, 40)),
        (3, np.linspace(-5, 3, 40))
    ]:
        assert np2.issimilar(
            center, np2.mean_distrib(
                np2.pmf_delta_aliased(center, xs),
                xs
            ),
            verbose=True
        )


def test_pmf_boxcar_aliased():
    for (vmin, vmax, xs) in [
        (0, 1, np.linspace(-5, 3, 17)),
        (-5, 3, np.linspace(-5, 3, 17)),
        (-5, 0, np.linspace(-5, 3, 17)),
        (-4.25, -3.75, np.linspace(-5, 3, 17)),
        (-4.25, -3.25, np.linspace(-5, 3, 17)),
        (0, 3, np.linspace(-5, 3, 17))
    ]:
        assert np2.issimilar(
            (vmin + vmax) / 2, np2.mean_distrib(
                np2.pmf_boxcar_aliased(vmin, vmax, xs),
                xs
            ),
            verbose=True
        )


def test_ShortStrAttributes():
    class Analyses(np2.AliasStrAttributes):
        def __init__(self):
            self.fig2 = np2.AliasStr('Figure 2')
            self.fig1 = np2.AliasStr('Figure 1')
    ss = Analyses()

    # # class attributes don't invoke __getattribute__(),
    # # so cannot be used
    # class ss(np2.AliasStrAttributes):
    #     fig2 = np2.AliasStr('Figure 2')
    #     fig1 = np2.AliasStr('Figure 1')
    #
    # print(ss.fig1)

    assert ss.fig1 == 'fig1'
    assert ss.fig1.orig == 'Figure 1'
    assert ss.fig2 == 'fig2'
    assert ss.fig2.orig == 'Figure 2'
    assert tuple(ss.dict.keys()) == ('fig2', 'fig1')


def test_ccw():
    assert np2.ccw(
        np.array([[0, 0]]),
        np.array([[1, 1]]),
        np.array([[3, 3]])
    ) == 0, 'colinear must be 0'

    assert np2.ccw(
        np.array([[0, 0]]),
        np.array([[1, 1]]),
        np.array([[3, 4]])
    ) == 1, 'counterclockwise must be 1'

    assert np2.ccw(
        np.array([[0, 0]]),
        np.array([[1, 1]]),
        np.array([[3, -4]])
    ) == -1, 'clockwise must be -1'


def test_intersect():
    A = np.array([[0, 0]])
    B = np.array([[2, 2]])
    C = np.array([[2, 0]])
    D = np.array([[0, 2]])

    assert all(np2.intersect(A, B, C, D))
    assert all(np2.intersect(A, B, D, C))
    assert all(np2.intersect(B, A, C, D))
    assert all(np2.intersect(B, A, D, C))

    assert all(np2.intersect(A, B, A, C))
    assert all(np2.intersect(A, B, B, C))
    assert all(np2.intersect(A, C, C, D))
    assert all(np2.intersect(A, D, C, D))

    assert not all(np2.intersect(A, C, B, D))
    assert not all(np2.intersect(C, A, B, D))
    assert not all(np2.intersect(A, C, D, B))
    assert not all(np2.intersect(C, A, D, B))

    assert not all(np2.intersect(A, D, B, C))
    assert not all(np2.intersect(A, D, C, B))
    assert not all(np2.intersect(D, A, B, C))
    assert not all(np2.intersect(D, A, C, B))


def test_ttest_mc():
    pval, tstat, df = np2.ttest_mc(np.ones(100))
    assert pval < 0.1
    assert tstat == np.inf
    assert df == 99

    pval, tstat, df = np2.ttest_mc(np.ones(1))
    assert pval == 1.
    assert tstat == np.inf
    assert df == 0


if __name__ == '__main__':
    test_pmf_boxcar_aliased()
    test_pmf_delta_aliased()
    test_index_arg()
    test_ShortStrAttributes()
    test_ccw()
    test_intersect()
    test_ttest_mc()

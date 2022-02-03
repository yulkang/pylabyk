#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.


import pytest
import numpy as np
from lib.pylabyk import np2



def test_ShortStrAttributes():
    class Analyses(np2.ShortStrAttributes):
        def __init__(self):
            self.fig2 = np2.ShortStr('Figure 2')
            self.fig1 = np2.ShortStr('Figure 1')
    ss = Analyses()

    assert ss.fig1 == 'fig1'
    assert ss.fig1.long == 'Figure 1'
    assert ss.fig2 == 'fig2'
    assert ss.fig2.long == 'Figure 2'
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


if __name__ == '__main__':
    test_ShortStrAttributes()
    test_ccw()
    test_intersect()
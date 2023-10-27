"""
from https://gist.github.com/fasiha/6c331b158d4c40509bd180c5e64f7924
"""

import unittest
import numpy as np
import numpy.linalg as linalg


def makeUnit(x):
    """Normalize entire input to norm 1. Not what you want for 2D arrays!"""
    return x / linalg.norm(x)


def xParV(x, v):
    """Project x onto v. Result will be parallel to v."""
    # (x' * v / norm(v)) * v / norm(v)
    # = (x' * v) * v / norm(v)^2
    # = (x' * v) * v / (v' * v)
    return np.dot(x, v) / np.dot(v, v) * v


def xPerpV(x, v):
    """Component of x orthogonal to v. Result is perpendicular to v."""
    return x - xParV(x, v)


def xProjectV(x, v):
    """Project x onto v, returning parallel and perpendicular components
    >> d = xProject(x, v)
    >> np.allclose(d['par'] + d['perp'], x)
    True
    """
    par = xParV(x, v)
    perp = x - par
    return {'par': par, 'perp': perp}


def rotateAbout(v, center, theta):
    """Rotate vector a about vector b by theta radians."""
    # Thanks user MNKY at http://math.stackexchange.com/a/1432182/81266
    proj = xProjectV(v, center)
    w = np.cross(center, proj['perp'])
    return (proj['par'] +
            proj['perp'] * np.cos(theta) +
            linalg.norm(proj['perp']) * makeUnit(w) * np.sin(theta))


class TestRotation(unittest.TestCase):
    def test_parallel2D(self):
        x = np.array([0, 1.0])
        v = np.array([-1, 1.0])
        vhat = v / linalg.norm(v)
        expected = vhat * np.cos(np.pi / 4)
        self.assertTrue(np.allclose(expected, xParV(x, v)))

    def test_perp3D(self):
        x = np.array([1, 1, 1.0])
        v = np.array([1, 0, 0.0])
        expected = np.array([0, 1, 1.0])
        self.assertTrue(np.allclose(expected, xPerpV(x, v)))

    def test_rotate3D(self):
        toRotate = np.array([1, 1, 1.0])
        about = np.array([0, 1, 0.0])
        actual = rotateAbout(toRotate, about, np.pi / 180 * 90)
        expected = np.array([1, 1, -1.0])
        self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
    unittest.main()